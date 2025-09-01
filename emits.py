#!/usr/bin/env python3
"""
emits: CE1 Mirror tracer + spec registry

CE1{
  lens=Mirror|mode=Bootstrap|Ξ=self:py3.13
  sense=[call,return,exc,import,mem]
  id=code(file,qual,hash)
  memory=append_only ledger±windowed
  reflect=top(paths,exc,drift)
  reflex=escalate(trace:on anomaly)
  emit=tracefile|stdout|genes
}

Notes
- Append-only, in-memory by default; optional plaintext file if explicitly enabled.
- Genes are registered from `metanion.pending` on import.
"""

from __future__ import annotations

import builtins
import inspect
import io
import os
import sys
import time
import threading
import traceback
import hashlib
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# Bootstrap imports (no circulars): rely on lightweight interfaces from metanion
import metanion as _meta


# --- Gene/spec registry -------------------------------------------------------

# Keep genes compatible with metanion.Spec instances
genes: Dict[str, List[_meta.Spec]] = {}
_registered_keys: set[Tuple[str, str]] = set()


def _register_pending_specs() -> None:
    """Move queued specs from metanion.pending into this module's genes."""
    for name, raw, fn in list(getattr(_meta, 'pending', [])):
        key = (name, raw)
        if key in _registered_keys:
            continue
        try:
            spec = _meta.recog(raw)
        except Exception:
            # Best-effort: skip malformed spec
            continue
        genes.setdefault(name, []).append(spec)
        _registered_keys.add(key)


# Eagerly register when module loads
_register_pending_specs()


# --- CE1 Mirror tracer --------------------------------------------------------

class _LedgerRow:
    __slots__ = (
        'ts', 'event', 'file', 'qual', 'line', 'func', 'code_hash', 'note'
    )

    def __init__(self, ts: float, event: str, file: str, qual: str,
                 line: int, func: str, code_hash: str, note: str = '') -> None:
        self.ts = ts
        self.event = event
        self.file = file
        self.qual = qual
        self.line = line
        self.func = func
        self.code_hash = code_hash
        self.note = note

    def to_plain(self) -> str:
        return f"{self.ts:.6f}\t{self.event}\t{self.file}\t{self.qual}\t{self.line}\t{self.func}\t{self.code_hash}\t{self.note}"


class CE1Mirror:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._lock = threading.RLock()
        self._ledger: List[_LedgerRow] = []
        self._stdout: bool = True
        self._tracefile: Optional[str] = None
        self._import_prev: Optional[Callable[..., Any]] = None
        self._counts: Counter[str] = Counter()
        self._exc_count: int = 0
        self._call_depth_by_thread: Dict[int, int] = defaultdict(int)
        self._scopes: List[str] = []

    # --- identity helpers
    @staticmethod
    def _qualname(frame) -> str:
        co = frame.f_code
        mod = frame.f_globals.get('__name__', '')
        cls = frame.f_locals.get(
            'self', None).__class__.__name__ if 'self' in frame.f_locals else None
        name = co.co_name
        if cls and name not in ('<module>', '<lambda>'):
            return f"{mod}.{cls}.{name}"
        return f"{mod}.{name}"

    @staticmethod
    def _code_hash(frame) -> str:
        co = frame.f_code
        h = hashlib.sha1()
        try:
            h.update(co.co_code)
            h.update(repr(co.co_consts).encode('utf-8'))
            h.update(repr(co.co_names).encode('utf-8'))
        except Exception:
            pass
        return h.hexdigest()[:16]

    def _row_from_frame(self, event: str, frame, note: str = '') -> _LedgerRow:
        ts = time.time()
        file = frame.f_code.co_filename
        line = frame.f_lineno
        qual = self._qualname(frame)
        code_hash = self._code_hash(frame)
        func = frame.f_code.co_name
        if self._scopes:
            note = f"[{self._scopes[-1]}] {note}" if note else f"[{self._scopes[-1]}]"
        return _LedgerRow(ts, event, file, qual, line, func, code_hash, note)

    # --- sinks
    def _emit(self, row: _LedgerRow) -> None:
        with self._lock:
            self._ledger.append(row)
            if self._stdout:
                print(row.to_plain())
            if self._tracefile:
                try:
                    # Plaintext append, one row per line
                    with open(self._tracefile, 'a', encoding='utf-8') as f:
                        f.write(row.to_plain() + "\n")
                except Exception:
                    pass

    # --- profile hooks
    def _trace(self, frame, event, arg):
        # Only track Python-level events
        if event not in ('call', 'return', 'exception'):
            return self._trace
        tid = threading.get_ident()
        if event == 'call':
            self._call_depth_by_thread[tid] += 1
            self._counts['call'] += 1
            self._emit(self._row_from_frame('call', frame))
        elif event == 'return':
            self._counts['return'] += 1
            self._emit(self._row_from_frame('return', frame))
            self._call_depth_by_thread[tid] = max(
                0, self._call_depth_by_thread[tid] - 1)
        elif event == 'exception':
            self._counts['exc'] += 1
            self._exc_count += 1
            note = ''
            try:
                etype, e, _tb = arg if isinstance(
                    arg, tuple) else (None, None, None)
                if etype is not None:
                    note = f"{getattr(etype, '__name__', str(etype))}: {e}"
            except Exception:
                pass
            self._emit(self._row_from_frame('exc', frame, note))
        return self._trace

    # --- import hook
    def _import_hook(self, name, globals=None, locals=None, fromlist=(), level=0):
        mod = None
        try:
            if self._import_prev is not None:
                mod = self._import_prev(name, globals, locals, fromlist, level)
            else:
                mod = builtins.__import__(
                    name, globals, locals, fromlist, level)
            frame = inspect.currentframe()
            if frame and frame.f_back:
                self._counts['import'] += 1
                self._emit(self._row_from_frame(
                    'import', frame.f_back, note=name))
            return mod
        except Exception as e:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                self._counts['exc'] += 1
                self._exc_count += 1
                self._emit(self._row_from_frame(
                    'exc', frame.f_back, note=f"import {name}: {e}"))
            raise

    # --- public API
    def enable(self, *, stdout: bool = True, tracefile: Optional[str] = None) -> None:
        if self._enabled:
            return
        self._stdout = bool(stdout)
        # Respect user preference: only write file if explicitly provided
        self._tracefile = str(tracefile) if tracefile else None
        sys.settrace(self._trace)
        self._import_prev = builtins.__import__
        builtins.__import__ = self._import_hook
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        sys.settrace(None)
        if self._import_prev is not None:
            builtins.__import__ = self._import_prev
        self._import_prev = None
        self._enabled = False

    def enabled(self) -> bool:
        return self._enabled

    # Append-only, windowed reads
    def window(self, n: int = 100) -> List[str]:
        with self._lock:
            rows = self._ledger[-max(0, int(n)):] if n else []
            return [r.to_plain() for r in rows]

    # Reflection: top(paths,exc,drift)
    def reflect(self, topk: int = 10) -> Dict[str, Any]:
        with self._lock:
            paths = Counter(r.file for r in self._ledger)
            funcs = Counter(
                r.qual for r in self._ledger if r.event in ('call', 'return'))
            excs = [r for r in self._ledger if r.event == 'exc']
            # drift = call - return per qual
            call_c = Counter(r.qual for r in self._ledger if r.event == 'call')
            ret_c = Counter(
                r.qual for r in self._ledger if r.event == 'return')
            drift = Counter()
            for q, c in call_c.items():
                drift[q] = c - ret_c.get(q, 0)
            return {
                'top_paths': paths.most_common(topk),
                'top_funcs': funcs.most_common(topk),
                'exceptions': [(r.qual, r.note) for r in excs[-topk:]],
                'drift': drift.most_common(topk),
            }

    # Reflex: escalate(trace:on anomaly)
    def escalate_on_anomaly(self, *, exc_threshold: int = 1) -> None:
        if self._exc_count >= int(exc_threshold) and not self._enabled:
            self.enable(stdout=True)

    # Manual mem event emission (hook point)
    def mem(self, note: str = '') -> None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            self._counts['mem'] += 1
            self._emit(self._row_from_frame('mem', frame.f_back, note=note))

    # Awareness helpers
    def where(self) -> Tuple[str, str, int]:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            f = frame.f_back
            return (f.f_code.co_filename, self._qualname(f), int(f.f_lineno))
        return ('', '', 0)

    def last(self, n: int = 1) -> List[str]:
        with self._lock:
            return [r.to_plain() for r in self._ledger[-max(0, int(n)):]]

    def push_scope(self, label: str) -> None:
        self._scopes.append(str(label))

    def pop_scope(self) -> None:
        if self._scopes:
            self._scopes.pop()


# Singleton tracer
ce1 = CE1Mirror()


def enable(stdout: bool = True, tracefile: Optional[str] = None) -> None:
    """Enable CE1 Mirror tracing."""
    ce1.enable(stdout=stdout, tracefile=tracefile)


def disable() -> None:
    ce1.disable()


def window(n: int = 100) -> List[str]:
    return ce1.window(n)


def reflect(topk: int = 10) -> Dict[str, Any]:
    return ce1.reflect(topk=topk)


def escalate_on_anomaly(exc_threshold: int = 1) -> None:
    ce1.escalate_on_anomaly(exc_threshold=exc_threshold)


def print_genes() -> None:
    """Print a concise view of registered genes/specs to stdout."""
    for name, specs in sorted(genes.items()):
        raws = [s.raw for s in specs]
        print(
            f"{name}: {len(specs)} specs :: {raws[:3]}{' …' if len(raws) > 3 else ''}")


class scope:
    """Context manager to tag events with a logical scope label."""

    def __init__(self, label: str):
        self._label = str(label)

    def __enter__(self):
        ce1.push_scope(self._label)

    def __exit__(self, exc_type, exc, tb):
        ce1.pop_scope()


# Optional: auto-enable if requested via env var (stdout only)
if os.environ.get('EONYX_CE1', '').lower() in ('1', 'true', 'yes', 'on'):
    try:
        ce1.enable(stdout=True, tracefile=None)
    except Exception:
        pass




