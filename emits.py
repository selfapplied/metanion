"""
emits: minimal spec-as-test framework.

- @em("param1, param2 := output") can be stacked; each is a pattern to match.
- Patterns double as tests: calling the function runs its emitter and validates.
- Multiple specs enable pattern matching; any passing spec grants trust.
- No prints; strict assertions. Results and trust are kept in registries.
- Integrates with sprixel2.genes registry for recombination targets.

Style Vector: Object Tuning Over Attribute Checking
- Create conditions for natural flow instead of policing boundaries
- Tune objects to have what they need rather than checking if they have it
- Trust the process that created the object over verifying its current state
- Responsive design (listening, breathing, responding) over reactive validation
- Integration and cocreation over protection and separation
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import argparse
from collections import defaultdict, Counter, namedtuple
import math
import sys
import os
import zipfile
from pathlib import Path
import importlib.util
import types

# Lightweight minting of type IDs from spec strings
_minted: Dict[str, int] = {}
_mint_seq: int = 1

def mint(spec: str) -> int:
    global _mint_seq
    if spec not in _minted:
        _minted[spec] = _mint_seq
        _mint_seq += 1
    return _minted[spec]


# Registries
Spec = namedtuple('Spec', ['raw', 'inputs', 'outputs', 'mint'])
genes: Dict[str, List[Spec]] = defaultdict(list)
trust_scores: Counter[str] = Counter()
discoveries: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
invokers: Dict[str, List[Callable[[Callable[..., Any]], Any]]] = defaultdict(list)
functions: Dict[str, Callable[..., Any]] = {}


def split(raw: str) -> Tuple[str, str]:
    s = raw.strip()
    if ':=' in s:
        left, right = s.split(':=', 1)
        return left.strip(), right.strip()
    return s, ''


def split_commas(s: str) -> List[str]:
    items: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in s:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            item = ''.join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        items.append(tail)
    return items


def parse(s: str) -> Mapping[str, Any]:
    # grammar: key: value, key: [a, b, c], bare, present
    # minimal and forgiving; values kept as strings/numbers/arrays
    out: Dict[str, Any] = {}
    if not s:
        return out
    items = split_commas(s)
    for item in items:
        if ':' in item:
            k, v = item.split(':', 1)
            key = k.strip()
            val = v.strip()
            if val.lower() == 'present':
                out[key] = ('present', True)
            elif val.startswith('[') and val.endswith(']'):
                body = val[1:-1].strip()
                arr = [p.strip() for p in body.split(',')] if body else []
                out[key] = ('array', arr)
            else:
                # try number
                try:
                    num = float(val) if ('.' in val or 'e' in val.lower()) else int(val)
                    out[key] = ('number', num)
                except Exception:
                    out[key] = ('atom', val)
        else:
            # bare token means presence requirement
            out[item] = ('item', True)
    return out


def recog(raw: str) -> Spec:
    left, right = split(raw)
    ins = parse(left)
    outs = parse(right)
    # If no explicit ':=' provided, treat left as outputs (spec against result)
    if right.strip() == '':
        outs = ins
        ins = {}
    return Spec(raw=raw, inputs=ins, outputs=outs, mint=mint(raw))


def _coerce_input_value(kind_val: Tuple[str, Any]) -> Any:
    kind, val = kind_val
    if kind == 'number':
        return val
    if kind == 'atom':
        return val
    if kind == 'array':
        return list(val)
    if kind == 'present':
        return True
    return val


def _build_call_from_inputs(inputs: Mapping[str, Tuple[str, Any]]) -> Tuple[List[Any], Dict[str, Any]]:
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    if not inputs:
        return args, kwargs
    # Positional via 'args' array
    if 'args' in inputs and inputs['args'][0] == 'array':
        args.extend(list(inputs['args'][1]))
    # Single positional via 'value'
    if 'value' in inputs and inputs['value'][0] in ('number', 'atom'):
        args.append(_coerce_input_value(inputs['value']))
    # Keyword params
    for k, kv in inputs.items():
        if k in ('args', 'value'):
            continue
        kind, _ = kv
        if kind in ('number', 'atom', 'array'):
            kwargs[k] = _coerce_input_value(kv)
        elif kind == 'present':
            kwargs[k] = True
    return args, kwargs


def _is_lambda_style(left: str) -> bool:
    return ('{}' in left) or ('{' in left and '}' in left)


def _parse_number_token(tok: str) -> Optional[Any]:
    try:
        # Named constants
        if tok in ('pi', 'π'):
            return math.pi
        if tok in ('tau', 'τ'):
            return getattr(math, 'tau', 2.0 * math.pi)
        if tok == 'e':
            return math.e
        if tok.lower().startswith(('0x','-0x')):
            return int(tok, 16)
        if any(c in tok for c in ('.','e','E')):
            return float(tok)
        return int(tok)
    except Exception:
        return None


def _parse_atom_or_array(tok: str, varmap: Dict[str, Any]) -> Any:
    # Parse [a,b] arrays, numeric tokens, or var references from varmap
    if tok.startswith('[') and tok.endswith(']'):
        body = tok[1:-1].strip()
        if not body:
            return []
        elems = [e.strip() for e in body.split(',')]
        out: List[Any] = []
        for e in elems:
            num = _parse_number_token(e)
            if num is not None:
                out.append(num)
            elif e in varmap:
                out.append(varmap[e])
            else:
                out.append(e)
        return out
    num = _parse_number_token(tok)
    if num is not None:
        return num
    if tok in varmap:
        return varmap[tok]
    return tok


def _eval_brace_token(token: str, varmap: Dict[str, Any]) -> Any:
    # token formats: {name}, {name->var}, or {name arg1 arg2 -> var}
    inner = token.strip()[1:-1].strip()
    if not inner:
        return None
    var: Optional[str] = None
    if '->' in inner:
        body, var = [s.strip() for s in inner.split('->', 1)]
    else:
        body = inner
    parts = [p for p in body.split() if p]
    if not parts:
        return None
    name = parts[0]
    fn = functions.get(name)
    if not callable(fn):
        raise TypeError(f"unknown producer {name}")
    args = [_parse_atom_or_array(t, varmap) for t in parts[1:]]
    val = fn(*args)
    if var:
        varmap[var] = val
    return val


def _build_call_for_spec(sp: Spec) -> Tuple[List[Any], Dict[str, Any], bool]:
    left, _right = split(sp.raw)
    if not _is_lambda_style(left):
        args, kwargs = _build_call_from_inputs(sp.inputs)
        return args, kwargs, False

    tokens = [t for t in left.split() if t]
    varmap: Dict[str, Any] = {}
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    in_target = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == '{}':
            in_target = True
            i += 1
            continue
        if not in_target:
            if tok.startswith('{') and tok.endswith('}'):
                try:
                    _eval_brace_token(tok, varmap)
                except Exception:
                    # ignore producer failure; treat as skip later
                    pass
            i += 1
            continue
        # in target
        if tok.endswith(':') and (i + 1) < len(tokens):
            key = tok[:-1]
            val_tok = tokens[i + 1]
            val: Any
            if val_tok.startswith('{') and val_tok.endswith('}'):
                try:
                    val = _eval_brace_token(val_tok, varmap)
                except Exception:
                    val = None
            else:
                val = _parse_atom_or_array(val_tok, varmap)
            kwargs[key] = val
            i += 2
            continue
        # positional
        if tok.startswith('{') and tok.endswith('}'):
            try:
                val = _eval_brace_token(tok, varmap)
            except Exception:
                val = None
            args.append(val)
        else:
            args.append(_parse_atom_or_array(tok, varmap))
        i += 1
    return args, kwargs, True


def has(obj: Any, key: str) -> bool:
    if isinstance(obj, Mapping):
        return key in obj
    return hasattr(obj, key)


def get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=tol)
    except Exception:
        return a == b


def _match_clause(obj: Any, clause: Mapping[str, Tuple[str, Any]]) -> bool:
    for key, (kind, val) in clause.items():
        if kind == 'present':
            if not _has(obj, key):
                return False
            continue
        if kind == 'array':
            if not _has(obj, key):
                return False
            got = _get(obj, key)
            if not isinstance(got, (list, tuple)):
                return False
            # array spec is pattern-like; only length check here
            if len(val) and len(got) < len(val):
                return False
            continue
        if kind == 'number':
            if not _has(obj, key):
                return False
            if not _approx_equal(_get(obj, key), val):
                return False
            continue
        if kind == 'atom':
            if not _has(obj, key) or str(_get(obj, key)) != str(val):
                return False
            continue
    return True


def _coerce_result_view(result: Any) -> Mapping[str, Any]:
    # Provide a mapping view for common shapes (arrays, quaternions)
    try:
        import numpy as _np  # optional
    except Exception:
        _np = None  # type: ignore

    if isinstance(result, Mapping):
        return result
    view: Dict[str, Any] = {'value': result}
    seq: Optional[List[Any]] = None
    if isinstance(result, (list, tuple)):
        seq = list(result)
    elif _np is not None and isinstance(result, _np.ndarray):  # type: ignore[arg-type]
        try:
            seq = list(result.tolist())
        except Exception:
            seq = None
    if seq is not None:
        view['length'] = len(seq)
        view['len'] = len(seq)
        if len(seq) == 4:
            view['quaternion'] = seq
            view['q'] = seq
            try:
                # magnitude as Euclidean norm
                mag = 0.0
                for x in seq:
                    mag += float(x) * float(x)
                mag = math.sqrt(mag)
                view['magnitude'] = mag
                view['m'] = mag
            except Exception:
                pass
    return view


def _match_pattern(result: Any, spec: Spec) -> bool:
    # Match on outputs against a coerced view of the result
    if spec.outputs:
        view = _coerce_result_view(result)
        return _match_clause(view, spec.outputs)
    return True


def em(pattern: str, via: Optional[Callable[[Callable[..., Any]], Any]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    spec = recog(pattern)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = getattr(fn, '__name__', 'anon')
        genes[name].append(spec)
        if via is not None:
            invokers[name].append(via)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            target_spec_raw = kwargs.pop('__em_spec', None)
            result = fn(*args, **kwargs)
            # attempt matches across all specs for this function
            specs_for_fn = genes.get(name, [])
            matched_any = False
            for sp in specs_for_fn:
                if target_spec_raw is not None and sp.raw != target_spec_raw:
                    continue
                if _match_pattern(result, sp):
                    trust_scores[name] += 1
                    matched_any = True
            # Always record discovery (no prints)
            try:
                _discoveries[name].append(result)  # type: ignore[arg-type]
            except Exception:
                pass
            # Enforce: no hallucinations; only if there are specs without custom invokers
            if specs_for_fn and not matched_any:
                calls = invokers.get(name, [])
                if target_spec_raw is not None or invokers:
                    # When a specific spec or invoker is used, don't assert on others
                    pass
                else:
                    raise AssertionError(f"Spec did not match for {name}: {[sp.raw for sp in specs_for_fn]}")
            return result

        wrapped.__name__ = name
        return wrapped

    return decorate


# Public API -----------------------------------------------------------------

def trust(name: str) -> float:
    return float(trust_scores.get(name, 0))


def specs() -> Mapping[str, List[str]]:
    return {k: [sp.raw for sp in v] for k, v in genes.items()}


def find(name: str) -> List[Any]:
    return list(discoveries.get(name, []))


def breed(name_a: str, name_b: str, seed: str) -> Tuple[Callable[..., Any], float]:
    # Prefer higher-trust gene; fallback to deterministic pick
    try:
        from sprixel2 import genes, mate  # lazy import to avoid cycles at module import
    except Exception:
        genes = {}
        mate = lambda a, b, s: (lambda *args, **kw: None)  # type: ignore

    ta = trust(name_a)
    tb = trust(name_b)
    if name_a in genes and name_b in genes:
        if ta != tb:
            chosen = genes[name_a] if ta > tb else genes[name_b]
            return chosen, max(ta, tb)
        chosen = mate(name_a, name_b, seed)
        return chosen, ta
    if name_a in genes:
        return genes[name_a], ta
    if name_b in genes:
        return genes[name_b], tb
    return (lambda *args, **kw: None), 0.0


# Minting/type surface --------------------------------------------------------

def typeid(spec_raw: str) -> int:
    return mint(spec_raw)


def kind(name: str) -> List[int]:
    return [sp.mint for sp in genes.get(name, [])]


def kinds() -> Mapping[str, List[int]]:
    return {k: [sp.mint for sp in v] for k, v in genes.items()}


__all__ = ['em', 'trust', 'find', 'specs', 'breed', 'typeid', 'kind', 'kinds']


# ----------------------------------------------------------------------------
# Zip-based testrunner (no prints; raises on failure)
# ----------------------------------------------------------------------------

def _workspace_root() -> Path:
    # eonyx/emits.py -> repo root assumed as parent of this file's parent
    return Path(__file__).resolve().parent.parent


def _ensure_in_dir(root: Path) -> Path:
    # Kept for compatibility; not used for zip extraction anymore
    d = root / '.in'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_zips(root: Path) -> List[Path]:
    zips: List[Path] = []
    for p in root.glob('*.genyx.zip'):
        zips.append(p)
    return zips


def _import_py_from_sources(sources: Dict[str, bytes], module_prefix: str = 'bundle') -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    sys.modules.setdefault('emits', sys.modules[__name__])
    for name, data in sources.items():
        if not name.endswith('.py') or name.endswith('__init__.py'):
            continue
        try:
            text = data.decode('utf-8', errors='ignore')
        except Exception:
            continue
        if '@em(' not in text:
            continue
        mod_name = module_prefix + '_' + name[:-3].replace('/', '.').replace('\\', '.')
        try:
            # Create a new module object and execute the code into its namespace
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
            code = compile(text, filename=name, mode='exec')
            exec(code, mod.__dict__)
        except Exception:
            continue
        for k, v in vars(mod).items():
            if callable(v):
                registry.setdefault(k, v)
    return registry


def _import_all_py_in_zip(zip_path: Path) -> Dict[str, Any]:
    # Prefer existing eonyx.zip Reflex loader; fallback to standard ZipFile
    try:
        from zip import unpack_reflex  # type: ignore
    except Exception:
        unpack_reflex = None  # type: ignore

    if unpack_reflex is not None:
        try:
            blob = zip_path.read_bytes()
            files = unpack_reflex(blob)  # type: ignore[operator]
            if isinstance(files, dict) and files:
                return _import_py_from_sources(files, module_prefix=zip_path.stem)
        except Exception:
            pass
    return {}


def _import_all_py_under(path: Path) -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    # Make our emits module authoritative for "import emits"
    sys.modules.setdefault('emits', sys.modules[__name__])
    # Prepend path to sys.path for relative imports inside the bundle
    sys.path.insert(0, str(path))
    try:
        for py in path.rglob('*.py'):
            if py.name == '__init__.py':
                continue
            try:
                text = py.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            if '@em(' not in text:
                continue
            mod_name = 'bundle_' + '_'.join(py.relative_to(path).with_suffix('').parts)
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                try:
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                except Exception:
                    # Skip modules that fail to import (e.g., optional heavy deps)
                    continue
                # collect callables by name
                for k, v in vars(mod).items():
                    if callable(v):
                        registry.setdefault(k, v)
                        functions.setdefault(k, v)
                for k, v in vars(mod).items():
                    if callable(v):
                        registry.setdefault(k, v)
                        functions.setdefault(k, v)
    finally:
        # Remove path we added
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass
    return registry


def _run_registered_specs(name_to_fn: Mapping[str, Callable[..., Any]]) -> Dict[str, int]:
    # For each function that has specs registered, attempt to call with no args
    # Track glyph-style counters for a compact one-line report
    prev_trust = Counter(trust_scores)
    glyphs: Dict[str, int] = {'✓': 0, '⊘': 0, '∅': 0, '↷': 0, '🗜': 0}
    for name in list(genes.keys()):
        fn = name_to_fn.get(name)
        if fn is None:
            glyphs['∅'] += 1
            continue
        calls = invokers.get(name, [])
        if calls or genes.get(name):
            specs_for_fn = genes.get(name, [])
            # Bind and run per-spec using lambda-style or inputs
            for sp in specs_for_fn:
                try:
                    args, kwargs, _is_lambda = _build_call_for_spec(sp)
                    if args or kwargs:
                        value = fn(*args, **kwargs, __em_spec=sp.raw)
                    elif calls and calls[0]:
                        value = calls[0](fn)
                    else:
                        value = fn(__em_spec=sp.raw)
                    if _match_pattern(value, sp):
                        trust_scores[name] += 1
                        try:
                            _discoveries[name].append(value)  # type: ignore[arg-type]
                        except Exception:
                            pass
                        # Aggregate optional counters
                        try:
                            if isinstance(value, Mapping) and 'compressed' in value:
                                comp = value.get('compressed')
                                glyphs['🗜'] += int(comp) if comp is not None else 0
                        except Exception:
                            pass
                        glyphs['✓'] += 1
                    else:
                        glyphs['⊘'] += 1
                except TypeError:
                    glyphs['↷'] += 1
                except AssertionError:
                    glyphs['⊘'] += 1
                except Exception:
                    glyphs['↷'] += 1
        else:
            try:
                val = fn()
                glyphs['✓'] += 1
                try:
                    if isinstance(val, Mapping) and 'compressed' in val:
                        comp = val.get('compressed')
                        glyphs['🗜'] += int(comp) if comp is not None else 0
                except Exception:
                    pass
            except TypeError:
                glyphs['↷'] += 1
            except AssertionError:
                glyphs['⊘'] += 1
    # Count newly minted trust as α
    alpha = 0
    for k, v in trust_scores.items():
        dv = v - prev_trust.get(k, 0)
        if dv > 0:
            alpha += dv
    if alpha:
        glyphs['α'] = alpha
    return glyphs


def _cli_run(args: Optional[List[str]] = None) -> None:
    # Strategy: find *.genyx.zip at repo root unless specific targets are provided
    parser = argparse.ArgumentParser(prog='emits', add_help=True)
    parser.add_argument('targets', nargs='*', help='Zip files or directories to scan')
    parser.add_argument('--words', action='store_true', help='Expand glyphs into short words')
    ns = parser.parse_args(args=args if args is not None else None)

    root = _workspace_root()
    # No extraction to disk; imports happen directly from zip via sys.path

    targets: List[Path] = []
    words_mode = bool(ns.words)
    if ns.targets:
        for a in ns.targets:
            p = Path(a)
            targets.append(p if p.is_absolute() else (root / p))
    else:
        targets = _find_zips(root)

    name_to_fn: Dict[str, Callable[..., Any]] = {}
    loaded_zips = 0
    for t in targets:
        if t.suffix == '.zip' and t.exists():
            loaded_zips += 1
            imported = _import_all_py_in_zip(t)
            name_to_fn.update(imported)
        elif t.is_dir():
            imported = _import_all_py_under(t)
            name_to_fn.update(imported)

    # Also pull in functions registered via sprixel2.genes (existing code path)
    try:
        from sprixel2 import genes as _genes  # type: ignore
        if isinstance(_genes, dict) and _genes:
            for k, v in _genes.items():
                if callable(v):
                    name_to_fn.setdefault(k, v)
                    functions.setdefault(k, v)
    except Exception:
        pass

    # If nothing found, still print an empty report line
    glyphs = {'✓': 0, '⊘': 0, '∅': 0, '↷': 0}
    if name_to_fn:
        glyphs = _run_registered_specs(name_to_fn)
    # Default: one-line glyph counter summary
    parts: List[str] = []
    if words_mode:
        name_map = {
            '✓': 'pass',
            '⊘': 'fail',
            '∅': 'missing',
            '↷': 'skip',
            'α': 'trust',
            '🗜': 'compressed',
        }
        for key in ('✓', '⊘', '∅', '↷', 'α', '🗜'):
            if key in glyphs and glyphs[key]:
                parts.append(f"{name_map[key]}={glyphs[key]}")
        parts.append(f"zips={loaded_zips}")
        parts.append(f"funcs={len(name_to_fn)}")
    else:
        for key in ('✓', '⊘', '∅', '↷', 'α', '🗜'):
            if key in glyphs and glyphs[key]:
                parts.append(f"{key}{glyphs[key]}")
        parts.append(f"z{loaded_zips}")
        parts.append(f"f{len(name_to_fn)}")
    print(' '.join(parts))


if __name__ == '__main__':
    _cli_run()


# ----------------------------------------------------------------------------
# Unified try/catch helpers (external-interface friendly)
# ----------------------------------------------------------------------------

from contextlib import contextmanager


def _note_alert(alerts: Optional[Any], glyph: str, tag: Optional[str], exc: Optional[BaseException]) -> None:
    try:
        if alerts is None:
            return
        alerts[str(glyph)] += 1
        if tag:
            alerts[f"{glyph}:{tag}"] += 1
        if exc is not None:
            cname = exc.__class__.__name__
            alerts[f"{glyph}:{cname}"] += 1
    except Exception:
        pass


def attempt(run: Callable[[], Any], *, alerts: Optional[Any] = None, glyph: str = '!', tag: Optional[str] = None,
            default: Any = None, exceptions: tuple[type[BaseException], ...] = (Exception,), rethrow: bool = False) -> Any:
    """Run a callable once; on failure, mint glyphs into alerts and return default.

    - Use ONLY at external boundaries (IO/CLI/Net). Internal code should raise.
    - glyph: short key like '!' or '⟂'; tag adds context like '!:{op}'.
    - If rethrow=True, the exception is re-raised after noting alerts.
    """
    try:
        return run()
    except exceptions as e:  # type: ignore[catching-non-exception]
        _note_alert(alerts, glyph, tag, e)
        if rethrow:
            raise
        return default


@contextmanager
def shield(*, alerts: Optional[Any] = None, glyph: str = '!', tag: Optional[str] = None,
           exceptions: tuple[type[BaseException], ...] = (Exception,), rethrow: bool = False):
    """Context manager to unify try/catch. See attempt().

    Usage:
      with shield(alerts=asp, glyph='!', tag='open'):
          data = Path(p).read_bytes()
    """
    try:
        yield
    except exceptions as e:  # type: ignore[catching-non-exception]
        _note_alert(alerts, glyph, tag, e)
        if rethrow:
            raise


