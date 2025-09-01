#!/usr/bin/env python3
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

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast
import inspect
import argparse
from collections import defaultdict, Counter, namedtuple
import math
import sys
import os
import numpy as np
from pathlib import Path
import types
from loader import import_zip, import_dir, import_paths, attempt, shield
from metanion import Spec, mint, recog, register_pending_specs
# Defer imports to break circular dependency
# import loom  
# from sprixel2 import gene

# Registries
genes: Dict[str, List[Spec]] = defaultdict(list)
trust_scores: Counter[str] = Counter()
discoveries: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
invokers: Dict[str, List[Callable[[Callable[..., Any]], Any]]] = defaultdict(list)
functions: Dict[str, Callable[..., Any]] = {}

# Lazy registration queue for specs from other modules
_pending_specs: List[Tuple[str, str, Callable[..., Any]]] = []  # (func_name, pattern, func)
_emits_ready = False


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
                # attempt numeric parse; default to atom
                num = attempt(
                    lambda: float(val) if (
                        '.' in val or 'e' in val.lower()) else int(val),
                    glyph='⟂', tag=key, default=None
                )
                if num is not None:
                    out[key] = ('number', num)
                else:
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


def in_val(kind_val: Tuple[str, Any]) -> Any:
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


def bind_args(inputs: Mapping[str, Tuple[str, Any]]) -> Tuple[List[Any], Dict[str, Any]]:
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    if not inputs:
        return args, kwargs
    if 'args' in inputs and inputs['args'][0] == 'array':
        args.extend(list(inputs['args'][1]))
    if 'value' in inputs and inputs['value'][0] in ('number', 'atom'):
        args.append(in_val(inputs['value']))
    for k, kv in inputs.items():
        if k in ('args', 'value'):
            continue
        kind, _ = kv
        if kind in ('number', 'atom', 'array'):
            kwargs[k] = in_val(kv)
        elif kind == 'present':
            kwargs[k] = True
    return args, kwargs


def is_lambda(left: str) -> bool:
    return ('{}' in left) or ('{' in left and '}' in left)


def read_num(tok: str) -> Optional[Any]:
    # Named constants
    if tok in ('pi', 'π'):
        return math.pi
    if tok in ('tau', 'τ'):
        return getattr(math, 'tau', 2.0 * math.pi)
    if tok == 'e':
        return math.e
    if tok.lower().startswith(('0x', '-0x')):
        return attempt(lambda: int(tok, 16), glyph='⟂', tag='hex', default=None)
    if any(c in tok for c in ('.', 'e', 'E')):
        return attempt(lambda: float(tok), glyph='⟂', tag='float', default=None)
    return attempt(lambda: int(tok), glyph='⟂', tag='int', default=None)


def read_atom(tok: str, varmap: Dict[str, Any]) -> Any:
    if tok.startswith('[') and tok.endswith(']'):
        body = tok[1:-1].strip()
        if not body:
            return []
        elems = [e.strip() for e in body.split(',')]
        out: List[Any] = []
        for e in elems:
            num = read_num(e)
            if num is not None:
                out.append(num)
            elif e in varmap:
                out.append(varmap[e])
            else:
                out.append(e)
        return out
    num = read_num(tok)
    if num is not None:
        return num
    if tok in varmap:
        return varmap[tok]
    return tok


def brace(token: str, varmap: Dict[str, Any]) -> Any:
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
    args = [read_atom(t, varmap) for t in parts[1:]]
    val = fn(*args)
    if var:
        varmap[var] = val
    return val


def plan(sp: Spec) -> Tuple[List[Any], Dict[str, Any], bool]:
    left, _right = split(sp.raw)
    if not is_lambda(left):
        args, kwargs = bind_args(sp.inputs)
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
                attempt(lambda: brace(tok, varmap), glyph='↷', tag='producer')
            i += 1
            continue
        if tok.endswith(':') and (i + 1) < len(tokens):
            key = tok[:-1]
            val_tok = tokens[i + 1]
            val: Any
            if val_tok.startswith('{') and val_tok.endswith('}'):
                val = attempt(lambda: brace(val_tok, varmap),
                              glyph='↷', tag='producer', default=None)
            else:
                val = read_atom(val_tok, varmap)
            kwargs[key] = val
            i += 2
            continue
        if tok.startswith('{') and tok.endswith('}'):
            val = attempt(lambda: brace(tok, varmap), glyph='↷',
                          tag='producer', default=None)
            args.append(val)
        else:
            args.append(read_atom(tok, varmap))
        i += 1
    return args, kwargs, True


def close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    fa = float(a)
    fb = float(b)
    return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=tol)


def match(obj: Any, clause: Mapping[str, Tuple[str, Any]]) -> bool:
    """Match object against clause specification using Style Vector principles.
    
    Style Vector: Object Tuning Over Attribute Checking
    - Trust the process that created the object
    - Work with data as-is, not converted
    - Create conditions for natural flow
    """
    obj = coerce(obj)
    for key, (kind, val) in clause.items():
        if kind == 'present':
            # Trust the object has what it needs - if not, it's a tuning issue
            if key not in obj:
                return False
            continue
        if kind == 'array':
            if key not in obj:
                return False
            got = obj[key]
            if not isinstance(got, (list, tuple)):
                return False
            # array spec is pattern-like; only length check here
            if len(val) and len(got) < len(val):
                return False
            continue
        if kind == 'number':
            if key not in obj:
                return False
            # Use the existing close function for approximate equality
            if not close(obj[key], val):
                return False
            continue
        if kind == 'atom':
            if key not in obj or str(obj[key]) != str(val):
                return False
            continue
    return True


def coerce(result: Any) -> Mapping[str, Any]:
    # Provide a mapping view for common shapes (arrays, quaternions)

    if isinstance(result, Mapping):
        return result
    view: Dict[str, Any] = {'value': result}
    seq: Optional[List[Any]] = None
    if isinstance(result, (list, tuple)):
        seq = list(result)
    elif np is not None and isinstance(result, np.ndarray):
        seq = attempt(lambda: list(result.tolist()),
                      glyph='⟂', tag='tolist', default=None)
    if seq is not None:
        view['length'] = len(seq)
        view['len'] = len(seq)
        if len(seq) == 4:
            view['quaternion'] = seq
            view['q'] = seq
            # magnitude as Euclidean norm

            def _mag() -> float:
                acc = 0.0
                for x in seq:
                    acc += float(x) * float(x)
                return math.sqrt(acc)
            mag = attempt(_mag, glyph='⟂', tag='magnitude', default=None)
            if mag is not None:
                view['magnitude'] = mag
                view['m'] = mag
    return view


def em(pattern: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    spec = recog(pattern)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = getattr(fn, '__name__', 'anon')
        if _emits_ready:
            # Emits is fully loaded, register immediately
            genes[name].append(spec)
        else:
            # Queue for later registration
            _pending_specs.append((name, pattern, fn))
        return fn

    return decorate


def _register_pending_specs():
    """Register all pending specs from other modules."""
    global _emits_ready
    _emits_ready = True
    
    for func_name, pattern, func in _pending_specs:
        spec = recog(pattern)
        genes[func_name].append(spec)
    
    _pending_specs.clear()


# Public API -----------------------------------------------------------------

def trust(name: str) -> float:
    return float(trust_scores.get(name, 0))


def specs() -> Mapping[str, List[str]]:
    return {k: [sp.raw for sp in v] for k, v in genes.items()}


def find(name: str) -> List[Any]:
    return list(discoveries.get(name, []))


def breed(name_a: str, name_b: str, seed: str) -> Tuple[Callable[..., Any], float]:
    # Prefer higher-trust gene; fallback to deterministic pick
    def _load():
        mod = __import__('sprixel2', fromlist=['genes', 'mate'])
        return getattr(mod, 'genes', {}), getattr(mod, 'mate', lambda a, b, s: (lambda *args, **kw: None))
    genes, mate = attempt(_load, glyph='∅', tag='sprixel2', default=(
        {}, lambda a, b, s: (lambda *args, **kw: None)))  # type: ignore

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


# Deferred import to break circular dependency
from sprixel2 import gene

@gene
@em(":= integrity_ok: bool")
def verify_codebase_integrity() -> bool:
    """
    Verifies that the live codebase structure matches the
    CE1 fingerprint embedded in loom.py.
    """
    # Deferred import to break circular dependency
    import loom
    # 1. Parse the embedded CE1 block from loom.py
    with open('loom.py', 'r') as f:
        content = f.read()

    import re
    docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
    if not docstring_match:
        raise ValueError("Could not find docstring in loom.py")

    docstring = docstring_match.group(1)
    ce1_block_match = re.search(r'CE1\{(.+?)\}', docstring, re.DOTALL)
    if not ce1_block_match:
        raise ValueError(
            "Could not find embedded CE1 block in loom.py docstring")

    embedded_ce1 = ce1_block_match.group(1)

    # Extract key metrics from embedded CE1
    matches_match = re.search(r'matches=(\d+)', embedded_ce1)
    energy_match = re.search(r'energy=([\d\.]+)', embedded_ce1)

    if not matches_match or not energy_match:
        raise ValueError("Embedded CE1 block is malformed")

    expected_matches = int(matches_match.group(1))
    expected_energy = float(energy_match.group(1))

    # 2. Generate a live CE1 block
    entries = loom.hilbert_walk('.', 10000)  # High energy budget to scan all
    live_metanion = loom.metanion_genesis(entries)
    live_ce1_block = loom.ce1_emission(live_metanion)

    # Extract key metrics from live CE1
    live_matches_match = re.search(r'matches=(\d+)', live_ce1_block)
    live_energy_match = re.search(r'energy=([\d\.]+)', live_ce1_block)

    if not live_matches_match or not live_energy_match:
        raise ValueError("Generated live CE1 block is malformed")

    live_matches = int(live_matches_match.group(1))
    live_energy = float(live_energy_match.group(1))

    # 3. Compare and verify
    matches_ok = (live_matches == expected_matches)
    # Allow for small float differences
    energy_ok = abs(live_energy - expected_energy) < 1.0

    if not matches_ok:
        print(
            f"Codebase integrity check failed: Mismatched file count. Expected {expected_matches}, got {live_matches}")

    if not energy_ok:
        print(
            f"Codebase integrity check failed: Mismatched energy. Expected {expected_energy}, got {live_energy}")

    return matches_ok and energy_ok


# Minting/type surface --------------------------------------------------------

def typeid(spec_raw: str) -> int:
    return mint(spec_raw)


def kind(name: str) -> List[int]:
    return [sp.mint for sp in genes.get(name, [])]


def kinds() -> Mapping[str, List[int]]:
    return {k: [sp.mint for sp in v] for k, v in genes.items()}


# Spec stacking and algebra removed from public surface to keep core minimal


__all__ = ['em', 'trust', 'find', 'specs', 'breed', 'typeid', 'kind', 'kinds']


# ----------------------------------------------------------------------------
# Zip-based testrunner (no prints; raises on failure)
# ----------------------------------------------------------------------------

# Canonical concept names and lexicons (glyphs/words) ---------------------------------
CANON_BY_GLYPH: Dict[str, str] = {
    '✓': 'pass',
    '⊘': 'fail',
    '∅': 'missing',
    '↷': 'skip',
    'α': 'trust',
    '🗜': 'compressed',
}

LEXICONS: Dict[str, Dict[str, str]] = {
    'glyph': {
        'pass': '✓',
        'fail': '⊘',
        'missing': '∅',
        'skip': '↷',
        'trust': 'α',
        'compressed': '🗜',
    },
    'word': {
        'pass': 'pass',
        'fail': 'fail',
        'missing': 'missing',
        'skip': 'skip',
        'trust': 'trust',
        'compressed': 'compressed',
    },
}


# Lexicon via grouped lists and quaternion connections -------------------------------
# File format (emits_lexicon.txt):
#   [group-name]
#   ✓ pass
#   ⊘ fail
#   ...

def _lexicon_path() -> Path:
    return Path(__file__).resolve().parent / 'emits_lexicon.txt'


def _load_lexicon_groups() -> List[List[str]]:
    p = _lexicon_path()
    text = attempt(lambda: p.read_text(encoding='utf-8'),
                   glyph='📖', tag='lexicon', default='')
    groups: List[List[str]] = []
    if not text:
        # Fallback default groups if file missing
        groups.append(['✓', 'pass'])
        groups.append(['⊘', 'fail'])
        groups.append(['∅', 'missing'])
        groups.append(['↷', 'skip'])
        groups.append(['α', 'trust'])
        groups.append(['🗜', 'compressed'])
        return groups
    current: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if current:
                groups.append(current)
                current = []
            continue
        if s.startswith('#'):
            continue
        if s.startswith('[') and s.endswith(']'):
            if current:
                groups.append(current)
                current = []
            continue
        parts = s.split()
        for tok in parts:
            current.append(tok)
    if current:
        groups.append(current)
    return [g for g in groups if g]


def _is_glyph_token(tok: str) -> bool:
    return len(tok) == 1 and not tok.isalnum()


def _group_quaternions(n_groups: int) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(max(1, n_groups)):
        theta = (2.0 * math.pi * i) / float(max(1, n_groups))
        half = 0.5 * theta
        w = math.cos(half)
        x = 0.0
        y = 0.0
        z = math.sin(half)
        out.append([w, x, y, z])
    return out


def render_summary_tokens(glyph_counts: Mapping[str, int], *, lexicon: str = 'glyph') -> List[str]:
    want_words = (lexicon == 'word')
    groups = _load_lexicon_groups()
    _ = _group_quaternions(len(groups))
    toks: List[str] = []
    for gkey in ('✓', '⊘', '∅', '↷', 'α', '🗜'):
        c = int(glyph_counts.get(gkey, 0)) if hasattr(
            glyph_counts, 'get') else 0
        if not c:
            continue
        gi = -1
        for idx, g in enumerate(groups):
            if gkey in g:
                gi = idx
                break
        label = gkey
        if gi >= 0:
            tokens = groups[gi]
            if want_words:
                for t in tokens:
                    if not _is_glyph_token(t):
                        label = t
                        break
                toks.append(f"{label}={c}")
            else:
                for t in tokens:
                    if _is_glyph_token(t):
                        label = t
                        break
                toks.append(f"{label}{c}")
        else:
            toks.append(f"{gkey}{c}" if not want_words else f"{gkey}={c}")
    return toks

def _workspace_root() -> Path:
    # eonyx/emits.py -> repo root assumed as parent of this file's parent
    return Path(__file__).resolve().parent


def _ensure_in_dir(root: Path) -> Path:
    # Kept for compatibility; not used for zip extraction anymore
    d = root / '.in'
    d.mkdir(parents=True, exist_ok=True)
    return d


# Removed: implicit scanning for any specific bundle extensions


def _import_py_from_sources(sources: Dict[str, bytes], module_prefix: str = 'bundle') -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    sys.modules.setdefault('emits', sys.modules[__name__])
    for name, data in sources.items():
        if not name.endswith('.py') or name.endswith('__init__.py'):
            continue
        text = attempt(lambda: data.decode(
            'utf-8', errors='ignore'), glyph='🔤', tag=name, default=None)
        if not text:
            continue
        if '@em(' not in text:
            continue
        mod_name = module_prefix + '_' + \
            name[:-3].replace('/', '.').replace('\\', '.')
        # Create a new module object and execute the code into its namespace
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod
        code = attempt(lambda: compile(text, filename=name,
                       mode='exec'), glyph='⟂', tag=mod_name, default=None)
        if code is None:
            continue
        ok = attempt(lambda: exec(code, mod.__dict__), glyph='⚡',
                     tag=mod_name, default=False) or True
        if not ok:
            continue
        for k, v in vars(mod).items():
            if callable(v):
                registry.setdefault(k, v)
    return registry


def _run_registered_specs(name_to_fn: Mapping[str, Callable[..., Any]]) -> Dict[str, int]:
    # Delegated to loader.run_specs for single source of truth
    from loader import run_specs as _runner
    return _runner(dict(name_to_fn))


def run_specs(name_to_fn: Mapping[str, Callable[..., Any]]) -> Dict[str, int]:
    return _run_registered_specs(name_to_fn)


def _cli_run(args: Optional[List[str]] = None) -> None:
    # Strategy: load explicit targets; default to current project directory
    parser = argparse.ArgumentParser(prog='emits', add_help=True)
    parser.add_argument('targets', nargs='*', help='Zip files or directories to scan')
    parser.add_argument('--words', action='store_true', help='Expand glyphs into short words')
    # no per-tool shuffle/seed; use eonyx seed if needed
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
        # Default: import the current project directory
        targets = [root]

    name_to_fn: Dict[str, Callable[..., Any]] = {}
    if targets:
        imported = import_paths(targets)
        name_to_fn.update(imported)

    # Also pull in functions registered via sprixel2.genes (existing code path)
    _genes = attempt(lambda: __import__('sprixel2', fromlist=[
                     'genes']).genes, glyph='∅', tag='sprixel2', default={})  # type: ignore
    if isinstance(_genes, dict) and _genes:
        for k, v in _genes.items():
            if callable(v):
                name_to_fn.setdefault(k, v)
                functions.setdefault(k, v)

    # Ensure producers are available to lambda-style specs
    for k, v in name_to_fn.items():
        if callable(v):
            functions.setdefault(k, v)

    # If nothing found, still print an empty report line
    glyphs = {'✓': 0, '⊘': 0, '∅': 0, '↷': 0}
    if name_to_fn:
        glyphs = _run_registered_specs(name_to_fn)
    # Default: one-line summary via lexicon tables
    parts: List[str] = render_summary_tokens(
        glyphs, lexicon=('word' if words_mode else 'glyph'))
    parts.append(
        f"funcs={len(name_to_fn)}" if words_mode else f"f{len(name_to_fn)}")
    print(' '.join(parts))


if __name__ == '__main__':
    _cli_run()

# Register any pending specs when emits is fully loaded
_register_pending_specs()




