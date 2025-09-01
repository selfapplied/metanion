"""
Gene style analyzer: overlap metrics, usefulness index, and splice suggestions.

Approach:
- Introspect genes from sprixel2.library()
- Build a lightweight behavioral surface for each gene:
  - required_arity
  - returns_callable (best-effort heuristic)
  - called_primitives (names of called functions from known libs)
  - lexical_tokens (filtered identifiers from source)
- Overlap = weighted Jaccard on primitives and tokens + return-shape match
- Usefulness index = (sum of overlaps to others) / surface_size
  where surface_size = 1 + |called_primitives| + 0.5*|lexical_tokens|
- Splice suggestions for pairs above a threshold
"""

from __future__ import annotations

from typing import Dict, Any, Callable, List, Tuple, Set
import inspect
import ast


def _safe_getsource(fn: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(fn)
    except Exception:
        return ""


def _required_arity(sig: inspect.Signature) -> int:
    n = 0
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if p.default is inspect._empty:
                n += 1
    return n


def _returns_callable_heuristic(src: str) -> bool:
    # crude: looks for nested def or lambda returning
    try:
        tree = ast.parse(src)
    except Exception:
        return False
    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Return(self, node: ast.Return):
            v = node.value
            if isinstance(v, ast.Lambda):
                self.found = True
            elif isinstance(v, ast.Name):
                # returning a local function name is common (factory)
                self.found = True
            self.generic_visit(node)
    f = Finder()
    f.visit(tree)
    return bool(f.found)


_KNOWN_PRIMITIVE_HINTS: Set[str] = {
    # sprixel primitives referenced across the codebase
    'reflect', 'reflect_wave', 'squeeze', 'forms', 'speak', 'bloom', 'fuse', 'dusk', 'neon', 'sea', 'solar',
}


def _collect_called_names(src: str) -> Set[str]:
    names: Set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return names
    class CallCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
            self.generic_visit(node)
    CallCollector().visit(tree)
    return names


def _collect_identifiers(src: str) -> Set[str]:
    toks: Set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return toks
    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            if len(node.id) >= 3 and node.id.isidentifier():
                toks.add(node.id)
    NameCollector().visit(tree)
    # filter out overly common words
    stop = {'return', 'for', 'in', 'if', 'else', 'True', 'False', 'None'}
    return {t for t in toks if t not in stop}


def analyze_gene_surface(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    src = _safe_getsource(fn)
    arity = _required_arity(sig)
    returns_callable = _returns_callable_heuristic(src)
    called = _collect_called_names(src)
    called_primitives = {n for n in called if n in _KNOWN_PRIMITIVE_HINTS}
    lexical_tokens = _collect_identifiers(src)
    return {
        'name': getattr(fn, '__name__', 'unknown'),
        'required_arity': arity,
        'returns_callable': returns_callable,
        'called_primitives': sorted(called_primitives),
        'lexical_tokens': sorted(lexical_tokens),
        'source_len': len(src),
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / float(u)


def overlap_score(s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, float]:
    calls1 = set(s1.get('called_primitives', []))
    calls2 = set(s2.get('called_primitives', []))
    toks1 = set(s1.get('lexical_tokens', []))
    toks2 = set(s2.get('lexical_tokens', []))
    call_j = _jaccard(calls1, calls2)
    tok_j = _jaccard(toks1, toks2)
    shape = 1.0 if bool(s1.get('returns_callable')) == bool(s2.get('returns_callable')) else 0.0
    # weight calls higher than tokens; shape as a bonus
    total = 0.6 * call_j + 0.3 * tok_j + 0.1 * shape
    return {'calls': call_j, 'tokens': tok_j, 'shape': shape, 'total': total}


def usefulness_index(surfaces: List[Dict[str, Any]]) -> Dict[str, float]:
    # Precompute totals; penalize large surfaces
    names = [s['name'] for s in surfaces]
    idx: Dict[str, float] = {}
    for i, si in enumerate(surfaces):
        sum_overlap = 0.0
        for j, sj in enumerate(surfaces):
            if i == j:
                continue
            sum_overlap += overlap_score(si, sj)['total']
        size = 1.0 + len(si.get('called_primitives', [])) + 0.5 * len(si.get('lexical_tokens', []))
        idx[names[i]] = sum_overlap / max(1.0, size)
    return idx


def suggest_splices(surfaces: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, si in enumerate(surfaces):
        for j in range(i + 1, len(surfaces)):
            sj = surfaces[j]
            sc = overlap_score(si, sj)
            if sc['total'] >= threshold:
                common_calls = sorted(set(si.get('called_primitives', [])) & set(sj.get('called_primitives', [])))
                common_tokens = sorted(set(si.get('lexical_tokens', [])) & set(sj.get('lexical_tokens', [])))
                action = 'extract_common_helper' if common_calls else 'delegate_to_base'
                out.append({
                    'a': si['name'],
                    'b': sj['name'],
                    'score': sc,
                    'common_calls': common_calls,
                    'common_tokens': common_tokens,
                    'suggested_action': action,
                })
    # sort by descending score
    out.sort(key=lambda d: d['score']['total'], reverse=True)
    return out


def analyze_all_genes() -> Dict[str, Any]:
    from sprixel2 import library
    reg = library()
    surfaces = [analyze_gene_surface(fn) for fn in reg.values()]
    return {
        'surfaces': surfaces,
        'usefulness': usefulness_index(surfaces),
        'splices': suggest_splices(surfaces),
    }


__all__ = [
    'analyze_gene_surface', 'overlap_score', 'usefulness_index', 'suggest_splices', 'analyze_all_genes'
]


