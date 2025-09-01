"""
Rich data-weave grammar for interwoven, data-heavy text.

- Small, composable builder that converts records into a Grammar
- Encourages cross-field co-occurrence to weave meaning
- No IO side-effects; returns objects for immediate use
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import re

from genome import Grammar, Genome


_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(value: Any) -> List[str]:
    if value is None:
        return []
    s = str(value)
    toks = [t.lower() for t in _WORD.findall(s)]
    return toks


def build_data_weave_grammar(
    records: List[Dict[str, Any]],
    *,
    field_weights: Optional[Dict[str, float]] = None,
    id_field: Optional[str] = None,
) -> Grammar:
    """Build a Grammar from structured records.

    - field_weights: control per-field influence (default 1.0)
    - id_field: optional field used for crosslinks like id:XYZ
    """
    field_weights = field_weights or {}
    # accumulate in float maps first, then quantize to Counters[int]
    u_map: Dict[str, float] = defaultdict(float)  # type: ignore[assignment]
    b_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # type: ignore[assignment]

    for rec in records or []:
        # Gather tokens per field; include field label token to tag provenance
        field_to_tokens: Dict[str, List[str]] = {}
        for k, v in rec.items():
            tokens = _tokenize(v)
            if not tokens:
                continue
            label_tok = f"{k.lower()}:"
            toks = [label_tok] + tokens
            field_to_tokens[k] = toks

            w = float(field_weights.get(k, 1.0))
            for t in toks:
                u_map[t] += w
            for a, b in zip(toks, toks[1:]):
                b_map[a][b] += w

        if not field_to_tokens:
            continue

        # Interweave fields: connect first tokens between fields (bidirectional)
        items = list(field_to_tokens.items())
        for i in range(len(items)):
            ki, toks_i = items[i]
            head_i = toks_i[0]
            for j in range(i + 1, len(items)):
                kj, toks_j = items[j]
                head_j = toks_j[0]
                # equalize weight between fields
                wij = 0.5 * (float(field_weights.get(ki, 1.0)) + float(field_weights.get(kj, 1.0)))
                b_map[head_i][head_j] += wij
                b_map[head_j][head_i] += wij

        # Crosslink by id if present
        if id_field and id_field in rec:
            id_tok = f"id:{str(rec[id_field]).lower()}"
            u_map[id_tok] += 1.0
            for k, toks in field_to_tokens.items():
                w = float(field_weights.get(k, 1.0))
                # link label->id and id->label for navigation
                if toks:
                    lbl = toks[0]
                    b_map[lbl][id_tok] += 0.5 * w
                    b_map[id_tok][lbl] += 0.5 * w

    # quantize to Counters of ints
    unig = Counter({k: int(round(v)) for k, v in u_map.items() if int(round(v))})
    bigr: Dict[str, Counter] = defaultdict(Counter)
    for src, row in b_map.items():
        qrow = Counter({k: int(round(v)) for k, v in row.items() if int(round(v))})
        if qrow:
            bigr[src] = qrow
    return Grammar(unigram_counts=unig, bigram_counts=bigr)


def merge_into_genome(genome: Genome, grammar: Grammar, *, weight: float = 1.0) -> Genome:
    """Merge the produced grammar into an existing genome in-memory.

    - Weighted additive merge; returns the same genome for chaining
    """
    w = float(max(0.0, weight))
    if w <= 0.0:
        return genome
    for tok, n in grammar.unigram_counts.items():
        genome.grammar.unigram_counts[tok] += int(round(w * float(n)))
    for a, row in grammar.bigram_counts.items():
        gr = genome.grammar.bigram_counts[a]
        for b, n in row.items():
            gr[b] += int(round(w * float(n)))
    return genome


def outline_record(rec: Dict[str, Any]) -> str:
    """Render a compact, data-forward outline of a record.

    Example output:
      • title: Example
        - metric: 12.3 (unit)
        - tag: alpha, beta
      → link: id:abc123
    """
    parts: List[str] = []
    title = rec.get('title') or rec.get('name') or rec.get('id')
    if title is not None:
        parts.append(f"• title: {title}")
    for k, v in rec.items():
        if k in ('title', 'name', 'id'):
            continue
        s = str(v)
        if len(s) > 80:
            s = s[:77] + '…'
        parts.append(f"  - {k}: {s}")
    if 'id' in rec:
        parts.append(f"  → link: id:{str(rec['id']).lower()}")
    return "\n".join(parts)


__all__ = [
    # Resonant surface (preferred)
    'axes', 'weave', 'lattice', 'infuse', 'sketch',
    # Legacy names (kept for compatibility)
    'build_data_weave_grammar', 'merge_into_genome', 'outline_record',
    'parse_basis', 'build_tensor_grammar_from_basis',
]


# --- Literal basis → tensor-space grammar ------------------------------------

def parse_basis(lines: List[str], *, delimiter: str = ',') -> Dict[str, str]:
    """Parse CSV-like basis lines: axis,chars

    - Ignores empty lines and lines starting with '#'
    - Returns mapping axis -> literal character string (order matters)
    """
    out: Dict[str, str] = {}
    for raw in lines or []:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        parts = [p.strip() for p in s.split(delimiter)]
        if len(parts) < 2:
            continue
        axis, chars = parts[0], parts[1]
        out[axis] = chars
    return out


def build_tensor_grammar_from_basis(
    bases: Dict[str, str],
    *,
    chain: bool = True,
    pairwise: bool = True,
    include_vectors: bool = False,
) -> Grammar:
    """Create a Grammar from literal basis characters defining a tensor space.

    - chain: link adjacent characters along each axis (axis order → sequence)
    - pairwise: align axes by index and link corresponding positions both ways
    - include_vectors: mint combined "vector" tokens from aligned positions
    """
    # float accumulators → quantize to Counters at the end
    u_map: Dict[str, float] = defaultdict(float)  # type: ignore[assignment]
    b_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # type: ignore[assignment]

    axes: List[Tuple[str, str]] = [(k, v) for k, v in bases.items() if v]
    if not axes:
        return Grammar()

    # Per-axis tokens and chains
    for axis, chars in axes:
        labels = [f"{axis}:{ch}" for ch in chars]
        for t in labels:
            u_map[t] += 1.0
        if chain and len(labels) > 1:
            for a, b in zip(labels, labels[1:]):
                b_map[a][b] += 1.0

    # Pairwise alignment across axes (index-aligned mapping)
    if pairwise and len(axes) >= 2:
        for i in range(len(axes)):
            name_i, chars_i = axes[i]
            L_i = max(1, len(chars_i) - 1)
            for j in range(i + 1, len(axes)):
                name_j, chars_j = axes[j]
                L_j = max(1, len(chars_j) - 1)
                for idx_i, ch_i in enumerate(chars_i):
                    # aligned index in j
                    idx_j = int(round((idx_i / L_i) * L_j)) if L_i > 0 else 0
                    idx_j = max(0, min(idx_j, len(chars_j) - 1))
                    ch_j = chars_j[idx_j]
                    ti = f"{name_i}:{ch_i}"
                    tj = f"{name_j}:{ch_j}"
                    b_map[ti][tj] += 1.0
                    b_map[tj][ti] += 1.0
                    if include_vectors:
                        vec = f"vec:{name_i}={ch_i}|{name_j}={ch_j}"
                        u_map[vec] += 1.0
                        # light linkage from components to vector token
                        b_map[ti][vec] += 0.5
                        b_map[tj][vec] += 0.5

    # Quantize
    unig = Counter({k: int(round(v)) for k, v in u_map.items() if int(round(v))})
    bigr: Dict[str, Counter] = defaultdict(Counter)
    for src, row in b_map.items():
        qrow = Counter({k: int(round(v)) for k, v in row.items() if int(round(v))})
        if qrow:
            bigr[src] = qrow
    return Grammar(unigram_counts=unig, bigram_counts=bigr)


# --- Resonant aliases ---------------------------------------------------------

def axes(lines: List[str], delimiter: str = ',') -> Dict[str, str]:
    return parse_basis(lines, delimiter=delimiter)


def weave(records: List[Dict[str, Any]], field_weights: Optional[Dict[str, float]] = None, id_field: Optional[str] = None) -> Grammar:
    return build_data_weave_grammar(records, field_weights=field_weights, id_field=id_field)


def lattice(bases: Dict[str, str], chain: bool = True, pairwise: bool = True, include_vectors: bool = False) -> Grammar:
    return build_tensor_grammar_from_basis(bases, chain=chain, pairwise=pairwise, include_vectors=include_vectors)


def infuse(genome: Genome, grammar: Grammar, weight: float = 1.0) -> Genome:
    return merge_into_genome(genome, grammar, weight=weight)


def sketch(rec: Dict[str, Any]) -> str:
    return outline_record(rec)



