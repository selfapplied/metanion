"""
Composable style algebra: positive/negative patterns and phase bridges.

Design goals:
- Small, resonant functions that compose naturally
- Uses existing primitives: reflect/squeeze (sprixel), resonance (loader)
- Descriptive outputs suitable for emits-style specs and registries
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any

from sprixel import reflect, squeeze
from loader import compute_resonance
from style import resolve_text_style


def measure_text_style(text: str) -> Dict[str, float]:
    """Measure simple style metrics on text.

    Metrics:
    - chars: character count
    - lines: line count
    - resonance: compression-based resonance proxy
    - mirror_delta: normalized difference in length vs reflected text
    """
    chars = float(len(text))
    lines = float(text.count('\n') + 1 if text else 0)
    try:
        r = float(compute_resonance(text.encode('utf-8', errors='ignore')))
    except Exception:
        r = 1.0
    try:
        mirrored = reflect(text, depth=1, drift=0, quiet=0.6)
        mlen = float(len(mirrored))
        mirror_delta = abs(chars - mlen) / max(1.0, max(chars, mlen))
    except Exception:
        mirror_delta = 1.0
    return {
        'chars': chars,
        'lines': lines,
        'resonance': r,
        'mirror_delta': mirror_delta,
    }


# --- Positive/Negative predicates -------------------------------------------

def positive_brevity(m: Dict[str, float]) -> bool:
    """Positive: keeps to concise fragments or high resonance per char."""
    if m['chars'] <= 240.0:
        return True
    density = m['resonance'] / max(1.0, m['chars'])
    return density >= 0.01


def positive_mirror_consistency(m: Dict[str, float]) -> bool:
    """Positive: mirrored text remains comparable in surface."""
    return m['mirror_delta'] <= 0.25


def negative_ramble(m: Dict[str, float]) -> bool:
    """Negative: long and low-resonance output."""
    return (m['chars'] > 600.0) and (m['resonance'] < 1.15)


def negative_unbalanced_mirror(m: Dict[str, float]) -> bool:
    """Negative: reflection diverges strongly from the original."""
    return m['mirror_delta'] > 0.5


# --- Phase bridges (transformations toward positives) -----------------------

def bridge_trim_chars(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def bridge_squeeze_lines(text: str, max_lines: int = 12) -> str:
    return squeeze(text, max_lines=max_lines, head=max(3, max_lines // 3), tail=max(3, max_lines // 2), keep_runs=1)


def bridge_mirror_balance(text: str) -> str:
    try:
        tail = reflect(text, depth=1, drift=0, quiet=0.6)
        return "\n".join([text, tail])
    except Exception:
        return text


# --- Algebra orchestrator ----------------------------------------------------

def evaluate_style(text: str) -> Dict[str, Any]:
    """Evaluate text against positive/negative style predicates."""
    m = measure_text_style(text)
    positives: List[str] = []
    negatives: List[str] = []
    if positive_brevity(m):
        positives.append('brevity')
    if positive_mirror_consistency(m):
        positives.append('mirror_consistency')
    if negative_ramble(m):
        negatives.append('ramble')
    if negative_unbalanced_mirror(m):
        negatives.append('unbalanced_mirror')
    score = float(len(positives)) - float(len(negatives))
    return {
        'metrics': m,
        'positives': positives,
        'negatives': negatives,
        'score': score,
    }


def apply_bridges(text: str, *, strict: bool = False) -> Dict[str, Any]:
    """Apply phase bridges to move text toward positive patterns.

    In strict mode, also resolve style from the global vector to choose length.
    """
    chosen_style, chosen_len = resolve_text_style(None, None)

    applied: List[str] = []
    out = text
    eval0 = evaluate_style(out)

    # Address ramble first
    if 'ramble' in eval0['negatives']:
        max_chars = int(chosen_len) if (strict and isinstance(chosen_len, int)) else 240
        out = bridge_trim_chars(out, max_chars=max_chars)
        applied.append('trim_chars')

    # Address mirror imbalance
    eval1 = evaluate_style(out)
    if 'unbalanced_mirror' in eval1['negatives']:
        out = bridge_mirror_balance(out)
        applied.append('mirror_balance')

    # Keep lines in check
    eval2 = evaluate_style(out)
    if eval2['metrics']['lines'] > 24:
        out = bridge_squeeze_lines(out, max_lines=12)
        applied.append('squeeze_lines')

    final_eval = evaluate_style(out)
    return {
        'text': out,
        'applied_bridges': applied,
        'before': eval0,
        'after': final_eval,
    }


__all__ = [
    'measure_text_style',
    'positive_brevity', 'positive_mirror_consistency',
    'negative_ramble', 'negative_unbalanced_mirror',
    'bridge_trim_chars', 'bridge_squeeze_lines', 'bridge_mirror_balance',
    'evaluate_style', 'apply_bridges',
]


