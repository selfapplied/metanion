"""
sprixel2: thin, minimal wrappers on sprixel

Carry-over operators with small names and gentle defaults.
"""

from typing import Callable, Iterable, Optional, Dict, Any, List, NamedTuple
from collections import namedtuple, Counter
import sys
import builtins
import pprint
import hashlib
import shutil
from itertools import chain, islice, cycle
import numpy as np

from sprixel import (
    splash,
    reflect_wave,
    squeeze,
    wire as wireframe,
    forms,
    signal,
    reflect,
    speak,
    bloom,
    fuse,
    dusk,
    neon,
    sea,
    solar,
)

from metanion import em

# Named tuples for DEFLATE operations
BlockMeta = namedtuple('BlockMeta', ['type', 'lit_len_code_lengths', 'distance_codes'])
EnhancedEvent = namedtuple('EnhancedEvent', ['original', 'quaternion', 'enhancement_type'])

# Type aliases for clarity
CodeLengths = Counter[int]
DistanceCodes = Counter[int]

# Common operators for counter operations
def normalize_counter(counter: Counter) -> Dict[int, float]:
    """Normalize Counter to probabilities"""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v/total for k, v in counter.items()}

def compute_entropy(counter: Counter) -> float:
    """Compute entropy from Counter"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count/total) * np.log2(count/total) for count in counter.values())

genes: Dict[str, Callable] = {}


@em("value: present")
def gene(fn: Callable[..., Any]) -> Callable[..., Any]:
    # Mark as recombinable (registry only; no attribute assignment)
    genes[fn.__name__] = fn
    return fn


@gene
@em("length: 1")
def tint(text: str, pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    out = "".join(splash(text, pal))
    label = getattr(pal, 'label', getattr(pal, '__name__', 'pal'))
    return out


@gene
@em("len: >=1")
def ripple(text: str, soft: bool = True, rows: int = 2) -> str:
    return reflect_wave(text, depth=6 if soft else 10, drift=1, quiet=0.5, amp=2.0, freq=0.12, fade=0.9, rows=rows)


@gene
@em("echo: present")
def mark(text: str, style: str = 'acetyl', pal=None) -> str:
    # This is a simple placeholder for now.
    # A real implementation would use the palette and style.
    return text


@gene
@em("lines: >=1")
def normalize(text: str, max_lines: int = 20, head: int = 6, tail: int = 10, keep_runs: int = 1) -> str:
    return squeeze(text, max_lines=max_lines, head=head, tail=tail, keep_runs=keep_runs)


@gene
@em("width: =5")
def frame(text: str, width: int) -> str:
    return wireframe(text, width)


@gene
@em("__call__: present")
def stage(*bands: tuple[int, Callable[[int], str]]) -> Callable[[int], str]:
    gate = forms(*bands)

    def at(width: int) -> str:
        return gate(width)(width)

    return at


@gene
@em("__call__: present")
def symmetry(motif: Iterable[str] | str, kind: str = "reflect") -> Callable[[int], str]:
    """Creates a repeating, symmetrical line of text."""
    # Ensure motif is a list for consistent processing
    m_list = list(motif)
    if not m_list:
        return lambda width: " " * width

    # Create the mirrored sequence for reflection
    mirrored_motif = list(chain(m_list, reversed(m_list[1:-1])))

    def at(width: int) -> str:
        if kind == "reflect":
            # Cycle through the mirrored motif to fill the width
            return "".join(list(islice(cycle(mirrored_motif), width)))
        else:
            # Just cycle through the original motif
            return "".join(list(islice(cycle(m_list), width)))

    return at


@gene
@em("len: =5")
def melt(a: str, b: str, width: int, t: float = 0.5, pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    return speak(a, b, width, t, pal)


@gene
@em("lines: >=1")
def fused(seed: str, width: int, lines: int = 12) -> str:
    g = fuse(seed)
    art = bloom(g)(width)
    return squeeze(art, max_lines=lines, head=max(3, lines // 3), tail=max(3, lines // 2))


@gene
@em("present: true")
def deflate_block_to_quaternion(block_meta: BlockMeta) -> Optional[np.ndarray]:
    """Convert DEFLATE block to quaternion using document 7's formulas"""
    if block_meta.type != 'dynamic' or not block_meta.lit_len_code_lengths:
        return None
    
    # Extract the 3D vector (x_b, y_b, z_b) from document 7
    x_b = compute_code_length_gradient(block_meta)
    y_b = compute_literal_entropy(block_meta)
    z_b = compute_distance_topology(block_meta)
    
    # Convert to unit quaternion: Q_b = (1/s, v_norm/s)
    v = np.array([x_b, y_b, z_b])
    v_norm = np.tanh(0.1 * (v - np.mean(v)))  # Center and scale
    s = np.sqrt(1 + np.linalg.norm(v_norm)**2)
    q = np.array([1/s, v_norm[0]/s, v_norm[1]/s, v_norm[2]/s])
    
    return q

@gene
def compute_code_length_gradient(block_meta: BlockMeta) -> float:
    """Compute x_b: Entropy-based gradient from code-length distribution"""
    code_lengths = block_meta.lit_len_code_lengths
    if len(code_lengths) < 2:
        return 0.0
    
    entropy = compute_entropy(code_lengths)
    return min(entropy / 8.0, 1.0)  # Normalize to [0,1]

@gene
def compute_literal_entropy(block_meta: BlockMeta) -> float:
    """Compute y_b: Entropy of literal/length distribution"""
    code_lengths = block_meta.lit_len_code_lengths
    if not code_lengths:
        return 0.0
    
    entropy = compute_entropy(code_lengths)
    return min(entropy / 8.0, 1.0)  # Normalize to [0,1]

@gene
def compute_distance_topology(block_meta: BlockMeta) -> float:
    """Compute z_b: Distance code complexity based on block type"""
    if block_meta.type == 'dynamic':
        return 0.8  # Dynamic blocks have complex distance patterns
    elif block_meta.type == 'static':
        return 0.5  # Static blocks have moderate patterns
    else:
        return 0.1  # Stored blocks have minimal patterns

@gene
def enhance_block_event(event_type: str, payload: BlockMeta) -> Optional[EnhancedEvent]:
    """Enhance block events with quaternion data"""
    if event_type == 'block':
        q = deflate_block_to_quaternion(payload)
        if q is not None:
            return EnhancedEvent(
                original=payload,
                quaternion=q,
                enhancement_type='quaternion_conversion'
            )
    return None

@gene
def compose_quaternion_sequence(quaternions: List[np.ndarray]) -> Optional[np.ndarray]:
    """Compose quaternions: Q_{1→k} = Q_1 · Q_2 ··· Q_k (safe)"""
    if len(quaternions) < 2:
        return quaternions[0] if quaternions else None
    
    # Start with first quaternion
    composition = quaternions[0]
    
    # Compose with each subsequent quaternion
    for q_i in quaternions[1:]:
        composition = quat_mul(composition, q_i)
    
    # Normalize to unit quaternion
    return quat_norm(composition)

# Import quaternion operations
from quaternion import quat_mul, quat_norm


# --- Friendly names and labeled mirrors -------------------------------------

def _center(text: str, n: int) -> str:
    s = text[:n]
    pad = max(0, n - len(s))
    left = pad // 2
    right = pad - left
    return (" " * left) + s + (" " * right)


def name_of(obj: Any) -> str:
    return getattr(obj, 'label', getattr(obj, '__name__', obj.__class__.__name__))


@gene
def tag(text: str, obj: Any, side: str = 'right', pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    label = f"‹{name_of(obj)}›"
    colored = "".join(splash(label, pal))
    if side == 'left':
        return f"{colored} {text}"
    return f"{text} {colored}"


@gene
def banner(obj: Any, width: int, pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    label = f"‹{name_of(obj)}›"
    line = _center(label, width)
    return "".join(splash(line, pal))


@gene
def mirror_of(fn: Callable[[int], str] | str, width: int, pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    head = banner(fn, width, pal)
    body = fn(width) if callable(fn) else str(fn)
    tail = reflect(body, depth=1, drift=0, quiet=0.6)
    return "\n".join([head, tail])


# --- Graceful tables ---------------------------------------------------------

def grid(rows: List[List[str]], widths: Optional[List[int]] = None, border: str = "│", sep: str = "│") -> str:
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    widths = widths or [0] * cols
    # compute widths by max content length per column
    for c in range(cols):
        w = max((len(r[c]) if c < len(r) else 0) for r in rows)
        widths[c] = max(widths[c], w)
    out: List[str] = []
    for r in rows:
        cells = []
        for i in range(cols):
            cell = r[i] if i < len(r) else ""
            cells.append(cell + " " * (widths[i] - len(cell)))
        out.append(f"{border} " + f" {sep} ".join(cells) + f" {border}")
    top = "╭" + "┄" * (len(out[0]) - 2) + "╮"
    bot = "╰" + "┄" * (len(out[0]) - 2) + "╯"
    return "\n".join([top] + out + [bot])


def dict_table(d: Dict[Any, Any], width: int) -> str:
    keys = list(d.keys())
    keys.sort(key=lambda k: str(k))
    rows: List[List[str]] = []
    # two-column table: key | value truncated to fit
    key_w = max(10, min(28, max((len(str(k)) for k in keys), default=10)))
    val_w = max(10, width - (key_w + 7))  # borders and spaces
    rows.append(["key", "value"])
    for k in keys:
        ks = str(k)[:key_w]
        vs = str(d[k])
        if len(vs) > val_w:
            vs = vs[:max(0, val_w - 1)] + "…"
        rows.append([ks, vs])
    return grid(rows)


def _ls_grid(items: List[str], width: int, padding: int = 2) -> str:
    if not items:
        return ""
    n = len(items)
    max_len = max(len(s) for s in items)
    col_w = max_len + padding
    cols = max(1, width // col_w)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    out: List[str] = []
    for r in range(rows):
        line_cells: List[str] = []
        for c in range(cols):
            idx = c * rows + r
            if idx < n:
                s = items[idx]
                pad = col_w - len(s)
                if c == cols - 1:
                    line_cells.append(s)
                else:
                    line_cells.append(s + " " * pad)
        out.append("".join(line_cells).rstrip())
    return "\n".join(out)


def dict_ls(d: Dict[Any, Any], width: int) -> str:
    keys = list(d.keys())
    keys.sort(key=lambda k: str(k))
    pairs: List[str] = []
    for k in keys:
        pairs.append(f"{k}: {d[k]}")
    return _ls_grid(pairs, width)


def render(obj: Any, width: Optional[int] = None) -> str:
    try:
        w = width or shutil.get_terminal_size().columns
    except Exception:
        w = 80
    if isinstance(obj, dict):
        return dict_ls(obj, max(40, min(120, w)))
    if callable(obj):
        return f"<{name_of(obj)}>"
    return str(obj)


def pretty(obj: Any, width: Optional[int] = None) -> Any:
    """Return a lightweight proxy whose str() uses our renderer."""
    class Pretty:
        __slots__ = ("_obj", "_width")
        def __init__(self, o: Any, w: Optional[int]):
            self._obj = o
            self._width = w
        def __str__(self) -> str:
            return render(self._obj, self._width)
        def __repr__(self) -> str:
            return self.__str__()
    return Pretty(obj, width)
def library() -> Dict[str, Callable]:
    """Return recombinable gene registry."""
    return dict(genes)


def mate(name_a: str, name_b: str, seed: str) -> Callable[..., Any]:
    """Deterministically choose between two genes by seed.
    (Lightweight recombination placeholder; extend to real crossovers later.)
    """
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).digest()
    pick = h[0] & 1
    chosen = genes[name_b] if pick and name_b in genes else genes.get(name_a, genes.get(name_b))
    return chosen if chosen is not None else (lambda *args, **kw: "")


def echo() -> Callable[[Any], None]:
    """Install a friendly REPL printer (displayhook)."""
    def show(value: Any) -> None:
        if value is None:
            return
        try:
            setattr(builtins, '_', value)
        except Exception:
            pass
        if hasattr(value, 'label'):
            print(f"<{getattr(value, 'label')}>")
            return
        if isinstance(value, dict):
            try:
                width = shutil.get_terminal_size().columns
            except Exception:
                width = 80
            print(dict_table(value, max(40, min(120, width))))
            return
        if callable(value):
            name = getattr(value, '__name__', value.__class__.__name__)
            print(f"<{name}>")
            return
        try:
            pprint.pprint(value)
        except Exception:
            print(str(value))
    sys.displayhook = show
    return show
__all__: List[str] = sorted(list(genes.keys()) + [
    "library", "mate", "echo", "dusk", "neon", "sea", "solar",
    "grid", "dict_table", "name_of", "banner", "tag", "mirror_of", "dict_ls", "render", "pretty",
    "mark"
])
