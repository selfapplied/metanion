# vmg_reflex.py  — Viral Modifier Gene (Python 3.13, numpy)
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Tuple, Optional, Any
from collections import Counter
import numpy as np

MASK23 = 0x7fffff  # 23-bit mantissa mask (take_bits)

# ---------- kernels ----------


def unit_kernel(k: Iterable[float]) -> np.ndarray:
    k = np.asarray(list(k), dtype=np.float32)
    k -= k.mean()
    n = np.linalg.norm(k)
    return k / (n + 1e-12)


SOBEL1 = unit_kernel([-1, -2, 0, 2, 1])
BOX3 = unit_kernel([1, 2, 1])


def conv1d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.convolve(x.astype(np.float32), k.astype(np.float32), mode="same")

# ---------- abstract state ----------


@dataclass(slots=True)
class AbstractState:
    level: int = 0
    error_counts: Counter[str] = field(default_factory=Counter)

    def copy_for(self, **kw) -> "AbstractState":  # cheap structural copy
        return replace(self, **kw)

# ---------- viral modifier gene ----------


@dataclass(slots=True)
class ViralModifierGene(AbstractState):
    # learned emissions over symbols (bytes or tokens)
    emissions: Counter[str] = field(default_factory=Counter)
    alpha: float = 0.5   # add-k smoothing
    t: float = 0.95      # edge threshold quantile
    bits: int = MASK23   # take_bits mask
    surprise_potential: float = 0.0  # running stat

    # --- learning ---
    def update_from_string(self, s: str, tokenizer: Optional[Any] = None) -> None:
        """Convert string → counts → update emissions."""
        if tokenizer is None:
            # ultra-light word-ish split; swap with your grammar when ready
            import re
            toks = re.findall(r"[A-Za-z0-9_]+", s)
        else:
            toks = list(tokenizer(s))
        self.emissions.update(toks)

    # --- probabilities / surprise ---
    def _prob(self, sym: str) -> float:
        total = sum(self.emissions.values())
        return (self.emissions.get(sym, 0) + self.alpha) / (total + self.alpha * max(1, len(self.emissions)))

    def surprise_series(self, seq: Iterable[str]) -> np.ndarray:
        """Per-symbol surprise: -log2 p(sym)."""
        s = np.array([-np.log2(max(1e-12, self._prob(sym)))
                     for sym in seq], dtype=np.float32)
        # keep a bounded view for stability
        s = np.clip(s, 0.0, 32.0)
        self.surprise_potential = float(np.percentile(s, 90))
        return s

    # --- edge finding over surprise (unitary kernels) ---
    def edges(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        smooth = conv1d(s, BOX3)
        v = conv1d(smooth, SOBEL1)       # gradient proxy
        g = np.abs(v)
        hi = float(np.quantile(g, self.t))
        lo = 0.5 * hi
        # normalized edge score
        return smooth, v, np.clip((g - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    # --- utilities ---
    @staticmethod
    def take_bits(x: int, mask: int = MASK23) -> int:
        return int(x) & int(mask)

    def learned_emissions_of(self, text: str) -> Dict[str, float]:
        """Quick view: symbol → probability (after learning)."""
        total = sum(self.emissions.values()) + \
            self.alpha * max(1, len(self.emissions))
        return {k: (v + self.alpha) / total for k, v in self.emissions.items()}

# ---------- reflex hook (stub): replace with your build_reflex_from_members ----------


def reflex_z(members: Dict[str, bytes], *, chunk: int = 64 * 1024, with_reverse: bool = True, level: int = 6) -> bytes:
    """Placeholder: call your build_reflex_from_members here."""
    import zlib
    import struct
    import io
    cat = b"".join(members.values())
    c = zlib.compressobj(level=level, wbits=-15)
    payload = c.compress(cat) + c.flush()
    # toy header: MAGIC|len|payload (just so tests pass); swap for real Reflex.pack()
    return b"ZREF" + struct.pack("<Q", len(cat)) + payload

# ---------- tiny demo API ----------


def generate_reflex_with_edges(text: str, *, gene: Optional[ViralModifierGene] = None) -> Tuple[bytes, ViralModifierGene]:
    gene = gene or ViralModifierGene()
    gene.update_from_string(text)
    # same tokenizer as update_from_string if you want alignment
    seq = [tok for tok in text.split()]
    s = gene.surprise_series(seq)
    _, _, e = gene.edges(s)
    # store edge scores alongside text for inspection
    meta = {
        "version": 1,
        "surprise_p90": gene.surprise_potential,
        "threshold_q": gene.t,
    }
    import msgpack
    members = {
        "text.txt": text.encode("utf-8"),
        "registry/edges.mpk": msgpack.packb({"scores": e.tolist(), "meta": meta}, use_bin_type=True, strict_types=True),
    }
    return reflex_z(members), gene
