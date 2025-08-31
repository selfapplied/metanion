#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any, cast
from collections import Counter
from collections import namedtuple
import struct
import io
import zlib
import zipfile
import re
import time
import numpy as np
from twinz import TwinzFSBuilder

# ---------------- small stable mixer (no hashlib) ----------------
MASK64 = 0xFFFFFFFFFFFFFFFF; PHI64 = 0x9E3779B97F4A7C15


def mix64(x: int) -> int:
    x = (x + PHI64) & MASK64
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= (x >> 31)
    return x & MASK64


def s_fingerprint(s: str) -> int:
    h = 0
    for b in s.encode('utf-8'):
        h = mix64((h + b) & MASK64)
    return h


def s_complement(s: str) -> int: return (~s_fingerprint(s)) & MASK64

def hsv_and_phase(counts: Counter[str]) -> Tuple[Tuple[float, float, float], str]:
    bad = counts.get('ℋ', 0)+counts.get('⊘', 0)+counts.get('∞', 0)
    good = counts.get('✓', 0)+counts.get('α', 0)
    tot = bad+good+sum(v for k, v in counts.items() if k not in ('ℋ','⊘','∞','✓','α'))
    if tot == 0:
        return (0.0, 0.0, 0.8), "steady"
    br, gr = bad/tot, good/tot
    h = (0.33*gr - 0.33*br) % 1.0; s = min(1.0, br); v = 0.5 + 0.5*gr
    phase = "calm" if (s < 0.15 and v > 0.85) else ("hot" if (s > 0.6 and v > 0.6 and (h < 0.08 or h > 0.92)) else ("alert" if (s > 0.6 and v > 0.6) else ("dim" if v < 0.3 else "steady")))
    return (float(h), float(s), float(v)), phase

def bit_stats(b: bytes) -> Tuple[float, float]:
    if not b:
        return 0.5, 0.0
    arr = np.frombuffer(b, dtype=np.uint8)
    p = float(np.bitwise_and(arr, 1).sum()) / max(1, int(arr.size))
    q = 1.0 - p
    ent = 0.0 if (p <= 0 or q <= 0) else float(-(p*np.log2(p)+q*np.log2(q)))
    return p, ent



@dataclass(slots=True)
class VMGene:
    level: int = 0
    alpha: float = 0.5            # add-k smoothing
    fade: float = 0.2             # 0→keep history, 1→trust new surprise; used in EMA
    t_quantile: float = 0.95      # edge threshold (quantile over |grad|)
    emissions: Counter[str] = field(default_factory=Counter)
    error_counts: Counter[str] = field(default_factory=Counter)
    surprise_ema: float = 0.0     # EMA of normalized surprise
    last_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_phase: str = "steady"
    last_bias: float = 0.5
    last_entropy: float = 0.0

    def update_from_text(self, text: str, preset: Optional[str] = None) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9_]+", text)
        c = Counter(toks)
        # phrase preset bias (shared vocabulary)
        if preset and preset in PHRASE_PRESETS:
            for w in PHRASE_PRESETS[preset]:
                c[w] += 1  # tiny nudge
        self.emissions.update(c)
        self.last_hsv, self.last_phase = hsv_and_phase(c)
        self.last_bias, self.last_entropy = bit_stats(text.encode('utf-8', errors='ignore'))
        return toks

    def _prob(self, sym: str) -> float:
        V = max(1, len(self.emissions))
        N = sum(self.emissions.values())
        return (self.emissions.get(sym, 0) + self.alpha) / (N + self.alpha * V)

    def surprise_series(self, tokens: Iterable[str]) -> np.ndarray:
        s = np.array([-np.log2(max(1e-12, self._prob(t)))
                     for t in tokens], dtype=np.float32)
        # normalize → tanh squash so “surp 1 → fade 0” is stable
        if s.size:
            mu, sigma = float(s.mean()), float(s.std() or 1.0)
            z = (s - mu) / sigma
            u = np.tanh(z)                       # [-1,1] scale
            # EMA: new = fade*u_mean + (1-fade)*old
            self.surprise_ema = float(
                self.fade * float(u.mean()) + (1.0 - self.fade) * self.surprise_ema)
            return u
        return s

    def edges(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if u.size == 0:
            return np.array([]), np.array([])
        # unit kernels
        K = np.array([-1, -2, 0, 2, 1], dtype=np.float32)
        K -= K.mean()
        K = K / (np.linalg.norm(K) + 1e-12)
        G = np.array([1, 2, 1], dtype=np.float32)
        G -= G.mean()
        G = G / (np.linalg.norm(G) + 1e-12)
        smooth = np.convolve(u, G, mode='same')
        grad = np.convolve(smooth, K, mode='same')
        g = np.abs(grad)
        hi = float(np.quantile(g, self.t_quantile))
        lo = 0.5 * hi
        mask = (g >= hi) | ((g >= lo) & np.concatenate(([False], g[1:] >= lo)))
        idx = np.nonzero(mask)[0]
        return idx, g

    # angle() removed (unused)


# ---------------- reflex (double-ended solid) --------------------
MAGIC = b"DE1\0"
LE = "<"


Entry = namedtuple("Entry", "name off length crc32")


ReflexDE = namedtuple("ReflexDE", "entries total_len f_index r_index payload_f payload_r")


def pack_reflex(reflex: ReflexDE) -> bytes:
    flags = 1 if reflex.payload_r else 0
    n = len(reflex.entries)
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack(LE+"HIQ", flags, n, reflex.total_len))
    for e in reflex.entries:
        nb = e.name.encode("utf-8")
        buf.write(struct.pack(LE+"H", len(nb)))
        buf.write(nb)
        buf.write(struct.pack(LE+"QQI", e.off, e.length, e.crc32))

    def _wix(ix: List[Tuple[int, int]]):
        buf.write(struct.pack(LE+"I", len(ix)))
        for u, c in ix:
            buf.write(struct.pack(LE+"QQ", u, c))
    _wix(reflex.f_index)
    buf.write(struct.pack(LE+"Q", len(reflex.payload_f)))
    buf.write(reflex.payload_f)
    _wix(reflex.r_index)
    buf.write(struct.pack(LE+"Q", len(reflex.payload_r)))
    buf.write(reflex.payload_r)
    return buf.getvalue()


def unpack_reflex(blob: bytes) -> ReflexDE:
    br = io.BytesIO(blob)
    assert br.read(4) == MAGIC
    flags, n, total = struct.unpack(LE+"HIQ", br.read(struct.calcsize(LE+"HIQ")))
    ents: List[Entry] = []
    for _ in range(n):
        (nl,) = struct.unpack(LE+"H", br.read(2))
        nm = br.read(nl).decode("utf-8")
        off, len_, c = struct.unpack(LE+"QQI", br.read(struct.calcsize(LE+"QQI")))
        ents.append(Entry(nm, off, len_, c))

    def _rix() -> List[Tuple[int, int]]:
        (cnt,) = struct.unpack(LE+"I", br.read(4))
        out: List[Tuple[int, int]] = []
        for _ in range(cnt):
            out.append(struct.unpack(LE+"QQ", br.read(16)))
        return out
    f_ix = _rix()
    (flen,) = struct.unpack(LE+"Q", br.read(8))
    fpay = br.read(flen)
    r_ix = _rix()
    (rlen,) = struct.unpack(LE+"Q", br.read(8))
    rpay = br.read(rlen)
    if not (flags & 1):
        r_ix, rpay = [], b""

    return ReflexDE(ents, total, f_ix, r_ix, fpay, rpay)


def _crc32(b: bytes) -> int: return zlib.crc32(b) & 0xFFFFFFFF


def _deflate_raw(b: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level=level, wbits=-15)
    return c.compress(b)+c.flush()


def build_reflex_double_ended(members: Dict[str, bytes], *, chunk: int = 64*1024,
                              with_reverse: bool = True, level: int = 6) -> bytes:
    names = sorted(members.keys())
    pos = 0
    tape = io.BytesIO()
    ents = []
    for n in names:
        data = members[n]
        tape.write(data)
        ents.append(Entry(n, pos, len(data), _crc32(data)))
        pos += len(data)
    U = tape.getvalue()
    # forward index & payload
    f_ix = []g
    comp = io.BytesIO()
    d = zlib.compressobj(level=level, wbits=-15)
    up = 0
    for w in range(0, len(U), chunk):
        if w > up:
            comp.write(d.compress(U[up:w]))
            up = w
        f_ix.append((w, comp.tell()))
    comp.write(d.compress(U[up:]))
    comp.write(d.flush())
    fpay = comp.getvalue()
    # reverse
    r_ix = []
    rpay = b""
    if with_reverse:
        R = U[::-1]
        comp2 = io.BytesIO()
        d2 = zlib.compressobj(level=level, wbits=-15)
        up2 = 0
        for w in range(0, len(R), chunk):
            if w > up2:
                comp2.write(d2.compress(R[up2:w]))
                up2 = w
            r_ix.append((w, comp2.tell()))
        comp2.write(d2.compress(R[up2:]))
        comp2.write(d2.flush())
        rpay = comp2.getvalue()
    return pack_reflex(ReflexDE(ents, len(U), f_ix, r_ix, fpay, rpay))

# ---------------- zip ingest → reflex + gene metadata ------------------------

def read_zip_members(zip_path: str) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            out[n] = zf.read(n)
    return out


def generate_reflex_from_zip(zip_path: str, *, preset: Optional[str] = None,
                             chunk: int = 64*1024, with_reverse: bool = True,
                             level: int = 6, add_twinz: bool = False,
                             twinz_window: int = 1024, twinz_K: int = 16,
                             twinz_q: float = 8.0, twinz_tail: int = 256) -> bytes:
    members = read_zip_members(zip_path)
    gene = VMGene(level=1)
    import msgpack
    meta: Dict[str, bytes] = {}
    for name, data in members.items():
        try:
            text = data.decode("utf-8")
        except Exception:
            continue
        u = gene.surprise_series(gene.update_from_text(text, preset=preset))
        idx, _ = gene.edges(u)
        Hdr = namedtuple("Hdr", "version phase hsv bitlsb_bias bitlsb_entropy surprise_ema edges_count sig sig_comp created")
        hdr = Hdr(1, gene.last_phase, [float(x) for x in gene.last_hsv],
                  gene.last_bias, gene.last_entropy, gene.surprise_ema, int(idx.size),
                  f"{s_fingerprint(text) & 0xFFFFFFFF:08x}", f"{s_complement(text) & 0xFFFFFFFF:08x}", int(time.time()))
        meta[f"registry/{name}.mpk"] = cast(bytes, msgpack.packb(hdr._asdict(), use_bin_type=True, strict_types=True))
    members.update(meta)
    if add_twinz:
        names = [n for n in sorted(members) if not n.startswith('registry/')]
        U = b"".join(members[n] for n in names)
        twinz = TwinzFSBuilder(window=twinz_window, K=twinz_K, q=twinz_q, tail=twinz_tail).build(U)
        members["registry/twinz_forward.mpk"] = cast(bytes, msgpack.packb(twinz._asdict(), use_bin_type=True, strict_types=True))
    return build_reflex_double_ended(members, chunk=chunk, with_reverse=with_reverse, level=level)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("vaez: viral convolution vessel")
    parser.add_argument('-p', nargs='?', default=None, help='Prefix for shifting into the right op')
    parser.add_argument('seed', nargs='?', default=None, help='Energy fed into engine.')
    parser.add_argument("dst", nargs='?', default=".vael.ce1", help='Name of the .ce1 files')
    parser.add_argument("ce1", nargs='*', default=None, help='Name of the .ce1 files')
    args = parser.parse_args()

    zips = Path(".").glob("*.ce1") if not args.ce1 else [Path(c) for c in args.ce1]
    dst  = Path(args.dst)
    
    for zp in zips:
        out = generate_reflex_from_zip(zp.as_posix(), preset="code")
        dst.write_bytes(out)
        print(dst.resolve().as_posix())
    