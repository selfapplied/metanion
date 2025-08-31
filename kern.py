# vmg_reflex_dena.py  — Python 3.13, stdlib only (no hashlib)
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Iterable, Optional, Any
from collections import Counter
import struct, io, zlib, time, re

# -------------------------- tiny stable hashing (no hashlib) -----------------
MASK64 = 0xFFFFFFFFFFFFFFFF
PHI64  = 0x9E3779B97F4A7C15  # 2^64 * golden ratio fraction

def mix64(x: int) -> int:
    # SplitMix64-ish mixer, non-crypto but stable
    x = (x + PHI64) & MASK64
    x ^= (x >> 30); x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= (x >> 27); x = (x * 0x94D049BB133111EB) & MASK64
    x ^= (x >> 31)
    return x & MASK64

def s_fingerprint(s: str) -> int:
    h = 0
    for b in s.encode('utf-8'):
        h = mix64((h + b) & MASK64)
    return h

def s_complement(s: str) -> int:
    # “s complement” = bitwise NOT of the stable fingerprint
    return (~s_fingerprint(s)) & MASK64

# ----------------------------- color phases (HSV-ish) ------------------------
def counts_to_color(counts: Counter[str]) -> Tuple[float, float, float]:
    bad = counts.get('ℋ', 0) + counts.get('⊘', 0) + counts.get('∞', 0)
    good = counts.get('✓', 0) + counts.get('α', 0)
    total = bad + good + sum(v for k, v in counts.items()
                             if k not in ('ℋ','⊘','∞','✓','α'))
    if total == 0: return (0.0, 0.0, 0.8)
    bad_r, good_r = bad/total, good/total
    hue = (0.33*good_r - 0.33*bad_r) % 1.0
    sat = min(1.0, bad_r)
    val = 0.5 + 0.5*good_r
    return (float(hue), float(sat), float(val))

def color_phase(hsv: Tuple[float,float,float]) -> str:
    h,s,v = hsv
    if s < 0.15 and v > 0.85: return "calm"
    if s > 0.6 and v > 0.6:
        return "hot" if h < 0.08 or h > 0.92 else "alert"
    if v < 0.3: return "dim"
    return "steady"

# ----------------------------- reflex (double-ended) -------------------------
MAGIC = b"DE1\0"  # 4B magic + NUL
LE = "<"          # little-endian

@dataclass(slots=True)
class Entry:
    name: str
    off: int
    length: int
    crc32: int

@dataclass(slots=True)
class ReflexDE:
    entries: List[Entry]
    total_len: int
    f_index: List[Tuple[int,int]]      # (uncomp_pos, comp_pos)
    r_index: List[Tuple[int,int]]      # mirrored index (optional)
    payload_f: bytes                   # raw deflate of concatenated tape
    payload_r: bytes                   # reverse raw deflate (may be empty)

    def pack(self) -> bytes:
        flags = 1 if self.payload_r else 0
        n = len(self.entries)
        buf = io.BytesIO()
        buf.write(MAGIC)
        buf.write(struct.pack(LE+"HIQ", flags, n, self.total_len))
        for e in self.entries:
            nb = e.name.encode("utf-8")
            buf.write(struct.pack(LE+"H", len(nb))); buf.write(nb)
            buf.write(struct.pack(LE+"QQI", e.off, e.length, e.crc32))
        def _windex(ix: List[Tuple[int,int]]):
            buf.write(struct.pack(LE+"I", len(ix)))
            for u,c in ix: buf.write(struct.pack(LE+"QQ", u, c))
        _windex(self.f_index)
        buf.write(struct.pack(LE+"Q", len(self.payload_f))); buf.write(self.payload_f)
        _windex(self.r_index)
        buf.write(struct.pack(LE+"Q", len(self.payload_r))); buf.write(self.payload_r)
        return buf.getvalue()

    @staticmethod
    def unpack(blob: bytes) -> "ReflexDE":
        br = io.BytesIO(blob)
        assert br.read(4) == MAGIC
        flags, n, total = struct.unpack(LE+"HIQ", br.read(struct.calcsize(LE+"HIQ")))
        ents: List[Entry] = []
        for _ in range(n):
            (nl,) = struct.unpack(LE+"H", br.read(2))
            name = br.read(nl).decode("utf-8")
            off, length, c32 = struct.unpack(LE+"QQI", br.read(struct.calcsize(LE+"QQI")))
            ents.append(Entry(name, off, length, c32))
        def _rindex() -> List[Tuple[int,int]]:
            (cnt,) = struct.unpack(LE+"I", br.read(4))
            out=[]; 
            for _ in range(cnt):
                u,c = struct.unpack(LE+"QQ", br.read(16)); out.append((u,c))
            return out
        f_ix = _rindex()
        (flen,) = struct.unpack(LE+"Q", br.read(8)); fpay = br.read(flen)
        r_ix = _rindex()
        (rlen,) = struct.unpack(LE+"Q", br.read(8)); rpay = br.read(rlen)
        if not (flags & 1): r_ix, rpay = [], b""
        return ReflexDE(ents, total, f_ix, r_ix, fpay, rpay)

# -------------------------- tape builder (solid forward+reverse) -------------
def _crc32(b: bytes) -> int: return zlib.crc32(b) & 0xFFFFFFFF

def _deflate_raw(b: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level=level, wbits=-15)
    return c.compress(b) + c.flush()

def build_reflex_double_ended(members: Dict[str, bytes],
                              *, chunk: int = 64*1024,
                              with_reverse: bool = True,
                              level: int = 6) -> bytes:
    # concat → forward index → reverse index (optional)
    names = list(members.keys())
    offs: Dict[str,int] = {}
    cat = io.BytesIO(); pos = 0; ents: List[Entry] = []
    for n in names:
        b = members[n]; offs[n] = pos; cat.write(b)
        ents.append(Entry(n, pos, len(b), _crc32(b))); pos += len(b)
    tape = cat.getvalue()
    # forward sparse index (uncompressed_pos → compressed_pos)
    f_ix: List[Tuple[int,int]] = []; comp = io.BytesIO()
    d = zlib.compressobj(level=level, wbits=-15); upos = 0
    for w in range(0, len(tape), chunk):
        if w > upos: comp.write(d.compress(tape[upos:w])); upos = w
        f_ix.append((w, comp.tell()))
    comp.write(d.compress(tape[upos:])); comp.write(d.flush())
    f_pay = comp.getvalue()
    # reverse
    r_ix: List[Tuple[int,int]] = []; r_pay = b""
    if with_reverse:
        rtape = tape[::-1]; comp2 = io.BytesIO()
        d2 = zlib.compressobj(level=level, wbits=-15); u2=0
        for w in range(0, len(rtape), chunk):
            if w > u2: comp2.write(d2.compress(rtape[u2:w])); u2 = w
            r_ix.append((w, comp2.tell()))
        comp2.write(d2.compress(rtape[u2:])); comp2.write(d2.flush())
        r_pay = comp2.getvalue()
    return ReflexDE(ents, len(tape), f_ix, r_ix, f_pay, r_pay).pack()

# ------------------------ Viral Modifier Gene (phenotype) --------------------
@dataclass(slots=True)
class VMGene:
    level: int = 0
    alpha: float = 0.5                # smoothing for emissions
    t_quantile: float = 0.95          # edge threshold
    emissions: Counter[str] = field(default_factory=Counter)
    error_counts: Counter[str] = field(default_factory=Counter)
    surprise_potential: float = 0.0

    # --- assessment context + phenotype ---
    def _create_assessment_context(self, text: str) -> Dict[str, Any]:
        toks = re.findall(r"[A-Za-z0-9_]+", text)
        c = Counter(toks)
        self.emissions.update(c)
        hsv = counts_to_color(c)
        return {
            "tokens": toks,
            "counts": c,
            "hsv": hsv,
            "phase": color_phase(hsv),
            "sig": f"{s_fingerprint(text) & 0xFFFFFFFF:08x}",
            "sig_comp": f"{s_complement(text) & 0xFFFFFFFF:08x}",
        }

    def _prob(self, sym: str) -> float:
        V = max(1, len(self.emissions)); N = sum(self.emissions.values())
        return (self.emissions.get(sym,0) + self.alpha) / (N + self.alpha*V)

    def _generate_pheno_only(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        toks: List[str] = ctx["tokens"]
        # per-token surprise (capped)
        surpr = [min(32.0, max(0.0, -float.__float__(__import__('math').log2(self._prob(t))))) for t in toks] if toks else []
        # simple edge picking: jumps in surprise above quantile
        if surpr:
            import statistics as stats
            q = sorted(abs((surpr[i] - surpr[i-1])) for i in range(1, len(surpr)))
            thr = q[int(max(0, min(len(q)-1, round(self.t_quantile*(len(q)-1)))))] if q else 0.0
            edges = [i for i in range(1, len(surpr)) if abs(surpr[i]-surpr[i-1]) >= thr]
            self.surprise_potential = float(stats.quantiles(surpr, n=10)[-1]) if len(surpr) >= 10 else (max(surpr) if surpr else 0.0)
        else:
            edges = []; thr = 0.0; self.surprise_potential = 0.0
        return {
            "hsv": ctx["hsv"], "phase": ctx["phase"],
            "edges": edges, "edge_threshold": thr, "surprise_p90": self.surprise_potential,
            "sig": ctx["sig"], "sig_comp": ctx["sig_comp"],
        }

    # --- branch cut: propose cut points (e.g., for dynamic blocks / DE waypoints)
    def branch_cut(self, text: str) -> List[int]:
        ctx = self._create_assessment_context(text)
        ph = self._generate_pheno_only(ctx)
        # declare branch cuts at high-contrast edges and around EOB-like sentinels
        cuts = set(ph["edges"])
        for m in re.finditer(r"\n{2,}|(?:\u0000|\u0001|\u0100)", text):  # crude paragraph/EOB-ish sentinels
            cuts.add(m.start())
        # ensure start/end guards
        return sorted({0, *cuts, len(text)})

# ------------------------ shared vocabulary → generative output --------------
def shared_vocabulary(texts: Iterable[str]) -> List[str]:
    token_sets = []
    for s in texts:
        toks = set(re.findall(r"[A-Za-z0-9_]+", s))
        if toks: token_sets.append(toks)
    if not token_sets: return []
    shared = set.intersection(*token_sets)
    # keep a modest, sorted vocabulary
    return sorted(shared)[:256]

def generative_output(shared_vocab: List[str], *, seed: Optional[int] = None, lines: int = 8) -> str:
    # tiny, deterministic “poem” from shared tokens using our mixer
    if not shared_vocab: return ""
    h = (seed if isinstance(seed, int) else 0) & MASK64
    out: List[str] = []
    for i in range(lines):
        h = mix64(h + i + 1)
        # pick 5 tokens per line via mixed indices
        toks = []
        for j in range(5):
            h = mix64(h + j + 7)
            idx = (h >> 11) % max(1, len(shared_vocab))
            toks.append(shared_vocab[idx])
        out.append(" ".join(toks))
    return "\n".join(out)

# ----------------------------- glue example ----------------------------------
def make_reflex_from_texts(files: Dict[str, str], *, with_reverse=True, chunk=64*1024) -> bytes:
    gene = VMGene(level=1)
    members: Dict[str, bytes] = {}
    for name, txt in files.items():
        ctx = gene._create_assessment_context(txt)
        ph  = gene._generate_pheno_only(ctx)
        meta = {
            "version": 1, "phase": ph["phase"], "hsv": ph["hsv"],
            "surprise_p90": ph["surprise_p90"], "sig": ph["sig"], "sig_comp": ph["sig_comp"],
            "created": int(time.time())
        }
        # store text and a tiny sidecar
        members[name] = txt.encode("utf-8")
        import msgpack
        members[f"registry/{name}.mpk"] = msgpack.packb(meta, use_bin_type=True, strict_types=True)
    # add a shared-vocab generative artifact
    vocab = shared_vocabulary(files.values())
    members["registry/shared.txt"] = generative_output(vocab).encode("utf-8")
    return build_reflex_double_ended(members, chunk=chunk, with_reverse=with_reverse, level=6)