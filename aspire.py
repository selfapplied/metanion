from __future__ import annotations

import struct
from collections import Counter
from typing import Dict, List, Optional, Tuple, Iterable, Callable, Mapping, Any, Iterator, TypeAlias
import msgpack

# Stable 64-bit mixing
MASK64 = 0xFFFFFFFFFFFFFFFF
PHI64 = 0x9e3779b97f4a7c15


def mix64(x: int) -> int:
    x = (x + PHI64) & MASK64
    x ^= (x >> 30)
    x = (x * 0xbf58476d1ce4e5b9) & MASK64
    x ^= (x >> 27)
    x = (x * 0x94d049bb133111eb) & MASK64
    x ^= (x >> 31)
    return x & MASK64


def stable_str_hash(s: str) -> int:
    h = 0
    for b in s.encode('utf-8'):
        h = (h + b) & MASK64
        h = mix64(h)
    return h


def pack_qnan_with_payload(payload_51bits: int) -> float:
    payload = payload_51bits & ((1 << 51) - 1)
    bits = (0x7ff << 52) | (1 << 51) | payload
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def extract_qnan_payload(value: float) -> Optional[int]:
    q = struct.unpack('>Q', struct.pack('>d', float(value)))[0]
    exp = (q >> 52) & 0x7ff
    frac = q & ((1 << 52) - 1)
    if exp == 0x7ff and frac != 0:
        return frac & ((1 << 51) - 1)
    return None


## (deprecated) rewrite_glyphs/counts_to_glyphs removed; use Aspirate/ops/render instead


def counts_to_color(counts: Counter) -> Tuple[float, float, float]:
    bad = counts.get('ℋ', 0) + counts.get('⊘', 0) + counts.get('∞', 0)
    good = counts.get('✓', 0) + counts.get('α', 0)
    neutral = sum(v for k, v in counts.items() if k not in ('ℋ', '⊘', '∞', '✓', 'α'))
    total = bad + good + neutral
    if total == 0:
        return (0.0, 0.0, 0.8)
    s = min(1.0, bad / total)
    v = 0.5 + 0.5 * (good / max(1, total))
    hue = (0.33 * (good / max(1, total)) + 0.0 * (neutral / max(1, total)) + 0.0) % 1.0
    hue = (hue - 0.33 * (bad / max(1, total))) % 1.0
    return (float(hue), float(s), float(v))


############################
# Modular Counter operators #
############################

OpixOp: TypeAlias = Callable[[Mapping[str, float]], Dict[str, float]]


def apply_ops(counts: Mapping[str, float], ops: Optional[Iterable[OpixOp]]) -> Dict[str, float]:
    out: Dict[str, float] = dict(counts)
    if not ops:
        return out
    for op in ops:
        out = op(out)
    return out


def keep(glyphs: Iterable[str]) -> OpixOp:
    keep_set = set(glyphs)
    return lambda d: {g: n for g, n in d.items() if g in keep_set}


def drop(glyphs: Iterable[str]) -> OpixOp:
    drop_set = set(glyphs)
    return lambda d: {g: n for g, n in d.items() if g not in drop_set}


def rename(mapping: Mapping[str, str]) -> OpixOp:
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g, n in d.items():
            k = mapping.get(g, g)
            out[k] = out.get(k, 0) + n
        return out
    return f


def coalesce(groups: Mapping[str, Iterable[str]]) -> OpixOp:
    # groups: dest -> iterable of source keys to merge into dest
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = dict(d)
        for dst, srcs in groups.items():
            total = 0.0
            for s in srcs:
                total += out.pop(s, 0)
            if total:
                out[dst] = out.get(dst, 0) + total
        return out
    return f


def thresh(threshold: float) -> OpixOp:
    return lambda d: {g: n for g, n in d.items() if n >= threshold}


def topk(k: int) -> OpixOp:
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return {g: n for g, n in items}
    return f


def normalize() -> OpixOp:
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        total = float(sum(d.values())) or 1.0
        return {g: (n / total) for g, n in d.items()}
    return f


def mapto(map_fn: Callable[[str, float], Tuple[str, float]]) -> OpixOp:
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g, n in d.items():
            ng, nn = map_fn(g, n)
            if nn:
                out[ng] = out.get(ng, 0) + nn
        return out
    return f


def order(order: Iterable[str], append_rest: bool = True) -> OpixOp:
    order_list = list(order)
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        present = set(d.keys())
        seq: List[Tuple[str, float]] = []
        for g in order_list:
            if g in d:
                seq.append((g, d[g]))
                present.discard(g)
        if append_rest:
            seq.extend((g, d[g]) for g in d.keys() if g in present)
        return {g: n for g, n in seq}
    return f


def sort(key: Callable[[Tuple[str, float]], Any], reverse: bool = False) -> OpixOp:
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        items = sorted(d.items(), key=key, reverse=reverse)
        return {g: n for g, n in items}
    return f


def sort_by_count(ascending: bool = False) -> OpixOp:
    return sort(lambda kv: kv[1], reverse=not ascending)


def sort_by_glyph(ascending: bool = True) -> OpixOp:
    return sort(lambda kv: kv[0], reverse=not ascending)


def blit(
    mask: Optional[Callable[[str, float], bool]] = None,
    *,
    add: float = 0.0,
    mul: float = 1.0,
    set_to: Optional[float] = None,
    clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
    dst: Optional[Callable[[str], str]] = None,
    keep_rest: bool = True,
) -> OpixOp:
    """Vector-style masked projection over counts.

    - mask(g, n): select entries (default: all)
    - projection: set_to or (n*mul + add)
    - clip=(lo, hi): clamp after projection (None to skip bound)
    - dst(g): optional rename per selected entry
    - keep_rest: keep unselected entries unchanged
    """
    def f(d: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g, n in d.items():
            selected = True if mask is None else bool(mask(g, n))
            if selected:
                v = set_to if set_to is not None else (n * mul + add)
                if clip is not None:
                    lo, hi = clip
                    if lo is not None and v < lo:
                        v = lo
                    if hi is not None and v > hi:
                        v = hi
                k = dst(g) if dst is not None else g
                if v:
                    out[k] = out.get(k, 0) + v
            else:
                if keep_rest:
                    out[g] = out.get(g, 0) + n
        return out
    return f


# legacy rewrite_counter removed in favor of op pipeline


def counts_render(
        counts: Mapping[str, float],
        *,
        ops: Iterable[OpixOp] = (),
        omit_ones: bool = True,
        sep: str = ' ',
        style: str = 'tight',  # 'tight' | 'kv' | 'repeat'
        reverse: bool = False,
        mirror: bool = True,
) -> str:
    """Render a Counter after rewrite; assumes rewrite handled ordering.

    style:
      - 'tight': g or gN (omitting 1s if omit_ones)
      - 'kv': g:N (omitting :1 if omit_ones)
      - 'repeat': g repeated N times per group (integers only)

    reverse/mirror act on the sequence of groups post-formatting.
    """
    view = apply_ops(counts, ops)
    items = list(view.items())
    parts: List[str] = []
    if style == 'repeat':
        for g, n in items:
            if not n:
                continue
            try:
                k = int(n)
            except Exception:
                k = int(float(n))
            if k > 0:
                parts.append(g * k)
    elif style == 'kv':
        for g, n in items:
            if not n:
                continue
            if omit_ones and n == 1:
                parts.append(f'{g}')
            else:
                parts.append(f'{g}:{n:g}')
    else:  # tight
        for g, n in items:
            if not n:
                continue
            if omit_ones and n == 1:
                parts.append(g)
            else:
                parts.append(f'{g}{n:g}')
    if reverse:
        parts = list(reversed(parts))
    if mirror:
        parts = parts + list(reversed(parts))
    return sep.join(parts)


def _infinite_sequence(parts: List[str], mode: str = 'mirror') -> Iterator[str]:
    if not parts:
        while True:
            yield ''
    if mode == 'cycle':
        while True:
            for p in parts:
                yield p
    elif mode == 'mirror':
        if len(parts) <= 1:
            while True:
                for p in parts:
                    yield p
        mirror_seq = parts + parts[-2:0:-1]
        while True:
            for p in mirror_seq:
                yield p
    else:
        raise ValueError('mode must be cycle or mirror')


def counts_iter(
        counts: Mapping[str, float],
        *,
        ops: Iterable[OpixOp] = (),
        omit_ones: bool = True,
        style: str = 'tight',
        mode: str = 'mirror',  # 'cycle' | 'mirror'
) -> Iterator[str]:
    view = apply_ops(counts, ops)
    items = list(view.items())
    parts: List[str] = []
    if style == 'repeat':
        for g, n in items:
            if not n:
                continue
            try:
                k = int(n)
            except Exception:
                k = int(float(n))
            if k > 0:
                parts.append(g * k)
    elif style == 'kv':
        for g, n in items:
            if not n:
                continue
            if omit_ones and n == 1:
                parts.append(f'{g}')
            else:
                parts.append(f'{g}:{n:g}')
    else:
        for g, n in items:
            if not n:
                continue
            if omit_ones and n == 1:
                parts.append(g)
            else:
                parts.append(f'{g}{n:g}')
    return _infinite_sequence(parts, mode=mode)


def counts_window(
        counts: Mapping[str, float],
        *,
        ops: Iterable[OpixOp] = (),
        omit_ones: bool = True,
        style: str = 'tight',
        mode: str = 'mirror',
        sep: str = ' ',
        groups: Optional[int] = None,
        chars: Optional[int] = None,
) -> str:
    it = counts_iter(counts, ops=ops, omit_ones=omit_ones, style=style, mode=mode)
    if groups is not None:
        out: List[str] = []
        for _ in range(max(0, groups)):
            out.append(next(it))
        return sep.join(out)
    if chars is not None:
        acc = ''
        while len(acc) < chars:
            if acc:
                acc += sep
            acc += next(it)
        return acc[:chars]
    # default: one mirrored cycle worth
    return counts_render(counts, ops=ops, omit_ones=omit_ones, style=style, sep=sep, mirror=(mode=='mirror'))


def reverse_bits(value: int, bit_count: int) -> int:
    out = 0
    for i in range(bit_count):
        out = (out << 1) | ((value >> i) & 1)
    return out


def build_huffman_tree(code_lengths: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    lengths = {sym: int(L) for sym, L in code_lengths.items() if int(L) > 0}
    if not lengths:
        return {}
    max_bits = max(lengths.values())
    bl_count = [0] * (max_bits + 1)
    for L in lengths.values():
        bl_count[L] += 1
    code = 0
    next_code = [0] * (max_bits + 1)
    for bits in range(1, max_bits + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    symbols_sorted = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    mapping: Dict[int, Tuple[int, int]] = {}
    for sym, L in symbols_sorted:
        c = next_code[L]
        next_code[L] += 1
        mapping[sym] = (c, L)
    return mapping


def build_heap_decoder(code_lengths: Dict[int, int]) -> List[Dict[str, Optional[int]]]:
    canonical = build_huffman_tree(code_lengths)
    if not canonical:
        return []
    nodes: List[Dict[str, Optional[int]]] = [{'left': None, 'right': None, 'value': None}]

    def add_leaf(rev_code: int, length: int, value: int):
        idx = 0
        for i in range(length):
            bit = (rev_code >> i) & 1
            key = 'right' if bit else 'left'
            nxt = nodes[idx][key]
            if nxt is None:
                nodes.append({'left': None, 'right': None, 'value': None})
                nxt = len(nodes) - 1
                nodes[idx][key] = nxt
            idx = nxt
        nodes[idx]['value'] = value

    for sym, (code, L) in canonical.items():
        rev = reverse_bits(code, L)
        add_leaf(rev, L, sym)
    return nodes


class Aspirate(Counter):
    def __init__(
        self,
        counts: Optional[Mapping[str, float] | Iterable[str]] = None,
        *,
        heap: Optional[List[Dict[str, Optional[int]]]] = None,
        length_hist: Optional[Counter] = None,
        shape_id: int = 0,
        max_bits: int = 0,
    ):
        # Allow Counter-style initialization: mapping or iterable of keys
        super().__init__(counts or {})
        self.heap = heap or []
        self.length_hist = length_hist or Counter()
        self.shape_id = int(shape_id)
        self.max_bits = int(max_bits)

    # --- Callable-friendly API ---
    def apply(self, *ops: Callable[[Mapping[str, float]], Dict[str, float]], inplace: bool = False) -> 'Aspirate':
        new_map = apply_ops(self, ops)
        if inplace:
            self.clear()
            self.update(new_map)
            return self
        return Aspirate(new_map, heap=self.heap, length_hist=self.length_hist, shape_id=self.shape_id, max_bits=self.max_bits)

    def __call__(self, *ops: Callable[[Mapping[str, float]], Dict[str, float]]) -> 'Aspirate':
        return self.apply(*ops, inplace=False)

    def __rshift__(self, op: Callable[[Mapping[str, float]], Dict[str, float]]) -> 'Aspirate':
        return self.apply(op, inplace=False)

    def __irshift__(self, op: Callable[[Mapping[str, float]], Dict[str, float]]) -> 'Aspirate':
        return self.apply(op, inplace=True)

    # --- Combo convenience ---
    def overlay_with(self, *others: 'Aspirate') -> 'Aspirate':
        merged = Aspirate(self)
        for op in others:
            for g, n in op.items():
                if n:
                    merged[g] += n
        return merged

    def hstack(self, *others: 'Aspirate', separator: str = ' ', format_spec: str = '') -> str:
        parts = [format(self, format_spec) if format_spec else str(self)]
        for op in others:
            parts.append(format(op, format_spec) if format_spec else str(op))
        return separator.join(parts)

    def vstack(self, *others: 'Aspirate', separator: str = ' ', format_spec: str = '') -> str:
        parts = [format(self, format_spec) if format_spec else str(self)]
        for op in others:
            parts.append(format(op, format_spec) if format_spec else str(op))
        return "\n".join(parts)

    def blend_with(self, *others: 'Aspirate', separator: str = ' ', format_spec: str = '') -> str:
        parts = [f"[{format(self, format_spec) if format_spec else str(self)}]"]
        for op in others:
            parts.append(f"[{format(op, format_spec) if format_spec else str(op)}]")
        return separator.join(parts)

    @staticmethod
    def from_code_lengths(code_lengths: Dict[int, int]) -> 'Aspirate':
        heap = build_heap_decoder(code_lengths)
        hist = Counter(int(L) for L in code_lengths.values() if int(L) > 0)
        acc = 0
        for L, cnt in sorted(hist.items()):
            acc = mix64(acc + mix64((L << 16) | (cnt & 0xFFFF)))
        max_bits = max(code_lengths.values()) if code_lengths else 0
        return Aspirate(heap=heap, length_hist=hist, shape_id=acc & MASK64, max_bits=int(max_bits))

    def color(self) -> Tuple[float, float, float]:
        base = self if len(self) else self.length_hist
        return counts_to_color(base)  # reuse mapping logic

    @staticmethod
    def from_heap(heap: List[Dict[str, Optional[int]]], code_lengths: Optional[Dict[int, int]] = None) -> 'Aspirate':
        # If code_lengths are provided, compute histogram/id; else minimal metadata
        if code_lengths:
            hist = Counter(int(L) for L in code_lengths.values() if int(L) > 0)
            acc = 0
            for L, cnt in sorted(hist.items()):
                acc = mix64(acc + mix64((L << 16) | (cnt & 0xFFFF)))
            max_bits = max(code_lengths.values()) if code_lengths else 0
        else:
            hist = Counter()
            acc = 0
            max_bits = 0
        a = Aspirate(heap=heap, length_hist=hist, shape_id=acc & MASK64, max_bits=int(max_bits))
        return a

    def __str__(self) -> str:
        # Prefer explicit glyphs if present, else render length histogram
        if len(self):
            return counts_render(self, ops=[sort_by_count(False)], style='tight', omit_ones=True)
        # Render code-length histogram as simple string keys
        as_map = {str(k): float(v) for k, v in self.length_hist.items()}
        return counts_render(as_map, ops=[sort_by_count(False)], style='tight', omit_ones=True)

    def __format__(self, spec: str) -> str:
        # Macro dispatch if a macro name matches the whole spec
        macro_obj = MACROS.get(spec.strip()) if spec else None
        if macro_obj:
            ops = build_ops_from_specs(macro_obj.get('ops', []))
            style = macro_obj.get('style', 'tight')
            reverse = bool(macro_obj.get('reverse', False))
            mirror = bool(macro_obj.get('mirror', False))
            omit_ones = bool(macro_obj.get('omit_ones', True))
            sep = str(macro_obj.get('sep', ' '))
            macro_map: Mapping[str, float] = self if len(self) else {str(k): float(v) for k, v in self.length_hist.items()}
            return counts_render(macro_map, ops=ops, style=style, omit_ones=omit_ones, sep=sep, reverse=reverse, mirror=mirror)

        # Parse simple comma-separated spec: e.g. "kv,order=count,reverse,sep=,"
        style = 'tight'
        reverse = False
        mirror = False
        omit_ones = True
        sep = ' '
        ops_list: List[Callable[[Mapping[str, float]], Dict[str, float]]] = []
        if spec:
            for token in (t.strip() for t in spec.split(',') if t.strip()):
                if token in ('tight', 'kv', 'repeat'):
                    style = token
                elif token == 'reverse':
                    reverse = True
                elif token == 'mirror':
                    mirror = True
                elif token == 'keep1':
                    omit_ones = False
                elif token == 'omit1':
                    omit_ones = True
                elif token.startswith('sep='):
                    sep = token[4:]
                elif token.startswith('order='):
                    key = token[6:]
                    if key == 'count':
                        ops_list.append(sort_by_count(False))
                    elif key == '-count':
                        ops_list.append(sort_by_count(True))
                    elif key == 'glyph':
                        ops_list.append(sort(lambda kv: kv[0], reverse=False))
                    elif key == '-glyph':
                        ops_list.append(sort(lambda kv: kv[0], reverse=True))
                elif token.startswith('topk='):
                    try:
                        k = int(token[6:])
                        ops_list.append(topk(k))
                    except Exception:
                        pass
        inline_map: Mapping[str, float] = self if len(self) else {str(k): float(v) for k, v in self.length_hist.items()}
        return counts_render(inline_map, ops=ops_list, style=style, omit_ones=omit_ones, sep=sep, reverse=reverse, mirror=mirror)


# Alias for ergonomics
Opix = Aspirate


## OpixCombo removed: use Aspirate.overlay_with() to merge, and hstack()/vstack()/blend_with() to render strings


# (Setattr helpers removed; methods defined on Aspirate)


# --- Msgpack-based Opix registry (ops + macros) ---

MACROS: Dict[str, Dict[str, Any]] = {}


def _build_op(name: str, args: Mapping[str, Any]) -> OpixOp:
    if name == 'keep':
        return keep(args.get('glyphs', []))
    if name == 'drop':
        return drop(args.get('glyphs', []))
    if name == 'rename':
        return rename(args.get('mapping', {}))
    if name == 'coalesce':
        return coalesce(args.get('groups', {}))
    if name == 'thresh':
        return thresh(float(args.get('threshold', 1)))
    if name == 'topk':
        return topk(int(args.get('k', 1)))
    if name == 'normalize':
        return normalize()
    if name == 'order':
        return order(args.get('order', []), append_rest=bool(args.get('append_rest', True)))
    if name == 'sort_by_count':
        return sort_by_count(bool(args.get('ascending', False)))
    if name == 'sort_by_glyph':
        return sort_by_glyph(bool(args.get('ascending', True)))
    if name == 'sort':
        key = args.get('key', 'glyph')
        asc = bool(args.get('ascending', True))
        if key == 'count':
            return sort_by_count(ascending=asc)
        return sort_by_glyph(ascending=asc)
    if name == 'blit':
        mask_spec = args.get('mask')
        mask_fn = None
        if isinstance(mask_spec, dict) and 'min' in mask_spec:
            m = float(mask_spec['min'])
            mask_fn = (lambda g, n, m=m: n >= m)
        clip_arg = args.get('clip')
        clip_val: Optional[Tuple[Optional[float], Optional[float]]] = None
        if isinstance(clip_arg, (list, tuple)) and len(clip_arg) == 2:
            lo = None if clip_arg[0] is None else float(clip_arg[0])
            hi = None if clip_arg[1] is None else float(clip_arg[1])
            clip_val = (lo, hi)
        dst_map = args.get('dst_map')
        dst_fn = (lambda g, m=dst_map: m.get(g, g)) if isinstance(dst_map, dict) else None
        return blit(
            mask=mask_fn,
            add=float(args.get('add', 0.0)),
            mul=float(args.get('mul', 1.0)),
            set_to=args.get('set_to', None),
            clip=clip_val,
            dst=dst_fn,
            keep_rest=bool(args.get('keep_rest', True)),
        )
    return lambda d: dict(d)


def build_ops_from_specs(specs: Iterable[Any]) -> List[OpixOp]:
    ops: List[OpixOp] = []
    for spec in specs or []:
        if isinstance(spec, str):
            ops.append(_build_op(spec, {}))
        elif isinstance(spec, dict):
            name = spec.get('op', '')
            args = spec.get('args', {})
            ops.append(_build_op(name, args))
    return ops


def load_opix_registry(data: bytes) -> None:
    try:
        obj = msgpack.unpackb(data, raw=False)
    except Exception:
        return
    if not isinstance(obj, dict):
        return
    macros = obj.get('macros', {})
    if isinstance(macros, dict):
        MACROS.update(macros)


def dump_opix_registry() -> bytes:
    obj = {
        'version': 1,
        'macros': MACROS,
    }
    try:
        data2: bytes = msgpack.packb(obj, use_bin_type=True) or b''
        return data2
    except Exception:
        return b''

