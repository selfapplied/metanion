import zlib
from bitstring import ConstBitStream, ReadError
from typing import Dict, Tuple, Optional, Union, SupportsBytes, Callable, Iterable
from collections import Counter
from typing import List
from aspire import (
    mix64,
    stable_str_hash,
    build_huffman_tree,
    build_heap_decoder,
    pack_qnan_with_payload,
    Opix,
)

KW_CARRY = '✓'
KW_BORROW = 'α'
KW_DRIFT = 'ℋ'
KW_CYCLE = '∞'


class BitLSB:
    """LSB-first bit reader over a ConstBitStream.

    DEFLATE specifies that within each byte, the least significant bit is read first.
    This helper buffers whole bytes from the underlying stream and then yields bits
    in LSB-first order. Use align_to_byte() when the format requires byte alignment
    (e.g., stored blocks).
    """

    def __init__(self, stream: ConstBitStream):
        self.stream = stream
        self._bit_buf = 0
        self._bit_count = 0

    def _fill_byte(self):
        if self._bit_count == 0:
            # Read next byte from the stream; if at EOF, raise
            byte_val = self.stream.read('uint:8')
            self._bit_buf = byte_val
            self._bit_count = 8

    def read_bit(self) -> int:
        if self._bit_count == 0:
            self._fill_byte()
        bit = self._bit_buf & 1
        self._bit_buf >>= 1
        self._bit_count -= 1
        return bit

    def read_bits(self, n: int) -> int:
        """Read n bits LSB-first and return as integer."""
        val = 0
        shift = 0
        while n > 0:
            if self._bit_count == 0:
                self._fill_byte()
            take = min(n, self._bit_count)
            # Take 'take' bits from LSB side of buffer
            mask = (1 << take) - 1
            val |= (self._bit_buf & mask) << shift
            self._bit_buf >>= take
            self._bit_count -= take
            n -= take
            shift += take
        return val

    def align_to_byte(self):
        """Advance to next byte boundary in the conceptual bitstream."""
        # Drop any buffered bits and advance underlying stream to next byte boundary
        self._bit_buf = 0
        self._bit_count = 0
        misalign = (-self.stream.pos) % 8
        if misalign:
            self.stream.pos += misalign

    def read_bytes(self, n: int) -> bytes:
        """Read n bytes. Requires byte alignment; discards any buffered bits."""
        self.align_to_byte()
        return self.stream.read(f'bytes:{n}')


# --- DEFLATE length/distance tables (RFC 1951, Section 3.2.5) ---
# Codes 257..285 map to lengths; 285 represents 258 with 0 extra bits
LENGTH_BASE = [
    3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227, 258
]
LENGTH_EXTRA = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5, 0
]

# Distance codes 0..29
DIST_BASE = [
    1, 2, 3, 4, 5, 7, 9, 13,
    17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577
]
DIST_EXTRA = [
    0, 0, 0, 0, 1, 1, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13
]


# use shared build_huffman_tree and build_heap_decoder from aspire


def parse_stored_block(reader: BitLSB, on_warning: Callable[[str], None] = lambda _k: None):
    """Parses a stored (uncompressed) block and returns the raw bytes."""
    # Align to next byte boundary per spec
    reader.align_to_byte()
    # Ensure we have at least LEN+NLEN available
    bytes_avail = (len(reader.stream) - reader.stream.pos) // 8
    if bytes_avail < 4:
        on_warning('⟂')  # stored_truncated_header
        return b""
    # LEN and NLEN are 16-bit little-endian
    len_bytes = reader.read_bytes(2)
    nlen_bytes = reader.read_bytes(2)
    length = int.from_bytes(len_bytes, 'little')
    nlen = int.from_bytes(nlen_bytes, 'little')
    # Validate one's complement (soft-fail)
    if (length ^ 0xFFFF) != nlen:
        on_warning('≠')  # LEN/NLEN mismatch
        return b""
    # Read stored data
    bytes_avail = (len(reader.stream) - reader.stream.pos) // 8
    if bytes_avail < length:
        on_warning('⋯')  # stored_truncated_data
        return b""
    data = reader.read_bytes(length)
    return data


def parse_stored_fallback(reader: BitLSB) -> bytes:
    """Byte-align and drain remaining bytes as a literal block (non-spec fallback)."""
    reader.align_to_byte()
    bytes_avail = (len(reader.stream) - reader.stream.pos) // 8
    if bytes_avail <= 0:
        return b""
    return reader.read_bytes(int(bytes_avail))


def get_static_huffman_tables():
    """Returns the fixed Huffman tables for static blocks (block_type == 1)."""
    # Fixed literal/length codes (RFC 1951 section 3.2.6)
    lit_len_lengths = {}
    for i in range(0, 144):
        lit_len_lengths[i] = 8
    for i in range(144, 256):
        lit_len_lengths[i] = 9
    for i in range(256, 280):
        lit_len_lengths[i] = 7
    for i in range(280, 288):
        lit_len_lengths[i] = 8

    # Fixed distance codes (RFC 1951 section 3.2.6)
    dist_lengths = {}
    for i in range(0, 32):
        dist_lengths[i] = 5

    lit_len_tree = build_heap_decoder(lit_len_lengths)
    dist_tree = build_heap_decoder(dist_lengths)

    return lit_len_tree, dist_tree


def parse_static_huffman_block(reader: BitLSB):
    """Parses a static Huffman block using the fixed tables."""
    lit_len_tree, dist_tree = get_static_huffman_tables()
    return lit_len_tree, dist_tree, {'block_type': 'static_huffman'}


def parse_dynamic_huffman_block(reader: BitLSB, on_error: Callable[[str], None] = lambda _k: None):
    """Parses a dynamic Huffman block and returns the decoding trees and meta."""
    hlit = reader.read_bits(5) + 257
    hdist = reader.read_bits(5) + 1
    hclen = reader.read_bits(4) + 4

    cl_order = [16, 17, 18, 0, 8, 7, 9, 6,
                10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    cl_code_lengths = {i: 0 for i in range(19)}
    for i in range(hclen):
        cl_code_lengths[cl_order[i]] = reader.read_bits(3)
    # Build canonical codes for code-length alphabet, then decode using LSB-first
    cl_tree = build_huffman_tree(cl_code_lengths)
    main_code_lengths = decode_run_length_codes(
        reader, cl_code_lengths=cl_code_lengths, num_codes=hlit + hdist, on_error=on_error)

    if not main_code_lengths or len(main_code_lengths) < (hlit + hdist):
        return {}, {}, {'hlit': hlit, 'hdist': hdist, 'hclen': hclen, 'cl_code_lengths': cl_code_lengths}

    lit_len_code_lengths = {i: main_code_lengths[i] for i in range(hlit)}
    dist_code_lengths = {i: main_code_lengths[i+hlit] for i in range(hdist)}

    lit_len_tree = build_heap_decoder(lit_len_code_lengths)
    dist_tree = build_heap_decoder(dist_code_lengths)

    has_new_tokens = bool(lit_len_code_lengths) and any(
        int(v) > 0 for v in lit_len_code_lengths.values())
    
    # Create block metadata for genetic enhancement
    block_meta = {
        'type': 'dynamic',
        'hlit': hlit, 'hdist': hdist, 'hclen': hclen,
        'cl_code_lengths': cl_code_lengths,
        'lit_len_code_lengths': lit_len_code_lengths,
        'dist_code_lengths': dist_code_lengths,
        'has_new_tokens': has_new_tokens,
    }
    
    return lit_len_tree, dist_tree, block_meta


def decode_run_length_codes(reader: BitLSB, cl_code_lengths: Dict[int, int], num_codes: int, on_error: Callable[[str], None] = lambda _k: None):
    """Decodes run-length encoded code lengths for HLIT+HDIST using heap traversal."""
    if not cl_code_lengths:
        return []

    cl_heap = build_heap_decoder(cl_code_lengths)

    def decode_symbol_from_heap(nodes: List[Dict[str, Optional[int]]]) -> Optional[int]:
        if not nodes:
            on_error('ℋ')
            return None
        idx = 0
        while True:
            node = nodes[idx]
            if node['value'] is not None:
                return int(node['value'])
            bit = reader.read_bit()
            next_idx = node['right'] if bit else node['left']
            if next_idx is None:
                on_error('ℋ')
                return None
            idx = next_idx

    out = []
    i = 0
    prev = 0
    while i < num_codes:
        sym = decode_symbol_from_heap(cl_heap)
        if sym is None:
            # Abort RLE decoding on error
            return out[:i]
        if 0 <= sym <= 15:
            out.append(sym)
            prev = sym
            i += 1
        elif sym == 16:
            repeat = reader.read_bits(2) + 3
            out.extend([prev] * repeat)
            i += repeat
        elif sym == 17:
            repeat = reader.read_bits(3) + 3
            out.extend([0] * repeat)
            prev = 0
            i += repeat
        elif sym == 18:
            repeat = reader.read_bits(7) + 11
            out.extend([0] * repeat)
            prev = 0
            i += repeat
        else:
            on_error('ℋ')
            break
    return out[:num_codes]

# --- ZIP-aware deflate extraction ---


def create_deflate_stream(compressed_data: Union[bytes, Dict], fallback_data: bytes) -> bytes:
    """
    Creates a deflate stream from either raw bytes or ZIP metadata.
    Falls back to decompressed content if needed.
    """
    if compressed_data and isinstance(compressed_data, dict):
        # We have ZIP metadata with raw deflate data
        return compressed_data['raw_deflate']
    else:
        # Fallback to decompressed content
        return fallback_data


def validate_deflate_stream(stream_bytes: bytes, expected_size: Optional[int] = None) -> bytes:
    """
    Validates and prepares deflate stream bytes.
    Ensures we have proper bytes and handles type conversion.
    """
    # Ensure we have bytes
    if isinstance(stream_bytes, str):
        stream_bytes = stream_bytes.encode('utf-8')
    if isinstance(stream_bytes, SupportsBytes):
        stream_bytes = bytes(stream_bytes)

    # Validate size if expected
    if expected_size and len(stream_bytes) != expected_size:
        raise ValueError(
            f"Stream size mismatch: expected {expected_size}, got {len(stream_bytes)}")

    return stream_bytes


# Legacy placeholder removed; decoding handled in BlockProcessor.decode_payload


class ErrorTracker:
    def __init__(self):
        self.error_counts: Counter = Opix(
            heap=[], length_hist=Counter(), shape_id=0, max_bits=0)
        self._error_window: List[str] = []
        self._seen_states: Dict[tuple, int] = {}
        self._cycle_count: int = 0
        self._step_idx: int = 0

    def count_error(self, key: str):
        self.error_counts[key] += 1

    def record_error(self, key: str, attempted_parse: str, stream_pos: int):
        self.count_error(key)
        self._error_window.append(key)
        if len(self._error_window) > 4:
            self._error_window.pop(0)

        phase = int(stream_pos % 8)
        byte_pos = int(stream_pos // 8)
        recent_hash = 0
        for k in self._error_window:
            recent_hash = mix64(recent_hash + stable_str_hash(k))

        state = (phase, attempted_parse, byte_pos, recent_hash)
        prev = self._seen_states.get(state)

        if prev is not None and (self._step_idx - prev) <= 128:
            self._cycle_count += 1
            if self._cycle_count >= 2:
                self.count_error('∞')
                return True  # Cycle detected

        self._seen_states[state] = self._step_idx
        self._step_idx += 1
        return False


def stream_events(asset_name: str, genome: 'Genome', ingest_text_func) -> Iterable[Tuple[str, Union[str, Tuple[int, int], Dict]]]:
    """
    Main entry point for processing a DEFLATE stream and yielding events.
    """
    compressed_bytes = genome.compressed_assets.get(asset_name, None)
    decompressed_bytes = genome.extra_assets.get(asset_name, b"")
    stream_bytes = create_deflate_stream(compressed_bytes, decompressed_bytes)
    stream_bytes = validate_deflate_stream(stream_bytes)

    stream_processor = StreamProcessor(stream_bytes)
    error_tracker = ErrorTracker()

    # Ingest text to build grammar
    ingest_text_func(asset_name, decompressed_bytes)

    yield from _process_events(asset_name, stream_processor, error_tracker)


def _process_events(asset_name: str, sp: 'StreamProcessor', et: 'ErrorTracker') -> Iterable[Tuple[str, Union[str, Tuple[int, int], Dict]]]:
    """
    Core event processing loop.
    """
    block_count = 0
    state_phi = 0.0

    def _format_atom_label(phase: int) -> str:
        glyphs_str = str(et.error_counts)
        return f"{asset_name}:{block_count}{phase}ºπ[{glyphs_str}]"

    while True:
        # Check for end of stream
        if sp.stream.pos >= sp.max_pos:
            break

        # Check for incomplete header
        if sp.max_pos - sp.stream.pos < 3:
            break

        last_block = bool(sp.reader.read_bits(1))
        block_type = sp.reader.read_bits(2)

        info = {}
        if block_type == 2:  # Dynamic Huffman
            try:
                lit_len_tree, dist_tree, meta = parse_dynamic_huffman_block(
                    sp.reader, on_error=lambda k: et.record_error(k, 'dynamic_header', sp.stream.pos))
                info = {'lit_len_tree': lit_len_tree,
                        'dist_tree': dist_tree, 'meta': meta}
            except Exception:
                if et.record_error('ℋ', 'dynamic_header', sp.stream.pos):
                    break
                # Fallback: parse remaining as a stored literal block
                data = parse_stored_fallback(sp.reader)
                if data:
                    et.count_error('✓')
                # Atomization: set state_phi to qNaN with payload and label
                atom_id = mix64(mix64(et._step_idx) ^
                                mix64(len(data))) & ((1 << 51) - 1)
                state_phi = pack_qnan_with_payload(atom_id)
                et.count_error('α')
                phase = int(sp.stream.pos % 8)
                atom_label = _format_atom_label(phase)
                meta = {'block_type': 'stored_fallback', 'stored_data': data,
                        'atom_id': int(atom_id), 'atom_label': atom_label}
                info = {'meta': meta}

        elif block_type == 1:  # Static Huffman
            lit_len_tree, dist_tree, meta = parse_static_huffman_block(
                sp.reader)
            info = {'lit_len_tree': lit_len_tree,
                    'dist_tree': dist_tree, 'meta': meta}

        elif block_type == 0:  # Stored
            stored_data = parse_stored_block(
                sp.reader, on_warning=lambda k: et.record_error(k, 'stored_header', sp.stream.pos))
            if stored_data:
                et.count_error('✓')
            info = {'meta': {'stored_data': stored_data}}

        else:
            # Reserved BTYPE=3; hard bail end-of-stream
            if et.record_error('⊘', 'header', sp.stream.pos):
                break
            # Fallback: parse remaining as a stored literal block
            data = parse_stored_fallback(sp.reader)
            if data:
                et.count_error('✓')
            # Atomization: set state_phi to qNaN with payload and label
            atom_id = mix64(mix64(et._step_idx) ^
                            mix64(len(data))) & ((1 << 51) - 1)
            state_phi = pack_qnan_with_payload(atom_id)
            et.count_error('α')
            phase = int(sp.stream.pos % 8)
            atom_label = _format_atom_label(phase)
            meta = {'block_type': 'stored_fallback', 'stored_data': data,
                    'atom_id': int(atom_id), 'atom_label': atom_label}
            info = {'meta': meta}

        # Emit events based on parsed block
        if info:
            block_meta = info.get('meta', {})
            if isinstance(block_meta, dict) and 'atom_id' in block_meta:
                yield ("atom", {"id": block_meta.get('atom_id'), "label": block_meta.get('atom_label')})
            if info.get('last_block'):
                break
            # Emit block event with type/meta for orientation logic upstream
            kind = 'dynamic' if block_type == 2 else (
                'static' if block_type == 1 else 'stored')
            payload = {"type": kind}
            if isinstance(block_meta, dict):
                ll = block_meta.get('lit_len_code_lengths', {})
                if ll:
                    payload['lit_len_code_lengths'] = ll
            yield ("block", payload)
            # Huffman-coded blocks: optionally emit phase hint
            phase = int(sp.stream.pos % 8)
            yield ("phase", {"phase": phase})
            # Decode payload to literals/matches, but keep minimal output
            lit_heap = block_meta.get('lit_len_tree')
            dist_heap = block_meta.get('dist_tree')
            if lit_heap and dist_heap:
                try:
                    # Minimal literal-only walk to avoid buffering
                    while True:
                        sym = sp._decode_symbol(lit_heap)
                        if sym < 256:
                            yield ("literal", chr(sym))
                            continue
                        if sym == 256:
                            break
                        # Length/distance
                        code = sym - 257
                        if code < 0 or code >= len(LENGTH_BASE):
                            break
                        base_len = LENGTH_BASE[code]
                        extra_len_bits = LENGTH_EXTRA[code]
                        match_len = base_len + \
                            (sp.reader.read_bits(extra_len_bits)
                             if extra_len_bits else 0)
                        dist_sym = sp._decode_symbol(dist_heap)
                        if dist_sym < 0 or dist_sym >= len(DIST_BASE):
                            break
                        base_dist = DIST_BASE[dist_sym]
                        extra_dist_bits = DIST_EXTRA[dist_sym]
                        match_dist = base_dist + \
                            (sp.reader.read_bits(extra_dist_bits)
                             if extra_dist_bits else 0)
                        yield ("match", (int(match_len), int(match_dist)))
                except Exception:
                    pass
        yield ("summary", {"glyphs": et.error_counts.copy()})
        if info.get('last_block'):
            break


class StreamProcessor:
    def __init__(self, stream_bytes: bytes):
        self.stream = ConstBitStream(bytes=stream_bytes)
        self.reader = BitLSB(self.stream)
        self.max_pos = len(self.stream.bytes) * 8

    def _decode_symbol(self, tree: List[Dict[str, Optional[int]]]) -> int:
        """Decodes a single symbol by traversing the heap-based Huffman tree."""
        if not tree:
            raise ReadError("Empty Huffman tree")
        idx = 0
        while True:
            node = tree[idx]
            if node['value'] is not None:
                return int(node['value'])
            bit = self.reader.read_bit()
            next_idx = node['right'] if bit else node['left']
            if next_idx is None:
                raise ReadError("Invalid Huffman traversal: missing branch")
            idx = next_idx


