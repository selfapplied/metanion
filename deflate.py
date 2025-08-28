import numpy as np
from bitstring import ConstBitStream, ReadError

def build_huffman_tree(code_lengths):
    """Builds a canonical Huffman code mapping from code lengths."""
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
    mapping = {}
    for sym, L in symbols_sorted:
        c = next_code[L]
        next_code[L] += 1
        mapping[sym] = (c, L)
    return mapping

def decode_run_length_codes(stream, cl_map, num_codes):
    """Decodes run-length encoded code lengths for HLIT+HDIST."""
    if not cl_map:
        return []
    
    max_bits = max(L for _, L in cl_map.values())
    by_len = {L: {} for L in range(1, max_bits + 1)}
    for sym, (code, L) in cl_map.items():
        by_len[L][code] = sym

    def decode_symbol():
        code = 0
        for bits in range(1, max_bits + 1):
            code = (code << 1) | stream.read('uint:1')
            bucket = by_len.get(bits)
            if bucket is not None and code in bucket:
                return bucket[code]
        raise ReadError('Invalid Huffman code')

    out = []
    i = 0
    prev = 0
    while i < num_codes:
        sym = decode_symbol()
        if 0 <= sym <= 15:
            out.append(sym)
            prev = sym
            i += 1
        elif sym == 16:
            repeat = stream.read('uint:2') + 3
            out.extend([prev] * repeat)
            i += repeat
        elif sym == 17:
            repeat = stream.read('uint:3') + 3
            out.extend([0] * repeat)
            prev = 0
            i += repeat
        elif sym == 18:
            repeat = stream.read('uint:7') + 11
            out.extend([0] * repeat)
            prev = 0
            i += repeat
        else:
            raise ReadError('Invalid RLE symbol')
    return out[:num_codes]

def parse_dynamic_huffman_block(stream):
    """Parses a dynamic Huffman block and returns the decoding trees and meta."""
    hlit = stream.read('uint:5') + 257
    hdist = stream.read('uint:5') + 1
    hclen = stream.read('uint:4') + 4

    cl_order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    cl_code_lengths = {i: 0 for i in range(19)}
    for i in range(hclen):
        cl_code_lengths[cl_order[i]] = stream.read('uint:3')
    
    cl_tree = build_huffman_tree(cl_code_lengths)
    main_code_lengths = decode_run_length_codes(stream, cl_tree, hlit + hdist)
    
    if not main_code_lengths or len(main_code_lengths) < (hlit + hdist):
        return {}, {}, {'hlit': hlit, 'hdist': hdist, 'hclen': hclen, 'cl_code_lengths': cl_code_lengths}
    
    lit_len_code_lengths = {i: main_code_lengths[i] for i in range(hlit)}
    dist_code_lengths = {i: main_code_lengths[i+hlit] for i in range(hdist)}

    lit_len_tree = build_huffman_tree(lit_len_code_lengths)
    dist_tree = build_huffman_tree(dist_code_lengths)

    return lit_len_tree, dist_tree, {
        'hlit': hlit, 'hdist': hdist, 'hclen': hclen,
        'cl_code_lengths': cl_code_lengths,
        'lit_len_code_lengths': lit_len_code_lengths,
        'dist_code_lengths': dist_code_lengths,
    }


