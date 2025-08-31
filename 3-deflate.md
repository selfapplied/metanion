The DEFLATE algorithm is a lossless data compression method that combines two other compression strategies: LZ77 and Huffman coding. It is widely used in file formats such as ZIP, GZIP, and PNG. 

The compression process works in two main stages: 

Duplicate elimination (LZ77): The algorithm replaces repeated sequences of data with a reference back to a previous occurrence.
Entropy encoding (Huffman coding): It then compresses the remaining data stream, which now consists of literal characters and match references, using Huffman coding. 

Stage 1: LZ77 compression
    The first stage of the DEFLATE algorithm replaces repeated byte sequences with
    pointers to previous occurrences. 

    Sliding window: The algorithm uses a "sliding window" of up to 32 KB, which
    stores the most recent uncompressed data.

    Match search: As the compressor moves through the input data, it searches for
    the longest sequence of bytes that matches a sequence already present in the
    sliding window.

    Outputting match pairs: When a match is found, the algorithm outputs a [length,
    distance] pair instead of the raw data.

    Length: The number of bytes in the matching sequence (3–258 bytes).

    Distance: The number of bytes to look backward in the sliding window to find
    the start of the match (1–32,768 bytes).

    Outputting literals: If no match is found, the algorithm simply outputs the
    single unmatched byte as a "literal". 


Example of LZ77 matching:
Consider the string Blah blah blah!. 
The first "Blah " is read and stored in the sliding window.
The algorithm then sees the next "blah". This is a match.
Instead of writing "blah", the compressor outputs a [length, distance] pair:
length = 4 (for "blah")
distance = 5 (to point back to "Blah ") 

Stage 2: Huffman coding
    The raw output from the LZ77 stage is a stream of literals and [length,
    distance] pairs. Huffman coding further compresses this stream. 
    Two Huffman trees: DEFLATE uses two Huffman trees to assign variable-length
    codes based on frequency:

    - One tree for the literal characters and the length codes.
    - A second tree for the distance codes.
    - Shorter codes for frequent data: Huffman coding works by creating a prefix code
    tree where shorter codes are assigned to more frequently occurring symbols
    (literals, lengths, or distances).

    Encoding the trees: The Huffman trees themselves are included at the start of
    each data block to tell the decompressor how to interpret the codes. 
    Blocks and compression levels

    The DEFLATE stream is split into blocks, and the compression can adapt to the
    data. The compressor can use one of three block types: 

    No compression (Stored): The data is simply stored without any compression.
    This is used for data that is already compressed or incompressible.

    Fixed Huffman codes: A pre-defined set of Huffman codes is used. This is faster
    because the tree doesn't need to be calculated or stored with the data but is
    not as efficient.

    Dynamic Huffman codes: The algorithm creates optimized Huffman trees for each
    block, which are then included with the compressed data. This provides the best
    compression ratio. 

The user-selected compression level (from 0 to 9 in zlib) controls the tradeoff
between compression speed and ratio. Higher levels spend more time searching
for longer and better matches, resulting in a smaller output file but taking
longer to compress. 
