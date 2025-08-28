#!/usr/bin/env python3
import argparse
import os
import sys
from collections import Counter, defaultdict
from ce1_genome import Genome, EngineConfig, Grammar
import py_compile
import zipfile
from typing import Optional
import tempfile
import re
import numpy as np
import math
import zlib
import isal.isal_zlib
import hashlib
from tqdm import tqdm
from bitstring import ConstBitStream, ReadError

# --- Helper Functions ---

def _build_huffman_tree(code_lengths):
    """Builds a canonical Huffman decoding tree from a list of code lengths."""
    tree = {}
    max_len = max(code_lengths.values()) if code_lengths else 0
    
    bl_count = {i: 0 for i in range(max_len + 1)}
    for length in code_lengths.values():
        if length > 0:
            bl_count[length] += 1
    
    next_code = {0: 0}
    for bits in range(1, max_len + 1):
        next_code[bits+1] = (next_code[bits] + bl_count[bits]) << 1

    for symbol in sorted(code_lengths.keys()):
        length = code_lengths[symbol]
        if length != 0:
            code = next_code[length]
            bits = f'{code:0{length}b}'
            tree[bits] = symbol
            next_code[length] += 1
            
    return tree

def _read_huffman_payload(stream, lit_len_tree, dist_tree, output_buffer):
    """Decodes the actual payload using the provided Huffman trees."""
    tokens = []
    while True:
        # This is a placeholder for a more efficient bit-by-bit tree traversal
        # For now, we read a few bits and try to match
        code = ""
        symbol = None
        while symbol is None:
            if stream.pos == len(stream): break
            code += stream.read('bin:1')
            if code in lit_len_tree:
                symbol = lit_len_tree[code]

        if symbol is None: break

        if symbol < 256: # Literal
            token = bytes([symbol])
            tokens.append(token)
            output_buffer.append(symbol)
        elif symbol == 256: # End of Block
            break
        else: # Length/Distance pair
            # This part is highly complex, involving reading extra bits for length and distance
            # and then copying from the output_buffer. This is a major undertaking.
            # For now, we acknowledge the back-ref without fully decoding it.
            pass
    return tokens

def create_genesis(sources: list, output_path: str, recursive: bool, include_all: bool):
    """Creates a new genesis genome from source files and directories."""
    print(f"Creating genesis genome and saving to '{output_path}'...")
    
    # 1. Define the Engine's base State Machine
    states = ['carry', 'borrow', 'drift']
    priors = defaultdict(dict)
    for s_from in states:
        for s_to in states:
            priors[s_from][s_to] = 1.0 / len(states)
            
    config = EngineConfig(
        symbols=states,
        priors={k: dict(v) for k, v in priors.items()}
    )

    # 2. Walk the sources to gather assets. No grammar is pre-compiled.
    extra_assets = {}
    
    for source in sources:
        if os.path.isfile(source):
            if not include_all and os.path.basename(source).startswith('.'):
                continue
            if os.path.basename(source).endswith('.genyx.zip'):
                continue
            
            asset_key = os.path.relpath(source, '.')
            try:
                with open(source, 'rb') as f:
                    content = f.read()
                    extra_assets[asset_key] = content
            except Exception as e:
                print(f"Warning: Could not read file '{source}': {e}")

        elif os.path.isdir(source):
            if recursive:
                for root, dirs, files in os.walk(source):
                    if not include_all:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    # Exclude .genyx.zip files from being included in the assets
                    files = [f for f in files if not (f.startswith('.') and not include_all) and not f.endswith('.genyx.zip')]
                    
                    for name in files:
                        file_path = os.path.join(root, name)
                        asset_key = os.path.relpath(file_path, '.')
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                extra_assets[asset_key] = content
                        except Exception as e:
                            print(f"Warning: Could not read file '{file_path}': {e}")
            else: # Not recursive
                for name in os.listdir(source):
                    if not include_all and name.startswith('.'):
                        continue
                    if name.endswith('.genyx.zip'):
                        continue
                    file_path = os.path.join(source, name)
                    if os.path.isfile(file_path):
                        asset_key = os.path.relpath(file_path, '.')
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                extra_assets[asset_key] = content
                        except Exception as e:
                            print(f"Warning: Could not read file '{file_path}': {e}")

    # 3. Create an empty genesis grammar.
    # All grammar discovery will now happen inside the VM at runtime.
    print("Creating empty genesis grammar...")
    grammar = Grammar()
    
    # 4. Assemble and Save the Genome
    genesis_genome = Genome(config=config, grammar=grammar, extra_assets=extra_assets)
    
    genesis_genome.save(output_path)
    print("Done.")

def add_opcode(genome_path: str, script_path: str):
    """Compiles a Python script and adds it to the genome's /ops directory."""
    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at '{genome_path}'")
        return
    if not os.path.exists(script_path):
        print(f"Error: Script file not found at '{script_path}'")
        return

    print(f"Compiling '{script_path}' and adding to '{genome_path}'...")
    
    # Create a temporary file for the bytecode
    with tempfile.NamedTemporaryFile(suffix='.pyc', delete=False) as tmp:
        bytecode_path = tmp.name
    
    try:
        # Compile the script to bytecode
        py_compile.compile(script_path, cfile=bytecode_path, dfile=script_path, doraise=True)
        
        # Read the compiled bytecode
        with open(bytecode_path, 'rb') as f:
            bytecode = f.read()
            
        # Add the bytecode to the genome's /ops directory
        op_name = os.path.basename(script_path)
        op_asset_path = f"ops/{os.path.splitext(op_name)[0]}.pyc"
        
        # Use zipfile to add the new file to the existing archive
        with zipfile.ZipFile(genome_path, 'a') as zf:
            zf.writestr(op_asset_path, bytecode)
            
        print(f"Successfully added opcode as '{op_asset_path}'")

    except py_compile.PyCompileError as e:
        print(f"Error compiling script: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary bytecode file
        if os.path.exists(bytecode_path):
            os.remove(bytecode_path)

def inspect_genome(path: str):
    """Prints a summary of a .genome file's contents."""
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'")
        return

    print(f"--- Inspecting Genome: {os.path.basename(path)} ---")
    
    try:
        genome = Genome.from_path(path)
        
        # Print Config Summary
        config = genome.config
        print("\n[Configuration]")
        print(f"  Symbols ({len(config.symbols)}): {config.symbols}")
        print(f"  Levels: {config.levels}")
        
        # Print Grammar Summary
        grammar = genome.grammar
        print("\n[Grammar]")
        if grammar.unigram_counts:
            top_10 = grammar.unigram_counts.most_common(10)
            print(f"  Top 10 Unigrams: {', '.join([f'{t[0]} ({t[1]})' for t in top_10])}")
        else:
            print("  Unigram counts are empty.")
            
        # Print Extra Assets
        print("\n[Extra Assets]")
        if genome.extra_assets:
            for name, data in genome.extra_assets.items():
                print(f"  - {name} ({len(data)} bytes)")
        else:
            print("  No extra assets found.")

    except Exception as e:
        print(f"Error loading or inspecting genome: {e}")

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="zipc: eonyx genome compiler",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- Create Command ---
    parser_create = subparsers.add_parser('create', help='Create a new genesis genome from files/directories.')
    parser_create.add_argument('output', help='Path for the new .genyx.zip file.')
    parser_create.add_argument('sources', nargs='*', default=['.'], help='Path(s) to source files/directories (default: current directory).')
    parser_create.add_argument('-r', '--recursive', action='store_true', help='Recursively include files in subdirectories.')
    parser_create.add_argument('-a', '--all', action='store_true', help='Include hidden dotfiles.')

    # --- Add-Op Command ---
    parser_addop = subparsers.add_parser('add-op', help='Compile and add an operator script to a genome.')
    parser_addop.add_argument('genome', help='Path to the .genyx.zip genome file.')
    parser_addop.add_argument('script', help='Path to the Python operator script.')

    # --- Inspect Command ---
    parser_inspect = subparsers.add_parser('inspect', help='Inspect a .genome file.')
    parser_inspect.add_argument('path', help='Path to the .genome file to inspect.')

    args = parser.parse_args()

    if args.command is None:
        # Default action: create from current directory with default options
        print("Defaulting to: create . -o .genyx.zip")
        create_genesis(['.'], '.genyx.zip', recursive=False, include_all=False)
    elif args.command == 'create':
        # Default to .genyx.zip if the user doesn't provide an extension
        output_path = args.output
        if not output_path.endswith('.zip'):
            output_path += '.genyx.zip'
        create_genesis(args.sources, output_path, args.recursive, args.all)
    elif args.command == 'add-op':
        add_opcode(args.genome, args.script)
    elif args.command == 'inspect':
        inspect_genome(args.path)

if __name__ == '__main__':
    # Ensure the project root is on the Python path to find the eonyx module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
