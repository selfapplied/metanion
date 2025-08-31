#!/usr/bin/env python3
import argparse
import os
import sys
from collections import Counter, defaultdict
from genome import Genome, EngineConfig, Grammar
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
from zip import manifest_from_directory, blit_to_reflex, FileOp, ScanResult, unpack_reflex, ZipFlexManifest
from aspire import Opix
from pathlib import Path
# Note: deflate helpers not needed here; removed stale import

# --- Helper Functions ---


def create_genesis(sources: list, output_path: str, recursive: bool, include_all: bool, bootstrap: bool = True, energy_budget: int = 10 * 1024 * 1024):
    """Creates a new genesis genome from source files and directories."""
    print(
        f"Creating genesis genome with energy budget {energy_budget} bytes, saving to '{output_path}'...")
    
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

    # 2. Walk the sources to gather assets using the new energy-budgeted manifest logic.
    all_ops = []
    total_frontier = []
    total_energy_spent = 0

    for source_str in sources:
        source = Path(source_str)
        scan_result: Optional[ScanResult] = None
        if source.is_dir():
            scan_result = manifest_from_directory(
                source,
                energy_budget=energy_budget - total_energy_spent,
                filter_fn=lambda p: include_all or not p.name.startswith('.')
            )
            # Prepend the directory name to the path for all ops
            ops_to_add = [op._replace(
                path=f"{source.name}/{op.path}") for op in scan_result.manifest.ops]
            all_ops.extend(ops_to_add)

        elif source.is_file():
            if (not include_all and source.name.startswith('.')) or source.name.endswith('.genyx.zip'):
                continue

            try:
                data = source.read_bytes()
                file_energy = len(data)
                if total_energy_spent + file_energy <= energy_budget:
                    all_ops.append(FileOp(source.name, data))
                    total_energy_spent += file_energy
                else:
                    total_frontier.append(source)
            except IOError as e:
                print(f"Warning: Could not read file '{source}': {e}")

        if scan_result:
            total_energy_spent += scan_result.energy_spent
            total_frontier.extend(scan_result.frontier)

    if total_frontier:
        print(f"\n[Energy Budget Reached] Frontier of unexplored paths:")
        for path in total_frontier[:10]:  # Print first 10
            print(f"  - {path}")
        if len(total_frontier) > 10:
            print(f"  ... and {len(total_frontier) - 10} more.")

    # 3. Create a genesis grammar
    grammar = Grammar()

    # 4. Add a default report prompt
    all_ops.append(FileOp('registry/report.txt', b"[SYSTEM_SUMMARY]"))

    # 5. Assemble and Save the Genome using the blit operation
    # First, create the in-memory genome to generate config/grammar files
    temp_assets = {op.path: op.data for op in all_ops}
    genesis_genome = Genome(
        config=config, grammar=grammar, extra_assets=temp_assets)

    if bootstrap:
        print("Bootstrapping shallow grammar from assets...")
        try:
            genesis_genome.build_shallow_grammar()
        except Exception as e:
            print(f"Warning: bootstrap grammar failed: {e}")

    # Now, get the final manifest including config and grammar, then blit
    final_manifest = genesis_genome.to_manifest()
    blit_to_reflex(final_manifest, Path(output_path))

    print(f"\nDone. Total energy spent: {total_energy_spent} bytes.")

def add_opcode(genome_path: str, script_path: str):
    """Compiles a Python script and adds it to a Reflex genome."""
    genome_p = Path(genome_path)
    script_p = Path(script_path)
    if not genome_p.exists():
        print(f"Error: Genome file not found at '{genome_path}'")
        return
    if not script_p.exists():
        print(f"Error: Script file not found at '{script_path}'")
        return

    print(f"Compiling '{script_path}' and adding to '{genome_path}'...")
    
    bytecode = b''
    try:
        # We compile to bytes in memory
        bytecode = py_compile.compile(
            str(script_p), dfile=str(script_p), doraise=True, optimize=2)
    except py_compile.PyCompileError as e:
        print(f"Error compiling script: {e}")
        return

    # Unpack the existing reflex archive into a dictionary of files
    try:
        existing_blob = genome_p.read_bytes()
        files = unpack_reflex(existing_blob)
    except (IOError, ValueError) as e:
        print(f"Error reading or unpacking existing genome: {e}")
        return

    # Add the new opcode
    op_name = script_p.stem
    op_asset_path = f"ops/{op_name}.pyc"
    files[op_asset_path] = bytecode

    # Create a new manifest from the updated file dictionary
    new_ops = [FileOp(path, data) for path, data in files.items()]
    new_manifest = ZipFlexManifest(
        ops=new_ops, comment=b'Eonyx Genome (Updated)')

    # Blit the new manifest back to the archive
    blit_to_reflex(new_manifest, genome_p)
            
    print(f"Successfully added opcode as '{op_asset_path}'")

def inspect_genome(path: str):
    """Prints a summary of a .genome file's contents."""
    if not Path(path).exists():
        print(f"Error: File not found at '{path}'")
        return

    print(f"--- Inspecting Genome: {Path(path).name} ---")
    
    # Genome.from_path is now the canonical way to load, handling errors internally.
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
        print(
            f"  Top 10 Unigrams: {', '.join([f'{t[0]} ({t[1]})' for t in top_10])}")
    else:
        print("  Unigram counts are empty.")
            
    # Print Extra Assets
    print("\n[Extra Assets]")
    if genome.extra_assets:
        for name, data in sorted(genome.extra_assets.items()):
            print(f"  - {name} ({len(data)} bytes)")
    else:
        print("  No extra assets found.")

    # Print Alerts from loading
    if genome.alerts:
        print("\n[Loading Alerts]")
        print(f"  {genome.alerts}")


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
    parser_create.add_argument('-e', '--energy', type=int, default=10 * 1024 *
                               1024, help='Energy budget in bytes for scanning sources (default: 10MB).')

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
        create_genesis(args.sources, output_path, args.recursive,
                       args.all, energy_budget=args.energy)
    elif args.command == 'add-op':
        add_opcode(args.genome, args.script)
    elif args.command == 'inspect':
        inspect_genome(args.path)

if __name__ == '__main__':
    # Ensure the project root is on the Python path to find the eonyx module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
