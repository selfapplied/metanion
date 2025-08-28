#!/usr/bin/env python3
import argparse
import os
import sys
import time
import tempfile
import shutil
from ce1 import CE1Core
from genome import Genome
from typing import Optional
from aspire import load_opix_registry, Aspirate


def run_genome(genome_path: str, seed_text: Optional[str], timeout: Optional[float]):
    """Loads a genome and runs the VM on all of its internal assets."""
    # Simple logging with print
    def log(level, message):
        print(f"[{level.upper()}] {message}")

    if not os.path.exists(genome_path):
        log('error', f"Genome file not found at '{genome_path}'.")
        return

    log('info', f"Loading genome from '{genome_path}'...")
    genome = Genome.from_path(genome_path)
    # Load opix registry macros (live)
    try:
        data = genome.registry_get(genome.reg_opx)
        if isinstance(data, (bytes, bytearray)):
            load_opix_registry(bytes(data))
    except Exception:
        pass
    # Shallow grammar bootstrap if empty
    if not genome.grammar.unigram_counts:
        genome.build_shallow_grammar()
        # Persist immediately so subsequent runs don't warn
        try:
            genome.save(genome_path)
        except Exception:
            pass
    engine = CE1Core(genome)

    # The runner now iterates through all assets, telling the VM to learn from each one.
    if not genome.extra_assets:
        log('warn', "No internal assets found for learning.")
        try:
            # Record alert glyph for missing assets
            genome.alerts['∅'] += 1
        except Exception:
            pass

    # Log how many assets were loaded
    try:
        loaded_count = len(genome.extra_assets)
        log('info', f"Loaded {loaded_count} assets from ZIP")
    except Exception:
        pass

    # Filter out meta entries
    asset_names = [
        n for n in sorted(genome.extra_assets.keys())
        if n not in ('config.msgpack', 'grammar.msgpack') and not n.startswith('registry/')
    ]

    total_stats = []
    last_result = None
    for asset_name in asset_names:
        result = engine.run_and_learn_asset(
            asset_name, time_budget_secs=timeout)
        last_result = result or last_result
        stats = (result or {}).get('stats')
        if stats:
            total_stats.append(stats)
        # Live-save updates after each asset
        try:
            # Persist worlds registry live as well
            try:
                engine.save_worlds()
            except Exception:
                pass
            genome.save(genome_path)
        except Exception:
            pass

    # --- Final Summary ---
    if total_stats:
        final_mean_surprise = sum(s['mean_surprise']
                                  for s in total_stats) / len(total_stats)
        carried = getattr(engine, 'last_minted_symbol', None)
        suffix = f" last: {carried}" if carried else ''
        # World summary
        try:
            worlds = getattr(engine, 'worlds', {})
            active = (worlds or {}).get('active', 'default')
            w = (worlds or {}).get('worlds', {}).get(active, {})
            wmask = int(w.get('typemask', 0))
            wsize = int(w.get('typemask_size', 0))
            wcomplete = bool(w.get('complete', False))
            wglyph = str(w.get('glyph', '')) if wcomplete else ''
            wcolor = tuple(w.get('color', (0.0, 0.0, 0.0)))
            wmark = wglyph or ('✓' if wcomplete else '…')
            wsfx = f" world:{active} {wmark} mask:{wmask:08b}/{wsize} hsv:{wcolor}"
        except Exception:
            wsfx = ''
        # Build aspiration carry using module-owned shapes
        try:
            # Genome carry: assets (§) + alerts overlay
            gcarry = genome.carry()
            # CE1 carry: ∑ and ∞ from event stream
            ecarry = getattr(engine, 'carry_ce1', None)
            carry_asp = gcarry.overlay_with(ecarry) if isinstance(
                ecarry, Aspirate) else gcarry
            carry = str(carry_asp)
            csfx = f" carry: {carry}" if carry else ''
        except Exception:
            csfx = ''
        log('info',
            f"\n--- Run Complete --- assets: {len(total_stats)} mean: {final_mean_surprise:.4f}{suffix}{wsfx}{csfx}")
    else:
        # Even without stats, print minimal run complete with carried symbol if any
        carried = getattr(engine, 'last_minted_symbol', None)
        suffix = f" last: {carried}" if carried else ''
        log('info', f"\n--- Run Complete --- assets: 0{suffix}")
    # Print minted symbols list (always if present)
    minted_list = getattr(engine, 'minted_symbols', []) or []
    if minted_list:
        log('info', f"symbols: {' '.join(minted_list)}")
    # Print any alerts collected on genome
    try:
        alerts = getattr(genome, 'alerts', None)
        if alerts and len(alerts):
            log('info',
                f"alerts: {alerts.hstack(separator=' ', format_spec='tight,order=count')}")
    except Exception:
        pass

    # Persist minted symbols into registry (msgpack list of strings)
    try:
        import msgpack
        minted = getattr(engine, 'minted_symbols', []) or []
        if minted:
            genome.registry_set(genome.reg_sym, msgpack.packb(
                {'symbols': minted}, use_bin_type=True))
            # Optionally save back to disk
            genome.save(genome_path)
            try:
                reg_path = genome.reg_sym
            except Exception:
                reg_path = 'registry/sym.mpk'
            log('info', f"Saved {len(minted)} new symbols to {reg_path}")
    except Exception as e:
        log('warn', f"Could not persist minted symbols: {e}")

    # Optional text generation when a seed is provided (after first full pass)
    if seed_text is not None:
        transformed_state = last_result if isinstance(
            last_result, dict) else {}
        if not transformed_state:
            transformed_state = {'final_qs': [
                [1.0, 0.0, 0.0, 0.0]], 'color_trail': []}
        text = engine.generate_text(seed_text, transformed_state)
        log('info', "\n--- Generated Text ---")
        print(text)


def main():
    parser = argparse.ArgumentParser(description="eonyx: The Eonyx Genome Virtual Machine")
    parser.add_argument('genome_path', nargs='?', default=None, help='Optional path to the .genyx.zip file to run. If not provided, will search for one in the current directory.')
    parser.add_argument('seed', nargs='?', default=None, help='Optional seed text for generation. If not provided, uses the most common word in the grammar.')
    parser.add_argument('--timeout', "-t", type=float, default=10.0,
                        help='Maximum time in seconds for the learning process. Default: 10s.')
    
    args = parser.parse_args()
    
    genome_path = args.genome_path
    seed_text = args.seed

    # If only one positional was provided and it's not a genome path, treat it as the prompt
    if genome_path is not None and seed_text is None and (not os.path.exists(genome_path)) and (not genome_path.endswith('.genyx.zip')):
        seed_text = genome_path
        genome_path = None

    if genome_path is None:
        # Search for a genome file nearby: prefer .genyx.zip, else any .zip, in . then ..
        print("Searching for genome zip nearby (preferring .genyx.zip)...")
        search_dirs = ['.', '..']
        prefer, fallback = [], []
        for d in search_dirs:
            try:
                for f in os.listdir(d):
                    if f.endswith('.genyx.zip'):
                        prefer.append(os.path.join(d, f))
                    elif f.endswith('.zip'):
                        fallback.append(os.path.join(d, f))
            except Exception:
                pass
        candidates = prefer or fallback
        if not candidates:
            print("No zip genome file (.genyx.zip or .zip) found nearby.")
            return
        genome_path = candidates[0]
        print(f"Found and using genome: {genome_path}")
    
    run_genome(genome_path, seed_text, args.timeout)

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
