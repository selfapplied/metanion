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
from aspire import load_opix_registry, Opix


def _report_run_summary(engine: CE1Core, total_stats: list, last_result: Optional[dict] = None):
    """Generates a run summary by feeding the final state to the engine itself."""
    genome = engine.genome

    # 1. Gather all reportable facts.
    facts = {}
    if total_stats:
        facts['assets_count'] = len(total_stats)
        facts['mean_surprise'] = sum(s['mean_surprise']
                                     for s in total_stats) / len(total_stats)

    carried = getattr(engine, 'last_minted_symbol', None)
    if carried:
        facts['last_minted'] = carried

    try:  # World info
        worlds = getattr(engine, 'worlds', {})
        active = (worlds or {}).get('active', 'default')
        w = (worlds or {}).get('worlds', {}).get(active, {})
        facts['world_name'] = active
        facts['world_glyph'] = str(
            w.get('glyph', '')) if w.get('complete') else '…'
    except Exception:
        pass

    try:  # Carry string
        g_carry = genome.carry()
        e_carry = getattr(engine, 'carry_ce1', None)
        carry_asp = g_carry.overlay_with(
            e_carry) if isinstance(e_carry, Opix) else g_carry
        if carry_asp:
            facts['carry_string'] = str(carry_asp)
    except Exception:
        pass

    minted_list = getattr(engine, 'minted_symbols', []) or []
    if minted_list:
        facts['symbols_string'] = ' '.join(minted_list)

    try:  # Alerts string
        alerts = getattr(genome, 'alerts', None)
        if alerts:
            facts['alerts_string'] = alerts.hstack(
                separator=' ', format_spec='tight,order=count')
    except Exception:
        pass

    # 2. Create a high-frequency context in the grammar.
    # This makes the facts "visible" to the generative walker.
    context_grammar = genome.grammar
    max_freq = max(context_grammar.unigram_counts.values()
                   ) if context_grammar.unigram_counts else 1
    for key, value in facts.items():
        fact_token = f"[{key.upper()}:{value}]"
        # Make it very prominent
        context_grammar.unigram_counts[fact_token] = max_freq * 2

    # 3. Load the report prompt from the genome.
    try:
        prompt = genome.registry_get('registry/report.txt', text=True)
        if not prompt:
            raise ValueError("Empty prompt")
    except Exception:
        prompt = "[SYSTEM_SUMMARY]"  # Fallback

    # 4. Generate the report.
    # Prefer the actual last_result state if available; fallback to neutral.
    start_state = last_result if isinstance(last_result, dict) else {'final_qs': [
        [1.0, 0.0, 0.0, 0.0]], 'color_trail': []}
    print("\nSummary:")
    report = engine.generate_text(prompt, start_state)
    print(report)


def _persist_learning(engine: CE1Core, genome_path: str):
    """Saves newly minted symbols from the engine back into the genome."""
    genome = engine.genome
    minted = getattr(engine, 'minted_symbols', []) or []
    if not minted:
        return

    try:
        import msgpack
        genome.registry_set(genome.reg_sym, msgpack.packb(
            {'symbols': minted}, use_bin_type=True))
        genome.save(genome_path)
        log('info', f"Saved {len(minted)} new symbols to '{genome.reg_sym}'")
    except Exception as e:
        log('error', f"Could not persist symbols: {e}")


def _generate_output(engine: CE1Core, last_result: Optional[dict], seed_text: str):
    """Generates and prints text based on the final state of the engine."""
    if not seed_text:
        return

    transformed_state = last_result if isinstance(last_result, dict) else {}
    if not transformed_state:
        transformed_state = {'final_qs': [
            [1.0, 0.0, 0.0, 0.0]], 'color_trail': []}

    log('info', "\n--- Generated Text ---")
    text = engine.generate_text(seed_text, transformed_state)
    print(text)

# Simple logging with print


def log(level, message):
    print(f"[{level.upper()}] {message}")


def render_epilogue(engine: CE1Core, total_stats: list, last_result: Optional[dict], seed_text: Optional[str], genome_path: str):
    """Renders the final summary and generative output after a genome run."""
    _report_run_summary(engine, total_stats, last_result)
    _persist_learning(engine, genome_path)
    # The _generate_output function is now obsolete, as the summary is generative.
    # _generate_output(engine, last_result, seed_text)


def _initialize_engine(genome_path: str) -> Optional[CE1Core]:
    """Loads, validates, and prepares a genome and its CE1Core engine."""
    if not os.path.exists(genome_path):
        log('error', f"Genome file not found at '{genome_path}'.")
        return None

    # log('info', f"Loading genome from '{genome_path}'...")
    genome = Genome.from_path(genome_path)

    # Load opix registry macros
    try:
        data = genome.registry_get(genome.reg_opx)
        if isinstance(data, (bytes, bytearray)):
            load_opix_registry(bytes(data))
    except Exception:
        pass

    # Bootstrap grammar if empty and save the upgraded genome
    if not genome.grammar.unigram_counts:
        # log('info', "Grammar is empty, bootstrapping from assets...")
        genome.build_shallow_grammar()
        try:
            genome.save(genome_path)
            # log('info', f"Saved bootstrapped grammar to '{genome_path}'.")
        except Exception:
            genome.alerts['💾!'] += 1

    return CE1Core(genome)


def _persist_live_state(engine: CE1Core, genome_path: str):
    """Saves the engine's worlds and the full genome during a run."""
    try:
        engine.save_worlds()
        engine.genome.save(genome_path)
    except Exception:
        pass  # Fail silently during live-save


def run_genome(genome_path: str, seed_text: Optional[str], timeout: Optional[float]):
    """Loads a genome and runs the VM on all of its internal assets."""
    engine = _initialize_engine(genome_path)
    if not engine:
        return

    genome = engine.genome
    if not genome.extra_assets:
        log('warn', "No internal assets found for learning.")
        genome.alerts['∅'] += 1

    # log('info', f"Loaded {len(genome.extra_assets)} assets from ZIP")
    asset_names = sorted(genome.extra_assets.keys())

    total_stats = []
    last_result = None
    for asset_name in asset_names:
        result = engine.run_and_learn_asset(
            asset_name, time_budget_secs=timeout)
        last_result = result or last_result
        stats = (result or {}).get('stats')
        if stats:
            total_stats.append(stats)

        _persist_live_state(engine, genome_path)

    render_epilogue(engine, total_stats, last_result, seed_text, genome_path)


def main():
    parser = argparse.ArgumentParser(description="eonyx: The Eonyx Genome Virtual Machine")
    parser.add_argument('genome_path', nargs='?', default=None, help='Optional path to the .genyx.zip file to run. If not provided, will search for one in the current directory.')
    parser.add_argument('seed', nargs='?', default=None, help='Optional seed text for generation. If not provided, uses the most common word in the grammar.')
    parser.add_argument('--timeout', "-t", type=float, default=10.0,
                        help='Maximum time in seconds for the learning process. Default: 10s.')
    parser.add_argument('--verbose', "-v", action='store_true',
                        help='Enable verbose logging.')
    
    args = parser.parse_args()
    
    genome_path = args.genome_path
    seed_text = args.seed

    # If only one positional was provided and it's not a genome path, treat it as the prompt
    if genome_path is not None and seed_text is None and (not os.path.exists(genome_path)) and (not genome_path.endswith('.genyx.zip')):
        seed_text = genome_path
        genome_path = None

    if genome_path is None:
        # Search for a genome file nearby: prefer .genyx.zip, else any .zip, in . then ..
        if args.verbose:
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
        if args.verbose:
            print(f"Found and using genome: {genome_path}")
    
    run_genome(genome_path, seed_text, args.timeout)

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
