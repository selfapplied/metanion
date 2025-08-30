#!/usr/bin/env python3
import argparse
import os
import sys
import time
import tempfile
import shutil
from fme_core import FractalMarkovAlleleEngine
from genome import Genome, EngineConfig, Grammar
from typing import Optional, Tuple, Dict, List
from aspire import Opix
from zip import extract_zip_deflate_streams, blit_to_reflex_bytes, DirectoryScanner, FileOp, unpack_reflex
from pathlib import Path
from collections import defaultdict
from fme_engine import EngineResult


def create_genesis_bytes(sources: list, recursive: bool, include_all: bool, bootstrap: bool = True, energy_budget: int = 10 * 1024 * 1024) -> bytes:
    """Creates a new genesis genome from source files and directories and returns it as bytes."""

    states = ['carry', 'borrow', 'drift']
    priors = defaultdict(dict)
    for s_from in states:
        for s_to in states:
            priors[s_from][s_to] = 1.0 / len(states)

    config = EngineConfig(
        symbols=states,
        priors={k: dict(v) for k, v in priors.items()}
    )

    all_ops = []
    total_energy_spent = 0

    for source_str in sources:
        source = Path(source_str)
        if source.is_dir():
            scanner = DirectoryScanner(
                source,
                energy_budget=energy_budget - total_energy_spent,
                filter_fn=lambda p: include_all or not p.name.startswith('.')
            )
            scan_result = scanner.scan()
            ops_to_add = [op._replace(
                path=f"{source.name}/{op.path}") for op in scan_result.manifest.ops]
            all_ops.extend(ops_to_add)
            total_energy_spent += scan_result.energy_spent
        elif source.is_file():
            if (not include_all and source.name.startswith('.')) or source.name.endswith('.genyx.zip'):
                continue
            try:
                data = source.read_bytes()
                file_energy = len(data)
                if total_energy_spent + file_energy <= energy_budget:
                    all_ops.append(FileOp(source.name, data))
                    total_energy_spent += file_energy
            except IOError:
                pass

    grammar = Grammar()
    all_ops.append(FileOp('registry/report.txt', b"[SYSTEM_SUMMARY]"))

    temp_assets = {op.path: op.data for op in all_ops}
    genesis_genome = Genome(
        config=config, grammar=grammar, extra_assets=temp_assets)

    if bootstrap:
        try:
            genesis_genome.build_shallow_grammar()
        except Exception:
            pass

    final_manifest = genesis_genome.to_manifest()
    return blit_to_reflex_bytes(final_manifest)


def _genome_from_directory(dir_path: str) -> Genome:
    """Create an in-memory Genome from all files under a directory."""
    extra_assets = {}
    # Walk directory and ingest every regular file
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            # Skip generated genome zips to avoid recursion
            if name.endswith('.genyx.zip'):
                continue
            file_path = os.path.join(root, name)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                rel_key = os.path.relpath(file_path, dir_path)
                extra_assets[rel_key] = content
            except Exception:
                continue

    # Minimal default genome; grammar bootstrapped later
    g = Genome(config=generation_defaults(),
               grammar=grammar_defaults(), extra_assets=extra_assets)
    # Seed a shallow grammar from ingested bytes and filenames
    try:
        g.build_shallow_grammar()
    except Exception:
        pass
    return g


def generation_defaults():
    # Late import to avoid circulars in type-checkers
    from genome import EngineConfig
    return EngineConfig()


def grammar_defaults():
    from genome import Grammar
    return Grammar()


def _report_run_summary(engine: FractalMarkovAlleleEngine, total_stats: list, last_result: Optional[dict] = None):
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
        if engine.genome:
            g_carry = engine.genome.carry()
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
    if engine.genome:
        context_grammar = engine.genome.grammar
        max_freq = max(context_grammar.unigram_counts.values()
                   ) if context_grammar.unigram_counts else 1
        for key, value in facts.items():
            fact_token = f"[{key.upper()}:{value}]"
            # Make it very prominent
            context_grammar.unigram_counts[fact_token] = max_freq * 2

    # 3. Load the report prompt from the genome.
    try:
        if engine.genome:
            prompt = engine.genome.registry_get(
                'registry/report.txt', text=True)
            if not prompt:
                raise ValueError("Empty prompt")
        else:
            prompt = "[SYSTEM_SUMMARY]"
    except Exception:
        prompt = "[SYSTEM_SUMMARY]"  # Fallback

    # 4. Generate the report.
    # Prefer the actual last_result state if available; fallback to neutral.
    start_state = last_result if isinstance(last_result, dict) else {'final_qs': [
        [1.0, 0.0, 0.0, 0.0]], 'color_trail': []}
    print("\nSummary:")
    report = engine.generate_text(prompt, result=start_state)
    print(report)


def _persist_learning(engine: FractalMarkovAlleleEngine, genome_path: str):
    """Saves newly minted symbols from the engine back into the genome."""
    genome = engine.genome
    minted = getattr(engine, 'minted_symbols', []) or []
    if not minted:
        return

    try:
        import msgpack
        if engine.genome:
            engine.genome.registry_set(engine.genome.reg_sym, msgpack.packb(
                {'symbols': minted}, use_bin_type=True))
            engine.genome.save(genome_path)
            log('info',
                f"Saved {len(minted)} new symbols to '{engine.genome.reg_sym}'")
    except Exception as e:
        log('error', f"Could not persist symbols: {e}")


def _generate_output(engine: FractalMarkovAlleleEngine, last_result: Optional[dict], seed_text: str):
    """Generates and prints text based on the final state of the engine."""
    if not seed_text:
        return

    transformed_state = last_result if isinstance(last_result, dict) else {}
    if not transformed_state:
        transformed_state = {'final_qs': [
            [1.0, 0.0, 0.0, 0.0]], 'color_trail': []}

    log('info', "\n--- Generated Text ---")
    text = engine.generate_text(seed_text, result=transformed_state)
    print(text)

# Simple logging with print


def log(level, message):
    print(f"[{level.upper()}] {message}")


def render_epilogue(engine: FractalMarkovAlleleEngine, total_stats: list, last_result: Optional[dict], seed_text: Optional[str], genome_path: str):
    """Renders the final summary and generative output after a genome run."""
    _report_run_summary(engine, total_stats, last_result)
    _persist_learning(engine, genome_path)
    # The _generate_output function is now obsolete, as the summary is generative.
    # _generate_output(engine, last_result, seed_text)


def _initialize_engine(genome_path: str) -> Optional[FractalMarkovAlleleEngine]:
    """Loads, validates, and prepares a genome and its FractalMarkovAlleleEngine."""
    if not os.path.exists(genome_path):
        log('error', f"Genome file not found at '{genome_path}'.")
        return None

    # Allow directories: ingest all files into an in-memory genome
    if os.path.isdir(genome_path):
        genome = _genome_from_directory(genome_path)
    else:
        genome = Genome.from_path(genome_path)

    # Load opix registry macros
    try:
        data = genome.registry_get(genome.reg_opx)
        if isinstance(data, (bytes, bytearray)):
            # load_opix_registry(bytes(data)) # This line was removed as per the new_code
            pass  # Assuming load_opix_registry is no longer needed or handled differently
    except Exception:
        pass

    # Bootstrap grammar if empty and apply viral ops
    if not genome.grammar.unigram_counts:
        genome.build_shallow_grammar()
    try:
        genome.apply_viral_ops()
    except Exception:
        pass
    try:
        genome.save(genome_path)
    except Exception:
        genome.alerts['💾!'] += 1

    # Pass an empty template for now
    engine = FractalMarkovAlleleEngine(template={})
    engine.genome = genome
    return engine


def _persist_live_state(engine: FractalMarkovAlleleEngine, genome_path: str):
    """Saves the engine's worlds and the full genome during a run."""
    try:
        if engine.genome:
            engine.genome.save(genome_path)
    except Exception:
        pass  # Fail silently during live-save


def run_fme_engine(genome_path: str, seed_text: Optional[str], timeout: Optional[float]) -> Optional[EngineResult]:
    """Initializes and runs the FractalMarkovAlleleEngine."""

    # Create a minimal config dictionary.
    default_config = {
        "levels": {"count": 4},
        "stochastic": {"seed": 1337, "noise": 0.01},
        "fractal": {
            "scales": {"l0": 1.0, "l1": 1.0, "l2": 1.0, "l3": 1.0},
            "self_similarity": 0.6,
            "branch_factor": 3,
        },
        "alphabet": {"symbols": ["carry", "borrow", "drift"]},
        "priors": {
            "carry": {"carry": 0.7, "borrow": 0.2, "drift": 0.1},
            "borrow": {"carry": 0.2, "borrow": 0.7, "drift": 0.1},
            "drift": {"carry": 0.1, "borrow": 0.2, "drift": 0.7},
        },
        "emissions": {
            "carry": {"axis": "x", "amplitude": [0.7, 1.2]},
            "borrow": {"axis": "y", "amplitude": [0.7, 1.2]},
            "drift": {"axis": "z", "amplitude": [0.3, 0.8]},
        },
        "seed": {
            "alpha": 0.1,
            "epsilon0": 1.0,
            "beta": 1.0,
            "kappa": 0.25,
            "delta_spawn": 0.05,
            "basis_weights": [0.34, 0.33, 0.33],
        },
    }

    engine = FractalMarkovAlleleEngine(template=default_config)

    result: Optional[EngineResult] = None
    if os.path.isdir(genome_path):
        zip_bytes = create_genesis_bytes(
            sources=[genome_path], recursive=True, include_all=False)
        # In-memory Reflex archive: use unpack_reflex
        try:
            extra_assets = unpack_reflex(zip_bytes)
        except Exception:
            extra_assets = {}
        if not extra_assets:
            log('error', "Could not extract assets from in-memory genome bytes.")
            return
        result = engine.train_from_assets(assets=extra_assets)
    else:
        # If it's a zip file, extract its assets into memory and run.
        stats = Opix()
        extra_assets, _ = extract_zip_deflate_streams(genome_path, stats)
        if not extra_assets:
            # Fallback to Reflex unpack
            try:
                blob = Path(genome_path).read_bytes()
                extra_assets = unpack_reflex(blob)
            except Exception:
                extra_assets = {}
        if not extra_assets:
            log('error', f"Could not extract assets from {genome_path}. Stats: {stats}")
            return
        result = engine.train_from_assets(assets=extra_assets)

    if seed_text and result:
        print("\n--- Generated Text ---")
        # Style Vector: direct field access, no internal exception handling
        # Type hints make the namedtuple structure explicit
        generated_text: str = engine.generate_text(
            seed_text, 
            banks=result.banks, 
            finals=result.final_q
        )
        print(generated_text)
    elif seed_text:
        print("No training result available for text generation")


def run_genome(genome_path: str, seed_text: Optional[str], timeout: Optional[float]):
    """Loads a genome and runs the VM on all of its internal assets."""
    # This function is now a wrapper around the FME engine runner.
    # The logic of iterating through assets is handled inside `train_from_directory`.
    run_fme_engine(genome_path, seed_text, timeout)


def resolve_genome_and_seed(path_arg: Optional[str], seed_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Determdines the correct genome path and seed text from CLI arguments."""
    genome_path = path_arg
    seed_text = seed_arg

    # If only one positional was provided and it's not a valid path, treat it as the prompt
    if genome_path is not None and seed_text is None and not os.path.exists(genome_path):
        seed_text = genome_path
        genome_path = None

    if genome_path is None:
        # Define search paths: current dir, script's dir.
        script_dir = os.path.abspath(os.path.dirname(__file__))
        search_dirs = [
            os.path.abspath('.'),
            script_dir,
        ]

        # Deduplicate search paths
        unique_search_dirs = sorted(list(set(search_dirs)))

        prefer, fallback = [], []
        for d in unique_search_dirs:
            try:
                for f in os.listdir(d):
                    if f.endswith('.genyx.zip'):
                        prefer.append(os.path.join(d, f))
                    elif f.endswith('.zip'):
                        fallback.append(os.path.join(d, f))
            except Exception:
                pass

        # Deduplicate found files and prioritize .genyx.zip
        candidates = sorted(list(set(prefer))) + \
            sorted(list(set(f for f in fallback if f not in prefer)))

        if candidates:
            genome_path = candidates[0]

    return genome_path, seed_text


def main():
    parser = argparse.ArgumentParser(
        description="eonyx: The Eonyx Genome Virtual Machine")
    parser.add_argument('genome_path', nargs='?', default=None,
                        help='Optional path to the .genyx.zip file or directory to run.')
    parser.add_argument('seed', nargs='?', default=None,
                        help='Optional seed text for generation.')
    parser.add_argument('--timeout', "-t", type=float, default=10.0,
                        help='Maximum time in seconds for the learning process per asset. Default: 10s.')

    args = parser.parse_args()

    genome_path, seed_text = resolve_genome_and_seed(
        args.genome_path, args.seed)

    if not genome_path and seed_text:
        # If no genome found but seed text provided, create a new genome from current directory
        genome_path = '.'
    
    if not genome_path:
        log('error', "No zip genome file (.genyx.zip or .zip) or directory found nearby.")
        return
    
    run_genome(genome_path, seed_text, args.timeout)

if __name__ == '__main__':
    # Add `eonyx` to the path to allow for imports from other files in the package.
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '')))
    main()
