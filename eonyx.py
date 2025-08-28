#!/usr/bin/env python3
import argparse
import os
import sys
import time
import tempfile
import shutil
from ce1_core import CE1_Core
from gennome import Genome
from typing import Optional

class Logger:
    def __init__(self, levels):
        self.levels = levels
    
    def log(self, level, message):
        if level in self.levels:
            print(f"[{level.upper()}] {message}")

def run_genome(genome_path: str, seed_text: Optional[str], timeout: Optional[float]):
    """Loads a genome and runs the VM on all of its internal assets."""
    logger = Logger({'info', 'warn', 'error'})

    if not os.path.exists(genome_path):
        logger.log('error', f"Genome file not found at '{genome_path}'.")
        return

    logger.log('info', f"Loading genome from '{genome_path}'...")
    try:
        genome = Genome.from_path(genome_path)
        engine = CE1_Core(genome)

        # The runner now iterates through all assets, telling the VM to learn from each one.
        if not genome.extra_assets:
            logger.log('warn', "No internal assets found for learning. Cannot run.")
            return

        total_stats = []
        last_result = None
        for asset_name in sorted(genome.extra_assets.keys()):
            # The timeout is now applied per-asset.
            result = engine.run_and_learn_asset(asset_name, time_budget_secs=timeout)
            last_result = result or last_result
            if result and 'stats' in result and result['stats']:
                total_stats.append(result['stats'])
                logger.log('info', f"Finished asset '{asset_name}': μΔ={result['stats']['mean_surprise']:.4f}, σΔ={result['stats']['std_surprise']:.4f}")

        # --- Final Summary ---
        if total_stats:
            final_mean_surprise = sum(s['mean_surprise'] for s in total_stats) / len(total_stats)
            logger.log('info', f"\n--- Run Complete ---")
            logger.log('info', f"Processed {len(total_stats)} assets.")
            logger.log('info', f"Final Average Mean Surprise (μΔ): {final_mean_surprise:.4f}")

        # Optional text generation when a seed is provided
        if seed_text is not None:
            transformed_state = last_result if isinstance(last_result, dict) else {}
            if not transformed_state:
                transformed_state = {'final_qs': [[1.0, 0.0, 0.0, 0.0]], 'color_trail': []}
            try:
                text = engine.generate_text(seed_text, transformed_state)
                logger.log('info', "\n--- Generated Text ---")
                print(text)
            except Exception as gen_err:
                logger.log('error', f"Generation failed: {gen_err}")

    except Exception as e:
        logger.log('error', f"An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="eonyx: The Eonyx Genome Virtual Machine")
    parser.add_argument('genome_path', nargs='?', default=None, help='Optional path to the .genyx.zip file to run. If not provided, will search for one in the current directory.')
    parser.add_argument('seed', nargs='?', default=None, help='Optional seed text for generation. If not provided, uses the most common word in the grammar.')
    parser.add_argument('--timeout', type=float, default=10.0, help='Maximum time in seconds for the learning process. Default: 10s.')
    
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
