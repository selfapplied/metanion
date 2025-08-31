from typing import List, Dict, Optional, Any, Tuple
import os
import hashlib
import numpy as np
import time
from collections import Counter
import math

def _assess_content_information(engine, content: str) -> float:
    """Quick assessment of content's information content."""
    try:
        if not content.strip():
            return 0.0

        features = engine._content_features(content)
        length = features.length
        
        basic_tokens = engine._tok(content)
        unique_tokens = len(set(basic_tokens))
        token_diversity = unique_tokens / max(len(basic_tokens), 1)
        char_entropy = features.entropy

        total_score = (
            token_diversity * 0.4 +
            (char_entropy / 8.0) * 0.4 +
            min(length / 1024.0, 1.0) * 0.2
        )
        
        return total_score
    except Exception:
        return 0.0

def _prioritize_assets_jit(engine, assets: Dict[str, bytes], top_percentage: float = 1.0) -> List[str]:
    """Prioritize assets and return their content as strings."""
    asset_scores = []
    for name, data in assets.items():
        try:
            content = data.decode('utf-8', errors='ignore')
            if not content.strip():
                continue
            score = _assess_content_information(engine, content)
            asset_scores.append((content, score))
        except Exception:
            continue

    asset_scores.sort(key=lambda x: x[1], reverse=True)

    num_high_value = max(1, int(len(asset_scores) * top_percentage))
    high_value_content = [content for content, score in asset_scores[:num_high_value]]
    
    return high_value_content

def train_from_assets(engine, assets: Dict[str, bytes], blend=0.5, excavation_iterations=3, enable_jit_excavation=True):
    """
    Trains the engine from in-memory assets, updating its grammar and running the simulation.
    """
    print(f"🚀 Starting training process from {len(assets)} in-memory assets")

    if not assets:
        print("No assets found to train on.")
        return

    if enable_jit_excavation:
        training_content = _prioritize_assets_jit(engine, assets)
    else:
        training_content = [data.decode('utf-8', errors='ignore') for data in assets.values()]

    # Learn tokenization and populate vocab
    engine._learn_tokenization_excavation(training_content, 0)

    print(f"🔍 Selected content from {len(training_content)} assets for training.")

    all_tokens = []
    for content in training_content:
        tokens = engine._tok(content)
        all_tokens.extend(tokens)

    if len(all_tokens) < 2:
        print("Not enough tokens to train.")
        return

    # Update grammar
    for t in all_tokens:
        engine.unigram_counts[t] = engine.unigram_counts.get(t, 0) + 1
    
    for a, b in zip(all_tokens[:-1], all_tokens[1:]):
        if a not in engine.bigram_counts:
            engine.bigram_counts[a] = {}
        engine.bigram_counts[a][b] = engine.bigram_counts[a].get(b, 0) + 1
    
    print(f"📚 Grammar updated with {len(all_tokens)} tokens.")

    # Run the engine
    print("🧠 Running FME engine simulation...")
    result = engine.run(steps=engine.planned_steps)
    
    banks = result.banks
    if banks:
        alleles_found = sum(len(b) for b in banks)
        print(f"🧬 Found {alleles_found} alleles.")

    print("✅ Training complete.")
    return result

def train_from_directory(engine, directory='.', blend=0.5, include_hidden=False, incremental_files=None, excavation_iterations=3, enable_jit_excavation=True):
    """
    Trains the engine from files in a directory by loading them into memory.
    """
    assets = {}
    file_paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if not include_hidden and name.startswith('.'):
                continue
            file_paths.append(os.path.join(root, name))

    for path in file_paths:
        try:
            with open(path, 'rb') as f:
                assets[path] = f.read()
        except Exception:
            continue
            
    return train_from_assets(engine, assets, blend, excavation_iterations, enable_jit_excavation)


def load_learned_if_cached(engine, directory='.') -> tuple[bool, Optional[Dict[str, tuple[str, str]]]]:
    # This function is a placeholder as TOML loading is removed.
    return False, None
