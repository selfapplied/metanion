import numpy as np
import random
from typing import List, Dict, Optional, Any

def generate_choreographed_text(engine, seed_text: str, result: Dict, length: int = 100) -> str:
    """
    Generates text using a choreographed walk based on the engine's state.
    Ported from the original CE1Core engine for higher quality output.
    """
    if not engine.unigram_counts:
        return "Error: Grammar is empty."

    seed_tokens = [t for t in seed_text.split() if t in engine.unigram_counts]
    if not seed_tokens:
        # Fallback to a random token if seed words are not in the grammar
        if not engine.unigram_counts: return "Error: Grammar has no words."
        current_token = engine.rng.choice(list(engine.unigram_counts.keys()))
    else:
        current_token = seed_tokens[0]
    
    output_sequence = [current_token]

    # Use the color trail from the result for the choreographed walk
    color_trail = result.color_trail if hasattr(
        result, 'color_trail') else result.get('color_trail', [])
    walk_steps = []
    if color_trail:
        # Use the deepest non-empty color trail
        for level_trail in reversed(color_trail):
            if level_trail:
                walk_steps = level_trail
                break

    if not walk_steps:
        # Fallback to a simple walk using the final quaternion
        final_qs = result.final_q if hasattr(
            result, 'final_q') else result.get('final_q')
        if final_qs:
            q_final = final_qs[-1]
            s = np.clip(np.abs(q_final[2]), 0, 1)  # Saturation/novelty
            v = np.clip(np.abs(q_final[0]), 0, 1)  # Value/temperature
            walk_steps = [{'s': s, 'v': v}] * length
        else: # If all else fails, do a neutral walk
            walk_steps = [{'s': 0.5, 'v': 0.5}] * length

    for step_info in walk_steps:
        if len(output_sequence) >= length:
            break
        if current_token not in engine.bigram_counts:
            break  # Dead end

        next_token_counts = engine.bigram_counts[current_token]
        tokens = list(next_token_counts.keys())
        
        if isinstance(step_info, dict):
            s = step_info.get('s', 0.5)
            v = step_info.get('v', 0.5)
        else:
            s = step_info.s
            v = step_info.v

        temperature = max(0.1, v * 2.0)
        novel_bias = s
        common_bias = 1.0 - s

        base_counts = np.array([next_token_counts[t] for t in tokens], dtype=np.float32)
        base_probs = base_counts / np.sum(base_counts)
        
        unigram_total = sum(engine.unigram_counts.values())
        novelty_scores = np.array([1.0 - (engine.unigram_counts.get(t, 0) / unigram_total) for t in tokens], dtype=np.float32)
        
        combined_scores = (common_bias * np.log(base_probs + 1e-9)) + (novel_bias * np.log(novelty_scores + 1e-9))
        
        probabilities = np.exp(combined_scores / temperature)
        probabilities /= np.sum(probabilities)

        try:
            next_token = engine.rng.choice(tokens, p=probabilities)
            output_sequence.append(next_token)
            current_token = next_token
        except (ValueError, ZeroDivisionError):
            break
    
    return " ".join(output_sequence)


def generate_text(engine, seed_text: str, banks: Optional[List[List[np.ndarray]]] = None, finals: Optional[List[np.ndarray]] = None,
                  result: Optional[Dict] = None, style: str = "choreographed", length: int = 100, **kwargs) -> str:
    """
    Unified text generation. Defaults to the high-quality choreographed style.
    """
    if result is None:
        # Reconstruct a dummy result if not provided
        result = {'banks': banks, 'final_q': finals, 'color_trail': []}

    if style == "choreographed":
        return generate_choreographed_text(engine, seed_text, result, length)
    elif style == "pheno":
        return engine.generate_pheno_text(seed_text, result.banks if hasattr(result, 'banks') else result.get('banks'),
                                          result.final_q if hasattr(
                                              result, 'final_q') else result.get('final_q'),
                                          sentences=max(1, length // 30))
    else:
        # Keep old styles as fallbacks if needed, but choreographed is the new default.
        return f"Unknown or unsupported generation style: {style}"
