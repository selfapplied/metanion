import math
import numpy as np
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import time
import re
import importlib.util
import inspect
from bitstring import ConstBitStream
import msgpack
from pathlib import Path
import ast

from aspire import Opix, stable_str_hash
from genome import Genome, Grammar, SeedConfig
import quaternion
from resonance import BlockEvent, get_rotation_from_block
from deflate import stream_events
from collections import namedtuple
from tokenizer import Tokenizer
from branch import BranchCutter, BranchState
from color import ColorEngine

# --- Function-table bitmask (inputs) ---
IN_NAME  = 1 << 0
IN_BYTES = 1 << 1
IN_TEXT  = 1 << 2

# --- Simple Logger ---

# Simple console handler
# ha
# --- Branch State (Branch-cuts over grammar space) ---
BranchState = namedtuple(
    'BranchState', ['sheet', 'cuts', 'winding', 'progress'])

# --- Block Processing Results ---
BlockResult = namedtuple('BlockResult', [
                         'final_q', 'final_qs', 'color_trail', 'mean_surprise', 'std_surprise'])

# --- World State ---
World = namedtuple('World', ['name', 'last_at', 'glyph', 'bits', 'types'])

# --- World Record ---
# A receipt for the computation on an asset, stored in the grammar.
WorldRecord = namedtuple('WorldRecord', ['name', 'glyph', 'bits', 'stats'])

# --- Operators ---


class RecordManager:
    def __init__(self, genome: Genome, minted_symbols: List[str]):
        self.genome = genome
        self.minted_symbols = minted_symbols
        self.records: Dict[str, WorldRecord] = self.load()
        self.active_name: Optional[str] = str(
            self.genome.grammar.unigram_counts.get('worldActive'))

    def to_camel(self, s: str) -> str:
        return "".join(word.capitalize() for word in re.split('[-_]', s))

    def load(self) -> Dict[str, WorldRecord]:
        records = {}
        for key, value in self.genome.grammar.unigram_counts.items():
            if not isinstance(key, str) or not key.startswith('world') or key == 'worldActive':
                continue
            try:
                record_tuple = ast.literal_eval(str(value))
                if isinstance(record_tuple, tuple) and len(record_tuple) == 4:
                    records[record_tuple[0]] = WorldRecord(*record_tuple)
            except (ValueError, SyntaxError):
                continue
        return records

    def save(self):
        keys_to_delete = [k for k in self.genome.grammar.unigram_counts if isinstance(
            k, str) and k.startswith('world')]
        for k in keys_to_delete:
            del self.genome.grammar.unigram_counts[k]

        if self.active_name:
            self.genome.grammar.unigram_counts['worldActive'] = self.to_camel(
                self.active_name)

        for name, record in self.records.items():
            key = f"world{self.to_camel(name)}"
            self.genome.grammar.unigram_counts[key] = repr(record)

    def sanitize(self, s: str, max_len: int = 32) -> str:
        s2 = ''.join(ch if (ch.isalnum() or ch in ('-', '_'))
                     else '-' for ch in s)
        s2 = s2.strip('-_')
        return s2[:max_len] if s2 else 'world'

    def gen_name(self, context: str) -> str:
        ctx_tokens = [t.lower()
                      for t in re.findall(r"[A-Za-z0-9_]+", context or '')]
        ug = self.genome.grammar.unigram_counts

        def score(tok: str) -> float:
            base = float(ug.get(tok, 0))
            overlap = 1.0 if tok.lower() in ctx_tokens else 0.0
            minted_bonus = 2.0 if tok in self.minted_symbols else 0.0
            return base + 3.0 * overlap + minted_bonus

        candidates = sorted(ug.keys(), key=score, reverse=True)[:16]
        primaries = [t for t in self.minted_symbols if t in ug] + \
            [t for t in candidates if t.lower() in ctx_tokens]
        secondaries = [t for t in candidates if t not in primaries]
        parts = []
        if primaries:
            parts.append(primaries[0])
        if secondaries:
            parts.append(secondaries[0])
        if not parts:
            parts = [context or 'world']
        return self.sanitize('-'.join(parts))

    def gen_glyph(self, seed: str) -> str:
        GLYPHS = "αβγδεζηθικλμνξοπρστυφχψω"
        h = hashlib.sha256(seed.encode('utf-8')).digest()
        return GLYPHS[int.from_bytes(h[:2], 'little') % len(GLYPHS)]


# --- Core Engine ---
class CE1Core:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.cfg = genome.config
        self.grammar = genome.grammar
        self.minted_symbols: List[str] = []

        self.levels = self.cfg.levels
        self.symbols = self.cfg.symbols
        self.S = len(self.symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.idx_to_symbol = {i: s for i, s in self.symbol_to_idx.items()}
        
        self.alpha = self.cfg.seed.alpha
        self.kappa = self.cfg.seed.kappa
        self.delta_spawn = self.cfg.seed.delta_spawn
        self.rng = np.random.RandomState(self.cfg.stochastic_seed)
        
        self.mu_rho = np.zeros(self.levels, dtype=float)
        self.surprise_potential = np.zeros(self.levels, dtype=float)
        
        self.records = RecordManager(self.genome, self.minted_symbols)
        self.branch_op = BranchCutter(self.levels, self.cfg.stochastic_seed)
        self.color_op = ColorEngine(self.levels)
        self.tokenizer = Tokenizer(self.genome.grammar)

        self.WORLD_BITS = {'block:stored': 1 << 0, 'block:static': 1 << 1, 'block:dynamic': 1 << 2,
                           'ev:literal': 1 << 3, 'ev:match': 1 << 4, 'ev:phase': 1 << 5, 'ev:atom': 1 << 6, 'ev:summary': 1 << 7}
        self.completeness_threshold = 6
        self.carry_ce1 = Opix()

    def _rotation_from_inverted_diff(self, external_tree: Dict, internal_counts: Dict) -> Tuple[np.ndarray, float]:
        # This now uses the branch_op's project3 method
        M = self.branch_op.project3(len(all_syms))
        # This needs access to color_op to push a color
        # Let's adjust the design slightly. enter_branch should also push the color.
        # Let's modify branch.py to accept the color_op
        pass

    def _embed_tag_in_q(self, q: np.ndarray, tag12: int, component: int = 3) -> np.ndarray:
        # This needs access to color_op to push a color
        # Let's adjust the design slightly. enter_branch should also push the color.
        # Let's modify branch.py to accept the color_op
        pass

    def resonate_with_block(self, q: np.ndarray, payload: Dict, bits_touched: int) -> Tuple[np.ndarray, int]:
        # ... (uses self.branch_op.enter) ...
        pass

    def run_and_learn_asset(self, asset_name: str, time_budget_secs: Optional[float] = None) -> Dict:
        """
        Processes a single, compressed asset from the genome, inhabiting its fractal structure.
        """
        start_time = time.time()
        
        self.records.active_name = self.records.gen_name(
            Path(asset_name).stem.split('.')[0])
        self.records.save()

        # The main state quaternion `q` now represents the VM's current "understanding"
        # or its active "Huffman tree" for the current context.
        q = np.array([1.0, 0.0, 0.0, 0.0]) 
        final_qs: List[np.ndarray] = [q.copy()]
        symbol = 'drift' # The abstract state machine
        all_deltas = []
        angle = 0.0
        
        # Consume unified event stream from the new deflate processor
        last_glyphs: Optional[str] = None
        last_counts = None
        surp = Opix()
        minted_in_asset = 0
        events_in_asset = 0
        bits_touched = 0
        for ev_type, payload in stream_events(asset_name, self.genome, self.tokenizer.ingest):
            # --- Time check per event ---
            if time_budget_secs and (time.time() - start_time) > time_budget_secs:
                logger.warning("Time budget exceeded. Halting.")
                break
            events_in_asset += 1
            if ev_type == 'block' and isinstance(payload, dict):
                q, bits_touched = self.resonate_with_block(
                    q, payload, bits_touched)
            # This part handles all non-block events
            else:
                bit_to_set = self.WORLD_BITS.get(f"ev:{ev_type}")
                if bit_to_set:
                    bits_touched |= bit_to_set

            # This logic was specific to the old mutable world and is no longer needed here
            # It's now handled when the WorldRecord is created at the end.
            if ev_type == 'atom' and isinstance(payload, dict):
                atom_label = str(payload.get(
                    'label', f"atom_{int(payload.get('id', 0)):x}"))[:48]
                self._ensure_symbol_exists(atom_label)
                self.grammar.unigram_counts[atom_label] += 1
                self.grammar.semantic_alleles[atom_label] = q.copy()
                self.minted_symbols.append(atom_label)
                self.last_minted_symbol = atom_label
                self.carry_ce1['∑'] += 1
                self._push_color(0, 0.1)
                if angle > 0:
                    all_deltas.append(angle)
                minted_in_asset += 1
                surp['Ω'] += 1
            elif ev_type == 'literal':
                self.wm.set_bit('ev:literal')
                self.wm.bump_type('ev:literal')
                surp['s'] += 1
            elif ev_type == 'match':
                self.wm.set_bit('ev:match')
                self.wm.bump_type('ev:match')
                surp['Δ'] += 1
            elif ev_type == 'phase':
                self.wm.set_bit('ev:phase')
                self.wm.bump_type('ev:phase')
                surp['S'] += 1
            elif ev_type == 'summary' and isinstance(payload, dict):
                self.wm.set_bit('ev:summary')
                self.wm.bump_type('ev:summary')
                glyph_counts = payload.get('glyphs')
                if glyph_counts:
                    last_counts = glyph_counts
                    # carry cycles from block processor
                    try:
                        cyc = int(glyph_counts.get('∞', 0))
                        if cyc:
                            self.carry_ce1['∞'] += cyc
                    except Exception:
                        pass
                    # Combine error glyphs with surprise bands
                    try:
                        err = Opix(glyph_counts)
                        combined = err.overlay_with(surp)
                        last_glyphs = str(combined)
                    except Exception:
                        last_glyphs = ' '.join(f"{k}:{v}" for k, v in sorted(
                            glyph_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))))

        # Single-line per-asset summary (from last summary event if any)
        glyphs = last_glyphs or '∅'
        # Modernized stats
        hsv = (0.0, 0.0, 0.0)
        # Update world aggregate stats
        if surp:
            surprise_events = surp.get('Ω', [])
            self.wm.update_stats(events_in_asset, surprise_events if isinstance(
                surprise_events, list) else [surprise_events])

        stats = {'mean_surprise': 0.0, 'std_surprise': 0.0,
                 'atoms': minted_in_asset, 'glyphs': glyphs, 'glyph_counts': (last_counts or {}), 'color': hsv}
        # inject CE1 carry into result for potential rendering
        stats['carry_ce1'] = dict(self.carry_ce1)

        # --- Crystallize Results into a WorldRecord ---
        final_stats = {
            'mean_surprise': np.mean(surp['Ω']) if surp['Ω'] else 0.0,
            'std_surprise': np.std(surp['Ω']) if surp['Ω'] else 0.0,
            'atoms': minted_in_asset,
            'events': events_in_asset,
            'glyphs': glyphs
        }

        coverage = bin(bits_touched).count('1')
        new_glyph = self.records.gen_glyph(
            f"{self.records.active_name}:{bits_touched}") if coverage >= self.completeness_threshold else ''

        new_record = WorldRecord(
            name=self.records.active_name,
            glyph=new_glyph,
            bits=bits_touched,
            stats=final_stats
        )
        self.records.records[self.records.active_name] = new_record
        self.records.save()

        return {'final_q': q, 'final_qs': final_qs, 'color_trail': self.color_op.get_trail(), 'stats': stats}

    def _fuse_tokens(self, token_a: str, token_b: str):
        """Merges two tokens into a new, higher-level token (a 'molecule')."""
        new_token = f"{token_a}{token_b}"
        
        # If this molecule already exists, just reinforce its count
        if new_token in self.grammar.unigram_counts:
            self.grammar.unigram_counts[new_token] += 1
            logger.debug(f"Reinforced existing molecule: '{new_token}'")
            return

        logger.info(f"Fusion event: '{token_a}' + '{token_b}' -> '{new_token}'")
        
        # Create a new semantic allele by combining the parents
        allele_a = self.grammar.semantic_alleles.get(token_a, np.array([1.0,0.0,0.0,0.0]))
        allele_b = self.grammar.semantic_alleles.get(token_b, np.array([1.0,0.0,0.0,0.0]))
        new_allele = quaternion.quat_mul(allele_a, allele_b)
        new_allele = quaternion.quat_norm(new_allele)
        
        logger.debug(f"Fused Alleles | a: {np.round(allele_a, 2)}, b: {np.round(allele_b, 2)} -> New: {np.round(new_allele, 2)}")

        # Add the new molecule to the grammar
        self.grammar.unigram_counts[new_token] = 1
        self.grammar.semantic_alleles[new_token] = new_allele
        
        # This is a simplification; a full implementation would also update bigrams
    
    # --- Generation ---
    def generate_text(self, seed_token: str, transformed_state: Dict):
        logger.info(f"Rendering transformed state for seed: '{seed_token}'...")

        if not self.genome.grammar.unigram_counts:
            return "Σ('error', 'grammar is empty')"

        # Find a starting token from the seed text
        seed_tokens = [t for t in seed_token.split() if t in self.genome.grammar.unigram_counts]
        current_token = seed_tokens[0] if seed_tokens else list(self.genome.grammar.unigram_counts.keys())[0]
        
        output_sequence = [current_token]

        # --- Choreographed Walk ---
        # Use the deepest non-empty color trail across levels.
        color_trail = transformed_state.get('color_trail', [])
        final_level_trail = []
        if isinstance(color_trail, list) and color_trail:
            for lvl in range(len(color_trail) - 1, -1, -1):
                if color_trail[lvl]:
                    final_level_trail = color_trail[lvl]
                    break

        if not final_level_trail:
            logger.warning("No color trail found, using final quaternion for a simple walk.")
            q_final = transformed_state['final_qs'][-1]
            # Create a synthetic 10-step trail using the final quaternion's components
            s = np.clip(np.abs(q_final[2]), 0, 1) # y for saturation/novelty
            v = np.clip(np.abs(q_final[0]), 0, 1) # w for value/temperature
            walk_steps = [{'s': s, 'v': v}] * 10
        else:
            logger.info(f"Performing choreographed walk with {len(final_level_trail)} steps...")
            walk_steps = final_level_trail
        
        for step_info in walk_steps:
            if current_token not in self.genome.grammar.bigram_counts:
                break # Dead end

            next_token_counts = self.genome.grammar.bigram_counts[current_token]
            tokens = list(next_token_counts.keys())
            
            # Use HSV from the color trail to guide the walk's biases
            s = step_info.get('s', 0.5) # Saturation
            v = step_info.get('v', 0.5) # Value (Brightness)

            temperature = max(0.1, v * 2.0) # Brighter = more random
            novel_bias = s                  # More saturated = more novel
            common_bias = 1.0 - s           # Less saturated = more common

            # --- Probability Calculation ---
            base_counts = np.array([next_token_counts[t] for t in tokens], dtype=np.float32)
            base_probs = base_counts / np.sum(base_counts)
            
            unigram_total = sum(self.genome.grammar.unigram_counts.values())
            novelty_scores = np.array([1.0 - (self.genome.grammar.unigram_counts.get(t, 0) / unigram_total) for t in tokens], dtype=np.float32)
            
            combined_scores = (common_bias * np.log(base_probs + 1e-9)) + (novel_bias * np.log(novelty_scores + 1e-9))

            # Common-prefix boost → treat as saturation-driven attraction (green bias guidance)
            def _cpl(a: str, b: str) -> int:
                m = min(len(a), len(b))
                i = 0
                while i < m and a[i] == b[i]:
                    i += 1
                return i
            prefix_raw = np.array([_cpl(current_token, t)
                                  for t in tokens], dtype=np.float32)
            if prefix_raw.size and float(prefix_raw.max()) > 0.0:
                prefix_scores = prefix_raw / float(prefix_raw.max())
                combined_scores = combined_scores + (s * prefix_scores)
            
            probabilities = np.exp(combined_scores / temperature)
            probabilities /= np.sum(probabilities)

            try:
                next_token = self.rng.choice(tokens, p=probabilities)
                output_sequence.append(next_token)
                current_token = next_token
            except (ValueError, ZeroDivisionError):
                break
        
        return " ".join(output_sequence)
