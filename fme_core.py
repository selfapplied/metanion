"""
CE1{listener=Forest⊙Ear⊙Memory⊙Map; Ξ=seed:stream{text};D=dim;
q_k=[w,x,y,z]∈S^3;μρ_k←(1-α)μρ_k+αρ;Δ=ρ-μρ_k; χ=sign(Δ);m=Heaviside(|Δ|-|x|);
e_vec=normalize(χ·m·[x,y,z]+(1-m)·[0,0,1]); r̂=rot(ε₀·|Δ|,axis=e_vec);q_k←turn(q_k⊗r̂);
spill=‖q_k‖²-1;q_k←gravity(q_k+(z/D)·spill·q_k); ledger←ledger+echo(spill);
ρ̃=(w_PK·PK+w_K·Krav+w_FR·FR)(ρ);Σ(w_PK,w_K,w_FR)=1; bead=mark(Δ-chain,D);
Ψ=(w_PK·PK+w_K·Krav+w_FR·FR)·q_k;T(σ→σ′)=exp(Ψ·Δ); σ′~ear(T/|x|);
rng=Ξ mod 2ᴰ;mode=1-2·(rng mod 2); invariant={‖q_k‖=1;bead.gcd=1;ΔE≤0;parity(ledger)=parity(D)}}
"""

import os
import math
import random
import numpy as np
from collections import Counter, namedtuple
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm.auto import tqdm
import hashlib

from fme_quaternion import quat_mul, quat_norm, axis_angle_quat
import fme_color
import fme_text_generation
import fme_tokenization
import fme_training
import fme_analysis
import fme_engine
from genome import Genome

ContentFeatures = namedtuple(
    'ContentFeatures', 'length lines entropy structure_ratio')


def word_to_quaternion_lens(props: Dict[str, Any]) -> np.ndarray:
    """Generates a quaternion 'lens' from a word's properties."""
    pos_map = {'NOUN': 0.1, 'VERB': 0.2, 'ADJ': 0.3,
               'ADV': 0.4, 'DET': 0.5, 'PUNCT': -0.1, 'SPACE': -0.2}
    shape_map = {'LOWER': 0.01, 'TITLE': 0.02, 'CAMEL': 0.05, 'SNAKE': 0.05}

    # Create a small, unique rotation based on word properties
    i = pos_map.get(props.get('pos', 'NOUN'), 0.0)
    j = shape_map.get(props.get('shape', 'LOWER'), 0.0)
    k = (props.get('length', 0) % 5) * 0.02

    # Return a near-identity quaternion that nudges the state
    return quat_norm(np.array([1.0, i, j, k]))


class FractalMarkovAlleleEngine:
    def __init__(self, template: Dict[str, Any]):
        if not isinstance(template, dict):
            raise TypeError("template must be a config dictionary.")

        cfg = template
        self.template_path = "."  # Default to current dir
        self.genome: Optional[Genome] = None

        self.cfg = cfg
        self.levels = int(cfg.get('levels', {}).get('count', 4))
        seed = cfg.get('stochastic', {}).get('seed', None)
        self.rng = np.random.RandomState(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.noise = float(cfg.get('stochastic', {}).get('noise', 0.0))
        self.symbols = list(cfg.get('alphabet', {}).get('symbols', ['carry','borrow','drift']))
        self.S = len(self.symbols)
        self.symbol_to_idx = {s:i for i,s in enumerate(self.symbols)}
        self.idx_to_symbol = {i:s for s,i in self.symbol_to_idx.items()}
        self.priors = self._build_prior_matrix(cfg.get('priors', {}))
        self.scales = self._read_scales(cfg.get('fractal', {}).get('scales', {}))
        self.self_similarity = float(cfg.get('fractal', {}).get('self_similarity', 0.6))
        self.branch_factor = int(cfg.get('fractal', {}).get('branch_factor', 3))
        self.emissions = cfg.get('emissions', {})
        self.alpha = float(cfg.get('seed', {}).get('alpha', 0.1))
        self.epsilon0 = float(cfg.get('seed', {}).get('epsilon0', 1.0))
        self.beta = float(cfg.get('seed', {}).get('beta', 1.0))
        self.kappa = float(cfg.get('seed', {}).get('kappa', 0.25))
        self.delta_spawn = float(cfg.get('seed', {}).get('delta_spawn', 0.05))
        self.basis_weights = np.array(cfg.get('seed', {}).get('basis_weights', [0.34, 0.33, 0.33]), dtype=float)
        self.basis_weights /= max(self.basis_weights.sum(), 1e-9)
        self.axis_dirs = np.zeros((self.S, 3), dtype=float)
        for j, s in enumerate(self.symbols):
            axis = self.emissions.get(s, {}).get('axis', 'z')
            self.axis_dirs[j] = np.array([1.0, 0.0, 0.0]) if axis=='x' else (np.array([0.0, 1.0, 0.0]) if axis=='y' else np.array([0.0, 0.0, 1.0]))
        self.mu_rho = np.zeros(self.levels, dtype=float)
        self.ledger = np.zeros(self.levels, dtype=float)
        in_dir = os.path.dirname(__file__)
        mem_hash = _hash_directory_mem(in_dir)
        self.rng_mod = int(mem_hash[:8], 16) % (2**self.levels)
        self.mode = 1 - 2 * (self.rng_mod % 2)
        self.transitions_per_level = self._build_level_kernels()
        self.starting_alleles: List[Optional[np.ndarray]] = [None]*self.levels
        self.vocab: List[str] = []
        self.planned_steps = 64
        self.bigram_counts: Dict[str, Dict[str,int]] = {}
        self.unigram_counts: Dict[str,int] = {}
        self.token_properties: Dict[str, Dict] = {}
        self.shape_distribution: Dict[str, float] = {}

        # Color phase allele system for geometric transformations
        self.color_phase_alleles: List[Dict[str, float]] = []
        self.color_cluster_centers: Optional[np.ndarray] = None
        self.color_phase_mapping: Dict[str, int] = {}
        # Optional sampling knobs set by CLI for speed; None means full content
        self.assess_sample_chars: Optional[int] = None
        self.train_sample_chars: Optional[int] = None

        # Kernels (one per level)
        self.kernels = self._build_level_kernels()

    def _progress_iter(self, items, desc: str):
        return tqdm(items, desc=desc, mininterval=0.1, leave=True)

    def _read_scales(self, table):
        return [float((table or {}).get(f"l{k}", 1.0)) for k in range(self.levels)]

    @staticmethod
    def _prompt_features(prompt: str) -> Dict[str,float]:
        n = len(prompt)
        if n == 0:
            return {'length':0.0,'entropy':0.0}
        c = Counter(prompt)
        ent = -sum((cnt/n)*math.log2(cnt/n) for cnt in c.values())
        return {'length': float(n), 'entropy': float(ent)}

    def _prompt_to_q_temp(self, prompt: str):
        f = self._prompt_features(prompt)
        ln = min(f['length']/256.0, 1.0)
        q = quat_norm(np.array([1.0, 1.0 - ln, 0.1, ln]))
        temp = 1.0 - 0.5 * ln
        return q, float(temp)

    def _build_prior_matrix(self, priors_table):
        M = np.zeros((self.S, self.S), dtype=float)
        for i, s in enumerate(self.symbols):
            row = priors_table.get(s, {})
            for j, t in enumerate(self.symbols):
                M[i, j] = float(row.get(t, 1.0/self.S))
            ssum = M[i].sum()
            M[i] = M[i]/ssum if ssum else np.full(self.S, 1.0/self.S)
        return M

    def _build_prior_from_dict(self, priors_table):
        M = np.zeros((self.S, self.S), dtype=float)
        for i, s in enumerate(self.symbols):
            row = priors_table.get(s, {})
            for j, t in enumerate(self.symbols):
                M[i, j] = float(row.get(t, 1.0/self.S))
            ssum = M[i].sum()
            M[i] = M[i]/ssum if ssum else np.full(self.S, 1.0/self.S)
        return M

    def _build_level_kernels(self):
        kernels = []
        base = self.priors
        for k in range(self.levels):
            M = self.self_similarity * base + (1.0 - self.self_similarity) * (base @ base)
            # Use deterministic randomness for reproducible noise
            rng = np.random.RandomState(self.rng_mod + k)  # Vary by level for diversity
            M = (1.0 - self.noise) * M + self.noise * rng.rand(self.S, self.S)
            M = np.clip(M * self.scales[k], 1e-9, None)
            M /= M.sum(axis=1, keepdims=True)
            kernels.append(M)
            base = M
        return kernels

    def step(self, level: int, q: np.ndarray, state_symbol: str):
        return fme_engine.step(self, level, q, state_symbol)

    def run(self, steps=64):
        return fme_engine.run(self, steps)

    def select_starting_alleles(self, seed_text: str, banks: List[List[np.ndarray]]):
        h = hashlib.sha256(seed_text.encode('utf-8', errors='ignore')).hexdigest()
        selections: List[Optional[np.ndarray]] = []
        for k in range(self.levels):
            alleles = banks[k]
            if not alleles:
                selections.append(None); continue
            idx = int(h[(k*4):(k*4+4)], 16) % len(alleles)
            self.starting_alleles[k] = alleles[idx]; selections.append(alleles[idx])
        return selections

    def select_starting_alleles_from_prompt(self, prompt: str, banks: Optional[List[List[np.ndarray]]]) -> List[Optional[np.ndarray]]:
        if banks is None:
            return [None] * self.levels
        q_p, _ = self._prompt_to_q_temp(prompt)
        e_dir = q_p[1:]; n = float(np.linalg.norm(e_dir)); e_dir = e_dir/(n if n else 1.0)
        chosen: List[Optional[np.ndarray]] = []
        for k in range(self.levels):
            alleles = banks[k]
            if not alleles:
                chosen.append(None); continue
            sims = [float(np.dot(e_dir, a[1:])) for a in alleles]
            idx = int(np.argmax(sims))
            self.starting_alleles[k] = alleles[idx]
            chosen.append(alleles[idx])
        return chosen

    @staticmethod
    def _format_q(q: np.ndarray) -> str:
        return f"[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}]"

    def _format_allele_as_glyph(self, q: np.ndarray) -> str:
        """Convert quaternion to legible token/glyph representation"""
        try:
            if q is None or len(q) < 4:
                return "∅"

            w, x, y, z = q

            # Use quaternion components to select meaningful symbols
            # Map to ranges that make sense for glyph selection
            angle = math.atan2(y, x)  # Angular component
            magnitude = math.sqrt(x*x + y*y)  # Vector magnitude in xy plane
            elevation = z  # Z component as elevation

            # Select glyph based on quaternion properties
            glyphs = {
                (0, 0): "⊙",    # Origin
                (1, 0): "⟐",    # Pure real
                (0, 1): "⟑",    # Pure imaginary
                (1, 1): "⟒",    # Full complex
            }

            # Classify based on dominant components
            real_dom = abs(w) > abs(x) and abs(w) > abs(y) and abs(w) > abs(z)
            imag_xy_dom = abs(x) > abs(w) and abs(y) > abs(
                w) and abs(x) > abs(z) and abs(y) > abs(z)
            imag_z_dom = abs(z) > abs(w) and abs(
                z) > abs(x) and abs(z) > abs(y)

            if real_dom:
                glyph = "⟐"
            elif imag_xy_dom:
                glyph = "⟑"
            elif imag_z_dom:
                glyph = "⟒"
            else:
                glyph = "⊙"

            # Add magnitude indicator
            norm = float(np.linalg.norm(q))
            if norm > 1.1:
                glyph += "⬤"  # Large
            elif norm < 0.9:
                glyph += "⬡"  # Small

            return glyph
        except (TypeError, IndexError, AttributeError):
            return "?"

    def _tok(self, text: str) -> List[str]:
        """Adaptive tokenization - learn tokens from data patterns"""
        return fme_tokenization.tok(self, text)

    def _learn_tokenization_excavation(self, training_texts: List[str], iteration: int,
                                      assessment_context: Optional[Dict[str, Any]] = None) -> None:
        """Unified tokenization learning with assessment-guided strategy"""
        return fme_tokenization.learn_tokenization_excavation(self, training_texts, iteration, assessment_context)

    def _learn_tokenization(self, training_texts: List[str]):
        """Placeholder for a more advanced tokenization learning algorithm."""
        pass

    def _refine_alphabet_for_iteration(self, word_counts, iteration):
        """Refine alphabet based on learned patterns from this iteration"""
        if not word_counts:
            return

        # Analyze current symbol usage
        total_symbols = len(self.symbols)
        symbol_usage = {s: 0 for s in self.symbols}

        # Count how often each symbol is used in frequent words
        frequent_words = [w for w, c in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]]
        for word in frequent_words:
            for symbol in self.symbols:
                if symbol in word.lower():
                    symbol_usage[symbol] += word_counts[word]

        # Find most and least used symbols
        sorted_usage = sorted(symbol_usage.items(), key=lambda x: x[1], reverse=True)
        most_used = [s for s, c in sorted_usage if c > sum(word_counts.values()) * 0.001]  # > 0.1% usage
        least_used = [s for s, c in sorted_usage if c < sum(word_counts.values()) * 0.0001]  # < 0.01% usage

        # Expand alphabet for complex patterns
        if iteration > 0 and len(most_used) > len(self.symbols) * 0.8:
            # Add new symbols for emerging patterns
            new_symbols = []
            if 'complex' not in self.symbols:
                new_symbols.append('complex')
            if 'pattern' not in self.symbols:
                new_symbols.append('pattern')
            if 'structure' not in self.symbols:
                new_symbols.append('structure')

            if new_symbols:
                self.symbols.extend(new_symbols)
                print(f"🔤 Iteration {iteration + 1}: Expanded alphabet with {new_symbols}")
                # Rebuild transition matrices with new symbols
                self.transitions_per_level = self._build_level_kernels()

        # Analyze token patterns to suggest alphabet modifications
        token_analysis = self._analyze_token_patterns_for_alphabet(frequent_words)

        if token_analysis['needs_expansion']:
            expansion_symbols = token_analysis['suggested_symbols']
            if expansion_symbols:
                existing_new = [s for s in expansion_symbols if s not in self.symbols]
                if existing_new:
                    self.symbols.extend(existing_new[:3])  # Add up to 3 new symbols
                    print(f"🔤 Iteration {iteration + 1}: Added symbols {existing_new[:3]} based on token analysis")
                    self.transitions_per_level = self._build_level_kernels()

    def _analyze_token_patterns_for_alphabet(self, tokens):
        """Analyze token patterns to suggest alphabet modifications"""
        patterns = {
            'numeric': 0,
            'alphabetic': 0,
            'structural': 0,
            'mixed': 0
        }

        for token in tokens:
            if any(c.isdigit() for c in token):
                patterns['numeric'] += 1
            if any(c.isalpha() for c in token):
                patterns['alphabetic'] += 1
            if any(c in '._-+/\\' for c in token):
                patterns['structural'] += 1
            if len(token) > 10:
                patterns['mixed'] += 1

        # Suggest symbols based on patterns
        suggestions = []
        if patterns['numeric'] > len(tokens) * 0.3:
            suggestions.extend(['numeric', 'digits'])
        if patterns['structural'] > len(tokens) * 0.2:
            suggestions.extend(['structure', 'boundary'])
        if patterns['mixed'] > len(tokens) * 0.1:
            suggestions.extend(['complex', 'compound'])

        return {
            'needs_expansion': len(suggestions) > 0,
            'suggested_symbols': suggestions,
            'pattern_analysis': patterns
        }

    def _ensure_bigrams_for(self, head: str, directory: Optional[str] = None) -> None:
        if head in self.bigram_counts and self.bigram_counts[head]:
            return
        directory = directory or os.path.dirname(__file__)
        counts: Dict[str,int] = {}
        try:
            for name in sorted(os.listdir(directory)):
                if name.startswith('.'):
                    continue
                path = os.path.join(directory, name)
                if not os.path.isfile(path):
                    continue
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                toks = self._tok(text)
                for a, b in zip(toks[:-1], toks[1:]):
                    if a == head:
                        counts[b] = counts.get(b, 0) + 1
        except Exception:
            pass
        if counts:
            self.bigram_counts[head] = counts
            # also update unigrams with discovered tails
            for t, c in counts.items():
                self.unigram_counts[t] = self.unigram_counts.get(t, 0) + c

    def generate_markov_text(self, prompt: str, length: int = 50) -> str:
        q_p, temp = self._prompt_to_q_temp(prompt); T = max(0.1, float(temp))
        toks = self._tok(prompt)
        # pick a starting token
        if toks:
            cur = toks[-1]
        elif self.unigram_counts:
            cur = max(self.unigram_counts, key=lambda k: self.unigram_counts.get(k, 0))
        else:
            # quick seed from files if unigrams empty
            self._ensure_bigrams_for('the')
            cur = 'the'
        out = [cur]
        in_dir = os.path.dirname(__file__)
        for _ in range(length-1):
            if cur not in self.bigram_counts or not self.bigram_counts[cur]:
                self._ensure_bigrams_for(cur, in_dir)
            nxts = self.bigram_counts.get(cur)
            if not nxts:
                # backoff to a frequent unigram
                cur = (toks[-1] if toks else (max(self.unigram_counts, key=lambda k: self.unigram_counts.get(k, 0)) if self.unigram_counts else 'ce1'))
                out.append(cur)
                continue
            keys = list(nxts.keys()); vals = np.array([nxts[k] for k in keys], dtype=float)
            vals /= vals.sum()
            logits = np.log(vals + 1e-12) / T
            logits -= logits.max(); probs = np.exp(logits); probs/=probs.sum()
            cur = str(np.random.choice(keys, p=probs))
            out.append(cur)
        s = ' '.join(out)
        return s[: max(40, length*6)] + ('' if s.endswith('.') else '.')

    def emit_text_from_allele(self, q: np.ndarray, length: int = 120) -> str:
        if q is None: return ""
        w, x, y, z = q
        s = f"q={self._format_q(q)} ; Ψ={float(np.dot(self.basis_weights, q[:3])):.3f} ; |x|={abs(x):.3f} ; |Δ|≈{self.delta_spawn:.3f}"
        if len(s) < length: s += " ; q⊗rot(Ψ,z) → CE1"
        return s[:length]

    def emit_ce1_color_info(self, delta: float) -> str:
        """Emit CE1 color information using floating point structure"""
        hue, saturation, brightness = fme_color.delta_to_color(delta)
        sign, exponent, mantissa = self._extract_float_bits(delta)

        return f"CE1{{Δ={delta:.6f}; sign={sign}; exp={exponent}; mantissa={mantissa:.6f}; color=hsb({hue:.1f}°, {saturation:.3f}, {brightness:.3f})}}"

    def allele_to_sentence(self, q_allele: np.ndarray) -> str:
        if q_allele is None or not self.unigram_counts:
            return ""

        length = int(20 + 40 * min(abs(q_allele[0]), 1.0))

        # Start at the origin, but nudge the state with the allele's essence
        q_state = quat_norm(
            np.array([1.0, 0.0, 0.0, 0.0]) + 0.1 * q_allele)

        start_token_options = self.vocab or list(self.unigram_counts.keys())
        if not start_token_options:
            return ""
        current_token = random.choice(start_token_options)

        output_sequence = [current_token]

        for _ in range(length - 1):
            self._ensure_bigrams_for(current_token)
            next_token_counts = self.bigram_counts.get(current_token)

            if not next_token_counts:
                current_token = random.choice(start_token_options)
                output_sequence.append(current_token)
                continue

            tokens = list(next_token_counts.keys())
            counts = np.array(
                [next_token_counts[t] for t in tokens], dtype=np.float32)

            # Use the current state to bias the selection
            for i, token in enumerate(tokens):
                props = self.token_properties.get(token)
                if props:
                    # How well does this token align with the current state?
                    lens = word_to_quaternion_lens(props)
                    # Dot product measures alignment of rotations
                    alignment = np.dot(q_state[1:], lens[1:])
                    counts[i] *= (1.0 + 0.5 * alignment)

            counts = np.maximum(counts, 1e-9)
            probabilities = counts / np.sum(counts)

            try:
                next_token = str(np.random.choice(tokens, p=probabilities))
                output_sequence.append(next_token)

                # Update state with the new word's lens
                props = self.token_properties.get(next_token)
                if props:
                    lens = word_to_quaternion_lens(props)
                    q_state = quat_norm(quat_mul(q_state, lens))

                current_token = next_token

            except (ValueError, ZeroDivisionError):
                current_token = random.choice(start_token_options)
                output_sequence.append(current_token)

        return "".join(output_sequence)

    def generate_pheno_text(self, seed_text: str, banks: Optional[List[List[np.ndarray]]], finals: Optional[List[np.ndarray]], sentences: int = 3) -> str:
        chosen = self.select_starting_alleles_from_prompt(seed_text, banks)
        qs = [q for q in chosen if q is not None] or (
            [finals[-1]] if finals else [])
        if not qs: return ""
        sents = [self.allele_to_sentence(q) for q in qs[:sentences]]
        if len(sents) < sentences: sents.extend(self.allele_to_sentence(qs[-1]) for _ in range(sentences - len(sents)))
        return ' '.join(sents[:sentences])

    def percept(self, rho) -> np.ndarray:
        """Perceptual processing hook for PK/Krav/FR modes"""
        rho_array = np.array(rho) if hasattr(rho, '__iter__') else np.array([rho])

        # PK: Pattern Recognition - detect structural patterns
        pk_features = self._extract_pattern_features(rho_array)

        # Krav: Creative Variation - add stochastic variation
        krav_features = self._add_creative_variation(pk_features)

        # FR: Fractal Resonance - apply fractal scaling
        fr_features = self._apply_fractal_resonance(krav_features)

        return fr_features

    def _extract_pattern_features(self, rho_array):
        """PK: Extract pattern features from input"""
        if len(rho_array) < 3:
            return rho_array

        # Detect edges, symmetries, repetitions
        features = []
        for i in range(len(rho_array)):
            # Local context window
            window = rho_array[max(0, i-2):min(len(rho_array), i+3)]
            if len(window) >= 3:
                # Edge detection: difference from neighbors
                edge = abs(window[len(window)//2] - np.mean(window))
                features.append(edge)

        return np.array(features) if features else rho_array

    def _add_creative_variation(self, features):
        """Krav: Add creative stochastic variation"""
        if len(features) == 0:
            return features

        # Add controlled randomness based on feature magnitude
        # Use deterministic random state for reproducibility
        rng = np.random.RandomState(self.rng_mod)
        variation = rng.normal(0, 0.1 * np.std(features), len(features))
        return features + variation

    def _apply_fractal_resonance(self, features):
        """FR: Apply fractal scaling and resonance"""
        if len(features) == 0:
            return features

        # Apply fractal scaling across levels
        scaled_features = []
        for level in range(min(self.levels, len(features))):
            scale_factor = self.scales[level]
            scaled = features[:level+1] * scale_factor
            scaled_features.extend(scaled)

        return np.array(scaled_features)

    def interpret_pheno_and_apply(self, pheno_text: str) -> Dict[str, float]:
        changes: Dict[str,float] = {}
        for tok in pheno_text.replace(';',' ').split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                try:
                    val = float(v)
                except Exception:
                    continue
                if hasattr(self, k):
                    setattr(self, k, val); changes[k] = val
        self.transitions_per_level = self._build_level_kernels()
        return changes

    @staticmethod
    def _content_features(content: str) -> ContentFeatures:
        length = len(content)
        lines = content.count('\n')
        if length == 0:
            return ContentFeatures(length=0, lines=0, entropy=0.0, structure_ratio=0.0)
        c = Counter(content)
        entropy = -sum((cnt/length)*math.log2(cnt/length)
                       for cnt in c.values())
        return ContentFeatures(length=float(length), lines=float(lines), entropy=float(entropy), structure_ratio=float(lines/max(length, 1)))

    def _symbol_from_content(self, content: str) -> str:
        f = self._content_features(content)
        return 'carry' if f.entropy > 4.0 else ('borrow' if f.structure_ratio > 0.1 else 'drift')

    @staticmethod
    def _amp_from_features(feat: ContentFeatures) -> float:
        return 0.2 + 1.2 * (0.5 * (min(feat.length/2000.0, 1.0) + min(feat.entropy/6.0, 1.0)))

    def _learn_transition_matrix(self, seq):
        C = np.full((self.S, self.S), 1e-6, dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            ia = self.symbol_to_idx.get(a)
            ib = self.symbol_to_idx.get(b)
            if ia is None or ib is None:
                continue
            C[ia, ib] += 1.0
        row_sums = C.sum(axis=1, keepdims=True)
        P = np.divide(C, row_sums, out=np.full_like(C, 1.0/self.S), where=row_sums>0)
        return P

    def _token_from_q(self, q: np.ndarray, symbol: str) -> str:
        """Generates a token from a quaternion and a symbol."""
        if self.vocab:
            # Simple heuristic: use quaternion components to select a word from the vocab
            idx = int(abs(q[1] * q[2]) * len(self.vocab)) % len(self.vocab)
            return self.vocab[idx]
        return symbol  # Fallback to the symbol if vocab is empty

    def _sample_symbol_sequence(self, level: int, steps: int, start_symbol: str, prompt: str = "") -> List[str]:
        """Samples a sequence of symbols from the transition matrix at a given level."""
        if start_symbol not in self.symbol_to_idx:
            start_symbol = 'drift'  # Fallback

        seq = [start_symbol]
        current_symbol = start_symbol

        for _ in range(steps - 1):
            idx = self.symbol_to_idx[current_symbol]
            probs = self.transitions_per_level[level][idx]
            next_symbol = np.random.choice(self.symbols, p=probs)
            seq.append(next_symbol)
            current_symbol = next_symbol

        return seq

    def train_from_directory(self, directory='.', blend=0.5, include_hidden=False, incremental_files=None, excavation_iterations=3, enable_jit_excavation=True):
        return fme_training.train_from_directory(self, directory, blend, include_hidden, incremental_files, excavation_iterations, enable_jit_excavation)

    def train_from_assets(self, assets: Dict[str, bytes], blend=0.5, excavation_iterations=3, enable_jit_excavation=True):
        return fme_training.train_from_assets(self, assets, blend, excavation_iterations, enable_jit_excavation)

    def load_learned_if_cached(self, directory='.') -> tuple[bool, Optional[Dict[str, tuple[str, str]]]]:
        return fme_training.load_learned_if_cached(self, directory)

    def transform_color_between_geometries(self, delta: float, source_geometry: str = 'default', target_geometry: str = 'default') -> Tuple[float, float, float]:
        """
        Transforms a color representation of a delta value between different geometric templates.
        """
        source_allele = self.color_phase_alleles.get(source_geometry)
        target_allele = self.color_phase_alleles.get(target_geometry)

        if source_allele and target_allele:
            # This is a simplified transformation, more sophisticated methods could be developed
            return fme_color.apply_color_phase_allele(delta, target_allele)
        else:
            return fme_color.delta_to_color(delta)

    def generate_text(self, seed_text: str, banks: Optional[List[List[np.ndarray]]] = None, finals: Optional[List[np.ndarray]] = None,
                      result: Optional[Dict] = None, style: str = "pheno", length: int = 100, **kwargs) -> str:
        """
        Unified text generation that combines multiple approaches.
        """
        return fme_text_generation.generate_text(self, seed_text, banks, finals,
                                                 result=result, style=style, length=length, **kwargs)

    def analyze_color_phases_clustering(self, color_samples: List[Tuple[float, float, float, float]], n_clusters: int = 8) -> Tuple[List[Dict[str, float]], np.ndarray]:
        return fme_analysis.analyze_color_phases_clustering(self, color_samples, n_clusters)

    def get_color_phase_statistics(self) -> Dict[str, Any]:
        return fme_analysis.get_color_phase_statistics(self)

    def demonstrate_color_phase_transformation(self, test_deltas: Optional[List[float]] = None) -> str:
        return fme_analysis.demonstrate_color_phase_transformation(self, test_deltas)


def _hash_directory_mem(directory='.', include_hidden=False):
    h = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        if not include_hidden and name.startswith('.'):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        h.update(name.encode('utf-8', errors='ignore'))
        try:
            with open(path, 'rb') as f:
                h.update(f.read())
        except Exception:
            continue
    return h.hexdigest()
