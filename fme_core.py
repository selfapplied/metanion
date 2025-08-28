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
import sys
import time
import tomllib as toml
import hashlib
from collections import Counter
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class FractalMarkovAlleleEngine:
    def __init__(self, template_path: str):
        with open(template_path, 'rb') as f:
            cfg = toml.load(f)
        self.cfg = cfg
        self.template_path = template_path
        self.levels = int(cfg.get('levels', {}).get('count', 4))
        seed = cfg.get('stochastic', {}).get('seed', None)
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

        # Color phase allele system for geometric transformations
        self.color_phase_alleles: List[Dict[str, float]] = []
        self.color_cluster_centers: Optional[np.ndarray] = None
        self.color_phase_mapping: Dict[str, int] = {}
        # Optional sampling knobs set by CLI for speed; None means full content
        self.assess_sample_chars: Optional[int] = None
        self.train_sample_chars: Optional[int] = None

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
        en = min(f['entropy']/6.0, 1.0)
        q = np.array([1.0, en, ln, 0.5*(en+ln)*(1 if (hash(prompt)&1)==0 else -1)], dtype=float)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        temp = 0.7 + 1.3*en
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

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=float)

    @staticmethod
    def _quat_norm(q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        return q / max(n, 1e-15)  # Much smaller epsilon for mantissa precision

    @staticmethod
    def _axis_angle_quat(axis_vec: np.ndarray, angle: float) -> np.ndarray:
        axis = axis_vec.astype(float)
        n = float(np.linalg.norm(axis))
        if n == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis /= n; h = 0.5 * angle; s = math.sin(h)
        return np.array([math.cos(h), s*axis[0], s*axis[1], s*axis[2]], dtype=float)

    def _emit_amp(self, symbol: str) -> float:
        lo, hi = self.emissions.get(symbol, {}).get('amplitude', [0.4, 0.9])
        return random.uniform(float(lo), float(hi))

    def step(self, level: int, q: np.ndarray, state_symbol: str):
        i = self.symbol_to_idx[state_symbol]
        P = self.transitions_per_level[level][i]
        amp = self._emit_amp(state_symbol)
        mu = (1.0 - self.alpha) * self.mu_rho[level] + self.alpha * amp
        self.mu_rho[level] = mu
        Delta = amp - mu
        x, y, z = q[1], q[2], q[3]
        # Strategic threshold: detect significant events clearly
        m = 1.0 if abs(Delta) > abs(x) else 0.0  # Heaviside detector for event significance
        chi = 1.0 if Delta >= 0 else -1.0        # Clear sign detection
        e_vec = chi * m * np.array([x, y, z]) + (1.0 - m) * np.array([0.0, 0.0, 1.0])
        # Let zero vectors be zero - the system should listen to silence too
        r_hat = self._axis_angle_quat(e_vec, self.epsilon0 * abs(Delta) * self.mode)
        q_rot_raw = self._quat_mul(q, r_hat)
        e_norm = float(np.linalg.norm(e_vec))
        e_dir = e_vec / max(e_norm, 1e-9) if e_norm > 1e-9 else np.array([0.0, 0.0, 0.0])
        q_inj = q_rot_raw + self.kappa * abs(Delta) * np.array([0.0, e_dir[0], e_dir[1], e_dir[2]])
        maxc_raw = float(np.max(np.abs(q_inj)))
        # Event-threshold renormalization: only correct when spill exceeds threshold
        q_new = q_rot_raw  # Let the quaternion evolve naturally
        spill = float(np.dot(q_new, q_new) - 1.0)  # Detect renormalization event

        # Honest invariant: only renormalize on significant events
        if abs(spill) > 0.01:  # Event threshold for renormalization
            q_new = self._quat_norm(q_new)
        self.ledger[level] += spill
        # Listen to the vector's natural scale, not arbitrary thresholds
        e_dir2 = e_vec / max(float(np.linalg.norm(e_vec)), 1e-9) if float(np.linalg.norm(e_vec)) > 1e-9 else np.array([0.0, 0.0, 0.0])
        alignment = self.axis_dirs @ e_dir2

        # Apply percept hook to enhance input processing
        enhanced_alignment = self.percept(alignment)

        # Ensure bias has same shape as P for broadcasting
        bias = self.beta * Delta * enhanced_alignment[:len(P)] if len(enhanced_alignment) > len(P) else self.beta * Delta * enhanced_alignment
        # Listen to the probability landscape without artificial floors or ceilings
        logits = np.log(np.maximum(P, 1e-9)) + bias / max(abs(x), 1e-9)
        logits -= np.max(logits)
        probs = np.exp(logits); probs /= probs.sum()
        j = np.random.choice(self.S, p=probs)
        symbol = self.idx_to_symbol[j]
        alleles = []
        # Strategic threshold for significant events, with probabilistic element
        if maxc_raw > 1.0 or abs(Delta) > self.delta_spawn:
            alleles.append(q_inj.copy())  # Let it drift naturally, don't force normalization

        # 0.1% tint: subtle influence on significant events
        if abs(Delta) > self.delta_spawn * 0.1:  # 10% of spawn threshold
            # Apply tiny tint to main quaternion evolution
            tint_strength = 0.001 * abs(Delta)  # 0.1% maximum
            q_new = q_new + tint_strength * q_inj
            # Renormalize after tint to maintain stability
            q_new = self._quat_norm(q_new)

        # Return CE1 color info as part of the result for visualization
        color_info = self.emit_ce1_color_info(Delta)
        return symbol, q_new, alleles, color_info

    def run(self, steps=64):
        q = np.array([1.0, 0.0, 0.0, 0.0]); symbol = 'drift'
        bank = [[] for _ in range(self.levels)]; finals = []
        color_trail = []  # Track CE1 color information
        for k in range(self.levels):
            qk = q.copy(); symk = symbol
            level_colors = []
            for _ in range(steps):
                symk, qk, born, color_info = self.step(k, qk, symk)
                if born: bank[k].extend(born)
                level_colors.append(color_info)
            finals.append(qk.copy()); q = qk
            color_trail.append(level_colors)
        return {'banks': bank, 'final_q': finals, 'color_trail': color_trail}

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
        imag_xy_dom = abs(x) > abs(w) and abs(y) > abs(w) and abs(x) > abs(z) and abs(y) > abs(z)
        imag_z_dom = abs(z) > abs(w) and abs(z) > abs(x) and abs(z) > abs(y)

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

    @staticmethod
    def _extract_float_bits(value: float) -> tuple[int, int, float]:
        """Extract sign, exponent, and mantissa from IEEE 754 float for CE1 color representation"""
        if abs(value) < 1e-9:  # Listen to near-zero conditions, not exact zero
            return 0, 0, 0.0

        # Convert to IEEE 754 representation
        import struct
        packed = struct.pack('>f', abs(value))
        bits = struct.unpack('>I', packed)[0]

        # Extract sign, exponent, mantissa
        sign = 1 if value < 0 else 0
        exponent = (bits >> 23) & 0xFF
        mantissa_bits = bits & 0x7FFFFF

        # Convert mantissa to 0-1 range (add implicit leading 1)
        mantissa = 1.0 + (mantissa_bits / float(1 << 23))

        return sign, exponent, mantissa

    def _delta_to_color(self, delta: float) -> tuple[float, float, float]:
        """Convert delta to CE1 color using Sierpiński HSV bead palette"""
        sign, exponent, mantissa = self._extract_float_bits(delta)

        # Sierpiński triangle HSV mapping
        hue, saturation, brightness = self._sierpinski_hsv_bead(delta, mantissa, exponent)

        return hue, saturation, brightness

    def _sierpinski_hsv_bead(self, delta: float, mantissa: float, exponent: int) -> tuple[float, float, float]:
        """Map delta to Sierpiński triangle HSV color space"""
        # Normalize mantissa to [0,1] for barycentric coordinates
        m = mantissa

        # Map exponent to triangle level (determines color intensity)
        level = max(0, min(8, exponent - 120))  # Focus on reasonable exponent range

        # Sierpiński triangle vertices in HSV space
        # Each vertex represents a primary color point
        vertices = {
            'red':    (0.0, 1.0, 0.8),    # Red vertex
            'green':  (120.0, 1.0, 0.8),  # Green vertex
            'blue':   (240.0, 1.0, 0.8),  # Blue vertex
        }

        # Determine which triangle we're in based on mantissa
        # This creates the fractal pattern
        triangle_coords = self._mantissa_to_triangle_coords(m)

        # Barycentric interpolation between triangle vertices
        hue = (triangle_coords[0] * vertices['red'][0] +
               triangle_coords[1] * vertices['green'][0] +
               triangle_coords[2] * vertices['blue'][0])

        saturation = (triangle_coords[0] * vertices['red'][1] +
                     triangle_coords[1] * vertices['green'][1] +
                     triangle_coords[2] * vertices['blue'][1])

        # Brightness varies with fractal level and delta magnitude
        base_brightness = (triangle_coords[0] * vertices['red'][2] +
                          triangle_coords[1] * vertices['green'][2] +
                          triangle_coords[2] * vertices['blue'][2])

        # Modulate brightness by delta and fractal level
        level_factor = 1.0 / (1.0 + level * 0.1)  # Deeper levels = dimmer
        delta_factor = min(1.0, abs(delta) * 10.0)  # Stronger deltas = brighter

        brightness = base_brightness * level_factor * delta_factor

        return hue, saturation, brightness

    def _mantissa_to_triangle_coords(self, mantissa: float) -> tuple[float, float, float]:
        """Convert mantissa to Sierpiński triangle barycentric coordinates"""
        # Use mantissa bits to determine position in fractal triangle
        # This creates the characteristic Sierpiński pattern

        # Convert mantissa to binary-like representation
        m_bits = []
        temp_m = mantissa
        for _ in range(12):  # Use 12 bits for good resolution
            temp_m *= 2
            bit = int(temp_m)
            m_bits.append(bit)
            temp_m -= bit

        # Use bit pattern to determine triangle subdivision
        # Each bit determines whether we're in upper or lower sub-triangle
        coords = [1.0, 0.0, 0.0]  # Start at red vertex

        for bit in m_bits[:8]:  # Use first 8 bits
            if bit == 0:
                # Move toward green vertex
                coords[0] *= 0.5  # Reduce red
                coords[1] += coords[0]  # Add to green
            else:
                # Move toward blue vertex
                coords[0] *= 0.5  # Reduce red
                coords[2] += coords[0]  # Add to blue

        # Normalize to ensure sum = 1
        total = sum(coords)
        if total > 0:
            coords = [c / total for c in coords]

        return (coords[0], coords[1], coords[2])

    def _analyze_color_phases_clustering(self, color_samples: List[Tuple[float, float, float, float]], n_clusters: int = 8) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        Perform cluster analysis on color samples to discover color phase alleles.
        Each sample is (delta, hue, saturation, brightness)
        """
        if len(color_samples) < n_clusters:
            # Not enough samples for clustering
            return [], np.array([])

        # Extract features for clustering: (hue, saturation, brightness, delta_magnitude)
        features = []
        for delta, h, s, b in color_samples:
            delta_mag = abs(delta)
            features.append([h, s, b, delta_mag])

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform k-means clustering using engine's deterministic random state
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.rng_mod)
        clusters = kmeans.fit_predict(features_scaled)

        # Analyze each cluster to create color phase alleles
        color_phases = []
        cluster_centers = kmeans.cluster_centers_

        for i in range(n_clusters):
            cluster_samples = [color_samples[j] for j in range(len(color_samples)) if clusters[j] == i]
            if not cluster_samples:
                continue

            # Analyze cluster characteristics
            deltas = [s[0] for s in cluster_samples]
            hues = [s[1] for s in cluster_samples]
            saturations = [s[2] for s in cluster_samples]
            brightnesses = [s[3] for s in cluster_samples]

            # Create color phase allele from cluster properties
            phase_allele = {
                'cluster_id': i,
                'hue_center': float(np.mean(hues)),
                'hue_range': float(np.std(hues)),
                'sat_center': float(np.mean(saturations)),
                'sat_range': float(np.std(saturations)),
                'bright_center': float(np.mean(brightnesses)),
                'bright_range': float(np.std(brightnesses)),
                'delta_center': float(np.mean(deltas)),
                'delta_range': float(np.std(deltas)),
                'sample_count': len(cluster_samples),
                'cluster_weight': len(cluster_samples) / len(color_samples)
            }
            color_phases.append(phase_allele)

        return color_phases, cluster_centers

    def _collect_color_samples_during_training(self, training_texts: List[str]) -> List[Tuple[float, float, float, float]]:
        """
        Collect color samples during training for cluster analysis.
        Returns list of (delta, hue, saturation, brightness) tuples.
        """
        color_samples = []

        for text in training_texts[:100]:  # Sample from first 100 texts for efficiency
            try:
                # Generate a temporary quaternion from text features
                q, temp = self._prompt_to_q_temp(text)
                symbol = self._symbol_from_content(text)
                feat = self._content_features(text)
                amp = self._amp_from_features(feat)

                # Simulate a few steps to get delta values
                q_current = q.copy()
                for step in range(min(10, len(text))):  # Limit steps per text
                    mu = 0.5  # Simplified mu for sampling
                    delta = amp - mu

                    # Get color for this delta
                    hue, saturation, brightness = self._delta_to_color(delta)
                    color_samples.append((delta, hue, saturation, brightness))

                    # Simple state evolution for next iteration
                    # Use deterministic randomness for reproducible sampling
                    rng = np.random.RandomState(self.rng_mod + step + hash(text) % 1000)
                    q_current = q_current + 0.1 * rng.randn(4)
                    q_current = self._quat_norm(q_current)
                    amp = amp * 0.9 + 0.1 * rng.random()  # Drift amplitude

            except Exception as e:
                continue

        return color_samples

    def _apply_color_phase_allele(self, delta: float, phase_allele: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Apply a color phase allele to transform a delta value to color.
        This allows geometric transformation by using learned color phase mappings.
        """
        # Use phase allele to modulate the standard color mapping
        base_hue, base_sat, base_bright = self._delta_to_color(delta)

        # Apply phase transformations
        phase_hue = base_hue
        phase_sat = base_sat
        phase_bright = base_bright

        # Modulate based on phase allele properties
        if 'hue_center' in phase_allele:
            # Blend between base hue and phase center based on delta similarity
            delta_similarity = 1.0 / (1.0 + abs(delta - phase_allele['delta_center']))
            phase_hue = base_hue * (1 - delta_similarity) + phase_allele['hue_center'] * delta_similarity

        if 'sat_center' in phase_allele:
            sat_factor = min(1.0, phase_allele['cluster_weight'] * 2.0)
            phase_sat = base_sat * (1 - sat_factor) + phase_allele['sat_center'] * sat_factor

        if 'bright_center' in phase_allele:
            bright_factor = min(1.0, phase_allele['cluster_weight'] * 1.5)
            phase_bright = base_bright * (1 - bright_factor) + phase_allele['bright_center'] * bright_factor

        return (phase_hue, phase_sat, phase_bright)

    def transform_color_between_geometries(self, delta: float, source_geometry: str = 'default',
                                         target_geometry: str = 'default') -> Tuple[float, float, float]:
        """
        Transform a delta value's color representation between different geometric templates
        using learned color phase alleles.
        """
        if not self.color_phase_alleles:
            # Fall back to standard color mapping if no phases learned
            return self._delta_to_color(delta)

        # Select appropriate color phase based on geometry and delta characteristics
        best_phase = None
        best_similarity = 0.0

        for phase in self.color_phase_alleles:
            # Calculate similarity based on delta properties and geometry
            delta_similarity = 1.0 / (1.0 + abs(delta - phase.get('delta_center', 0)))
            geometry_weight = phase.get('cluster_weight', 0.1)

            # Combine similarity metrics
            similarity = delta_similarity * geometry_weight

            if similarity > best_similarity:
                best_similarity = similarity
                best_phase = phase

        if best_phase:
            return self._apply_color_phase_allele(delta, best_phase)
        else:
            return self._delta_to_color(delta)

    def get_color_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned color phase alleles"""
        if not self.color_phase_alleles:
            return {'total_phases': 0}

        stats = {
            'total_phases': len(self.color_phase_alleles),
            'phases': []
        }

        for phase in self.color_phase_alleles:
            phase_stats = {
                'id': phase.get('cluster_id', 0),
                'sample_count': phase.get('sample_count', 0),
                'cluster_weight': phase.get('cluster_weight', 0),
                'delta_range': ".3f",
                'hue_range': ".1f",
                'color_center': ".1f"
            }
            stats['phases'].append(phase_stats)

        return stats

    def demonstrate_color_phase_transformation(self, test_deltas: Optional[List[float]] = None) -> str:
        """
        Demonstrate color phase transformations for geometric template conversion.
        """
        if not self.color_phase_alleles:
            return "No color phase alleles available for demonstration"

        if test_deltas is None:
            test_deltas = [0.1, 0.5, -0.2, 1.0, -0.8]  # Default test values

        demonstration = ["🎨 Color Phase Transformation Demonstration", "=" * 50]

        for delta in test_deltas:
            # Standard color mapping
            std_hue, std_sat, std_bright = self._delta_to_color(delta)

            # Phase-transformed color mapping
            phase_hue, phase_sat, phase_bright = self.transform_color_between_geometries(delta)

            demonstration.append(f"\nΔ = {delta:.3f}:")
            demonstration.append(f"  Standard:  H={std_hue:.1f}° S={std_sat:.3f} B={std_bright:.3f}")
            demonstration.append(f"  Phase-X:   H={phase_hue:.1f}° S={phase_sat:.3f} B={phase_bright:.3f}")

            # Show the difference
            hue_diff = abs(phase_hue - std_hue)
            sat_diff = abs(phase_sat - std_sat)
            bright_diff = abs(phase_bright - std_bright)
            demonstration.append(f"  ΔTransform: H±{hue_diff:.1f}° S±{sat_diff:.3f} B±{bright_diff:.3f}")

        return '\n'.join(demonstration)

    def _sample_symbol_sequence(self, level: int, steps: int, start_symbol: str = 'drift', prompt: Optional[str]=None) -> List[str]:
        seq = [start_symbol]; i = self.symbol_to_idx.get(start_symbol, 0)
        align = None; temp = 1.0
        if prompt:
            q_p, temp = self._prompt_to_q_temp(prompt)
            e_dir = q_p[1:]; n = float(np.linalg.norm(e_dir)); e_dir = e_dir/(n if n else 1.0)
            align = self.axis_dirs @ e_dir
        for _ in range(steps-1):
            baseP = self.transitions_per_level[level][i]
            if align is None:
                probs = baseP
            else:
                logits = np.log(baseP + 1e-12) + 0.8*align
                logits = logits / max(0.1, float(temp))
                logits -= np.max(logits)
                probs = np.exp(np.clip(logits, -60.0, 60.0)); probs /= probs.sum()
            j = int(np.random.choice(self.S, p=probs))
            seq.append(self.idx_to_symbol[j]); i = j
        return seq

    def _token_from_q(self, q: np.ndarray, symbol: str) -> str:
        psi = float(np.dot(self.basis_weights, q[:3])); eps = max(self.delta_spawn, abs(psi))
        return (f"q⊗rot({eps:.2f},x) ⊕= ℛ_F(q)" if symbol=='carry' else (f"q⊗rot({eps:.2f},y) → fuse(σ,σ′,e)" if symbol=='borrow' else f"q⊗rot({eps:.2f},z) → drift"))

    def generate_text(self, seed_text: str, banks: Optional[List[List[np.ndarray]]] = None, finals: Optional[List[np.ndarray]] = None,
                      style: str = "hybrid", length: int = 100, **kwargs) -> str:
        """
        Unified text generation that combines multiple approaches.

        Styles:
        - "hybrid": Combine markov + phenotype + CE1 algebra
        - "pheno": Phenotype sentences from alleles
        - "markov": Bigram model from training data
        - "algebra": CE1 mathematical notation
        - "descriptive": Human-readable allele descriptions

        Args:
            seed_text: Input prompt/seed
            banks, finals: Allele data (required for pheno/algebra styles)
            style: Generation style
            length: Target output length in characters
        """
        if style == "pheno" and banks and finals:
            return self._generate_pheno_only(seed_text, banks, finals, max(1, length//30))
        elif style == "markov":
            return self._generate_markov_only(seed_text, length)
        elif style == "algebra" and banks and finals:
            return self._generate_algebra_only(seed_text, banks, finals, length)
        elif style == "descriptive":
            return self._generate_descriptive_only(finals or [], length)

        # Hybrid approach - combine all methods
        return self._generate_hybrid(seed_text, banks, finals, length)

    # ===== TEXT GENERATION METHODS =====

    def _generate_pheno_only(self, seed_text: str, banks: Optional[List[List[np.ndarray]]], finals: Optional[List[np.ndarray]], sentences: int = 3) -> str:
        """Generate phenotype sentences from alleles"""
        return self.generate_pheno_text(seed_text, banks, finals, sentences)

    def _generate_markov_only(self, prompt: str, length: int = 50) -> str:
        """Generate text using trained bigram model"""
        return self.generate_markov_text(prompt, length)

    def _generate_algebra_only(self, seed_text: str, banks: Optional[List[List[np.ndarray]]], finals: Optional[List[np.ndarray]], tokens: int = 80) -> str:
        """Generate CE1 mathematical notation"""
        chosen = self.select_starting_alleles_from_prompt(seed_text, banks)
        qs = [q for q in chosen if q is not None]
        q_base = self._quat_norm(np.mean(np.stack(qs, axis=0), axis=0)) if qs else np.array([1.0,0.0,0.0,0.0])
        q_p, _ = self._prompt_to_q_temp(seed_text)
        q_mix = self._quat_norm(q_base + q_p)
        seq = self._sample_symbol_sequence(0, max(3, tokens//4), 'drift', prompt=seed_text)
        return (' ; '.join(self._token_from_q(q_mix, s) for s in seq))[: max(tokens, 40)]

    def _generate_descriptive_only(self, finals: List[np.ndarray], length: int = 120) -> str:
        """Generate human-readable descriptions of final quaternions"""
        if not finals:
            return "No final states to describe"
        descriptions = []
        for i, q in enumerate(finals):
            desc = self.emit_text_from_allele(q, length//len(finals))
            descriptions.append(f"Final state {i}: {desc}")
        return ' | '.join(descriptions)

    def _generate_hybrid(self, seed_text: str, banks: Optional[List[List[np.ndarray]]], finals: Optional[List[np.ndarray]], length: int = 100) -> str:
        """Generate hybrid text combining multiple approaches"""
        parts = []
        remaining = length

        # Start with phenotype description (20% of length)
        if banks and finals:
            pheno_len = max(20, length//5)
            parts.append(f"PHENOTYPE: {self._generate_pheno_only(seed_text, banks, finals, 1)}")
            remaining -= pheno_len

        # Add mathematical description (30% of length)
        if banks and finals and remaining > 10:
            math_len = max(20, remaining//3)
            parts.append(f"MATHEMATICS: {self._generate_algebra_only(seed_text, banks, finals, math_len)}")
            remaining -= math_len

        # Fill with markov text (remaining length)
        if remaining > 10:
            parts.append(f"MARKOV: {self._generate_markov_only(seed_text, remaining)}")

        return ' | '.join(parts)

    def _tok(self, text: str) -> List[str]:
        """Adaptive tokenization - learn tokens from data patterns"""
        # Use learned tokenization if available, otherwise fall back to basic
        if hasattr(self, 'learned_tokenizer') and self.learned_tokenizer:
            return self._apply_learned_tokenizer(text)
        else:
            return self._basic_tok(text)

    def _basic_tok(self, text: str) -> List[str]:
        """Basic fallback tokenization"""
        return [''.join(ch for ch in tok.lower() if ch.isalnum()) for tok in text.split()
                if len(''.join(ch for ch in tok if ch.isalnum())) >= 2]

    def _apply_learned_tokenizer(self, text: str) -> List[str]:
        """Apply learned tokenization patterns"""
        tokens = []
        i = 0
        while i < len(text):
            # Try longest learned tokens first
            found = False
            for token_len in range(min(self.max_token_len, len(text) - i), 0, -1):
                candidate = text[i:i + token_len]
                if self.learned_tokenizer and candidate in self.learned_tokenizer:
                    tokens.append(candidate)
                    i += token_len
                    found = True
                    break
            if not found:
                # Single character fallback
                tokens.append(text[i])
                i += 1
        return tokens

    def _learn_tokenization_excavation(self, training_texts: List[str], iteration: int,
                                      assessment_context: Optional[Dict[str, Any]] = None) -> None:
        """Unified tokenization learning with assessment-guided strategy"""
        # Use assessment context to determine optimal strategy
        if assessment_context and iteration == 0:
            info_content = assessment_context.get('avg_info_content', 0.5)
            if info_content > 0.7:
                # High-quality data: Use sophisticated first-pass tokenization
                print(f"🔬 Iteration 0: Sophisticated tokenization (high-quality data)")
                self._learn_tokenization_sophisticated(training_texts, assessment_context)
            elif info_content > 0.4:
                # Medium-quality data: Use enhanced statistical approach
                print(f"🔬 Iteration 0: Enhanced statistical tokenization")
                self._learn_tokenization_enhanced(training_texts)
            else:
                # Low-quality data: Conservative approach
                print(f"🔬 Iteration 0: Conservative tokenization")
            self._learn_tokenization(training_texts)
        elif iteration == 0:
            # No assessment context: Fall back to enhanced approach
            print(f"🔬 Iteration 0: Enhanced statistical tokenization")
            self._learn_tokenization_enhanced(training_texts)
        else:
            # Subsequent iterations: use markov model to guide tokenization
            print(f"🔬 Using markov-guided tokenization for iteration {iteration + 1}")

            # Get frequent sequences from current markov model
            if hasattr(self, 'bigram_counts') and self.bigram_counts:
                # Find high-probability sequences that should be single tokens
                markov_guided_tokens = self._extract_markov_guided_tokens()

                # Combine with existing learned tokens
                if hasattr(self, 'learned_tokenizer') and self.learned_tokenizer:
                    self.learned_tokenizer.update(markov_guided_tokens)
                else:
                    self.learned_tokenizer = markov_guided_tokens

                print(f"🔬 Markov-guided: added {len(markov_guided_tokens)} new tokens")
            else:
                # Fall back to basic learning if no markov model yet
                self._learn_tokenization(training_texts)

    def _extract_markov_guided_tokens(self) -> set:
        """Extract tokens based on markov model high-probability sequences"""
        guided_tokens = set()

        # Find frequent bigrams with high transition probabilities
        for word, next_words in self.bigram_counts.items():
            total_count = sum(next_words.values())
            for next_word, count in next_words.items():
                if count > total_count * 0.1:  # >10% of transitions from this word
                    # This is a strong transition - consider merging
                    if len(word) + len(next_word) < 15:  # Reasonable length
                        merged_token = f"{word} {next_word}"
                        guided_tokens.add(merged_token)

        # Also consider trigrams from high-frequency patterns
        frequent_words = [w for w, c in sorted(self.unigram_counts.items(), key=lambda x: x[1], reverse=True)[:50]]
        for i in range(len(frequent_words) - 2):
            w1, w2, w3 = frequent_words[i:i+3]
            # Check if w1->w2 and w2->w3 are both strong transitions
            if (w1 in self.bigram_counts and w2 in self.bigram_counts[w1] and
                w2 in self.bigram_counts and w3 in self.bigram_counts[w2]):
                trigram_token = f"{w1} {w2} {w3}"
                guided_tokens.add(trigram_token)

        return guided_tokens

    def _learn_tokenization_enhanced(self, training_texts: List[str]) -> None:
        """Enhanced tokenization learning using statistical analysis"""
        print(f"📊 Analyzing {len(training_texts)} texts for tokenization patterns...")

        # Multi-pass analysis for better pattern discovery
        word_freq = {}
        word_pairs = {}
        char_ngrams = {}
        text_lengths = []

        for text in training_texts:
            text_lengths.append(len(text))
            basic_tokens = self._basic_tok(text)

            # Word-level analysis
            for i, token in enumerate(basic_tokens):
                if token and len(token) >= 2:
                    word_freq[token] = word_freq.get(token, 0) + 1

                if i < len(basic_tokens) - 1:
                    next_token = basic_tokens[i + 1]
                    if token and next_token and len(token) >= 2 and len(next_token) >= 2:
                        pair = f"{token} {next_token}"
                        word_pairs[pair] = word_pairs.get(pair, 0) + 1

            # Character n-gram analysis for subword patterns
            for n in [3, 4, 5]:  # Trigram to pentagram
                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    # Only consider alphanumeric ngrams
                    if all(c.isalnum() or c.isspace() for c in ngram):
                        char_ngrams[ngram] = char_ngrams.get(ngram, 0) + 1

        # Calculate corpus statistics
        total_words = sum(word_freq.values())
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

        print(f"📈 Corpus stats: {total_words} words, {len(word_pairs)} pairs, "
              f"{len(char_ngrams)} ngrams, avg length {avg_text_length:.1f}")

        # Enhanced token learning
        self.learned_tokenizer = set()
        self.max_token_len = 15  # Allow longer tokens

        # 1. High-frequency words (top percentile)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        freq_threshold = max(3, int(len(training_texts) * 0.001))  # 0.1% frequency
        frequent_words = [word for word, freq in sorted_words if freq >= freq_threshold]

        print(f"📝 Found {len(frequent_words)} frequent words")

        # 2. Collocation analysis (mutual information)
        collocations = []
        for pair, pair_freq in word_pairs.items():
            words = pair.split()
            if len(words) == 2:
                w1, w2 = words
                w1_freq = word_freq.get(w1, 1)
                w2_freq = word_freq.get(w2, 1)

                # Calculate mutual information-like score
                expected = (w1_freq * w2_freq) / total_words
                if expected > 0:
                    mi_score = pair_freq * math.log(pair_freq / expected)
                    collocations.append((pair, mi_score, pair_freq))

        # Sort by MI score and take top collocations
        collocations.sort(key=lambda x: x[1], reverse=True)
        strong_collocations = [pair for pair, score, freq in collocations[:100]]  # Top 100

        print(f"🔗 Found {len(strong_collocations)} strong collocations")

        # 3. Statistical n-gram patterns
        sorted_ngrams = sorted(char_ngrams.items(), key=lambda x: x[1], reverse=True)
        ngram_threshold = max(2, len(training_texts) // 100)  # Adaptive threshold
        frequent_ngrams = [ngram for ngram, freq in sorted_ngrams[:200]  # Top 200
                          if freq >= ngram_threshold and ' ' in ngram]  # Must contain space

        print(f"🔤 Found {len(frequent_ngrams)} frequent ngrams")

        # Combine all discovered patterns
        all_candidates = set()
        all_candidates.update(frequent_words)
        all_candidates.update(strong_collocations)
        all_candidates.update(frequent_ngrams)

        # Filter and add to learned tokenizer
        for candidate in all_candidates:
            if 2 <= len(candidate) <= self.max_token_len:
                # Additional filtering: prefer meaningful patterns
                if any(c.isalnum() for c in candidate):  # Must contain some alphanumeric
                    self.learned_tokenizer.add(candidate)

        # Show learning results
        if self.learned_tokenizer:
            single_words = [t for t in self.learned_tokenizer if ' ' not in t]
            phrases = [t for t in self.learned_tokenizer if ' ' in t]

            print(f"📚 Enhanced tokenization learned {len(self.learned_tokenizer)} patterns")
            print(f"   📝 Single words: {len(single_words)}")
            print(f"   💬 Phrases: {len(phrases)}")

            # Show diverse examples
            examples = []
            if single_words:
                examples.extend(sorted(single_words)[:5])
            if phrases:
                examples.extend(sorted(phrases)[:3])
            if examples:
                print(f"   Examples: {', '.join(examples)}")
        else:
            print("📚 Using basic tokenization patterns")

        # If we learned too few patterns, fall back to basic learning
        if len(self.learned_tokenizer) < 10:
            print("⚠ Insufficient enhanced patterns, falling back to basic learning")
            self._learn_tokenization(training_texts)

    def _learn_tokenization_sophisticated(self, training_texts: List[str], assessment_context: Dict[str, Any]) -> None:
        """Sophisticated tokenization using assessment context for optimal pattern discovery"""
        print(f"🎯 Sophisticated tokenization with assessment-guided optimization")

        # Extract assessment insights
        avg_entropy = assessment_context.get('avg_entropy', 3.0)
        linguistic_complexity = assessment_context.get('linguistic_complexity', 0.5)
        pattern_diversity = assessment_context.get('pattern_diversity', 0.5)

        # Adaptive parameters based on data characteristics
        if avg_entropy > 4.0:  # High entropy = complex language
            min_freq_threshold = max(3, int(len(training_texts) * 0.0005))  # 0.05%
            ngram_range = [2, 3, 4, 5]  # Longer patterns for complex content
            mi_threshold = 2.0  # Higher mutual information threshold
        elif avg_entropy > 2.5:  # Medium entropy = natural language
            min_freq_threshold = max(2, int(len(training_texts) * 0.001))  # 0.1%
            ngram_range = [2, 3, 4]  # Standard patterns
            mi_threshold = 1.5
        else:  # Low entropy = structured content
            min_freq_threshold = max(2, int(len(training_texts) * 0.002))  # 0.2%
            ngram_range = [2, 3]  # Shorter patterns for structured content
            mi_threshold = 1.0

        print(f"   📊 Adaptive parameters: freq≥{min_freq_threshold}, ngrams={ngram_range}, MI≥{mi_threshold:.1f}")

        # Enhanced pattern discovery with assessment-guided filtering
        word_freq = {}
        word_pairs = {}
        char_ngrams = {}
        text_lengths = []

        for text in training_texts:
            text_lengths.append(len(text))
            basic_tokens = self._basic_tok(text)

            # Word-level analysis with frequency filtering
            for i, token in enumerate(basic_tokens):
                if len(token) >= 2:
                    word_freq[token] = word_freq.get(token, 0) + 1

                if i < len(basic_tokens) - 1:
                    next_token = basic_tokens[i + 1]
                    if len(token) >= 2 and len(next_token) >= 2:
                        pair = f"{token} {next_token}"
                        word_pairs[pair] = word_pairs.get(pair, 0) + 1

            # Adaptive n-gram analysis
            for n in ngram_range:
                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    if all(c.isalnum() or c.isspace() for c in ngram):
                        char_ngrams[ngram] = char_ngrams.get(ngram, 0) + 1

        # Sophisticated collocation analysis with higher threshold
        collocations = []
        for pair, pair_freq in word_pairs.items():
            words = pair.split()
            if len(words) == 2:
                w1, w2 = words
                w1_freq = word_freq.get(w1, 1)
                w2_freq = word_freq.get(w2, 1)

                if pair_freq >= min_freq_threshold:
                    expected = (w1_freq * w2_freq) / max(sum(word_freq.values()), 1)
                    if expected > 0:
                        mi_score = pair_freq * math.log(pair_freq / expected)
                        if mi_score >= mi_threshold:  # Higher threshold for sophisticated analysis
                            collocations.append((pair, mi_score, pair_freq))

        # Sort by MI score
        collocations.sort(key=lambda x: x[1], reverse=True)
        strong_collocations = [pair for pair, score, freq in collocations[:50]]  # Top 50

        # Initialize tokenizer with sophisticated patterns
        self.learned_tokenizer = set()
        self.max_token_len = 20  # Allow very long tokens for sophisticated analysis

        # Add patterns with quality filtering
        for candidate in strong_collocations:
            if self._assess_pattern_quality(candidate, assessment_context):
                self.learned_tokenizer.add(candidate)

        # Add high-frequency words with quality filtering
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:100]:  # Top 100 words
            if freq >= min_freq_threshold and self._assess_pattern_quality(word, assessment_context):
                self.learned_tokenizer.add(word)

        # Add sophisticated n-grams
        sorted_ngrams = sorted(char_ngrams.items(), key=lambda x: x[1], reverse=True)
        for ngram, freq in sorted_ngrams[:100]:  # Top 100 ngrams
            if freq >= min_freq_threshold and ' ' in ngram and self._assess_pattern_quality(ngram, assessment_context):
                self.learned_tokenizer.add(ngram)

        print(f"🎯 Sophisticated tokenization completed: {len(self.learned_tokenizer)} high-quality patterns")

    def _assess_pattern_quality(self, pattern: str, assessment_context: Dict[str, Any]) -> bool:
        """Assess if a pattern is worth including based on assessment context"""
        # Basic quality checks
        if len(pattern) < 2 or len(pattern) > self.max_token_len:
            return False

        # Linguistic quality assessment
        linguistic_complexity = assessment_context.get('linguistic_complexity', 0.5)

        if linguistic_complexity > 0.7:
            # High complexity: Prefer longer, more specific patterns
            return len(pattern) >= 4 and any(c.isalpha() for c in pattern)
        elif linguistic_complexity > 0.3:
            # Medium complexity: Standard patterns
            return len(pattern) >= 3
        else:
            # Low complexity: Shorter, more frequent patterns
            return len(pattern) >= 2

    def _learn_tokenization(self, training_texts: List[str]) -> None:
        """Learn tokenization patterns using conservative word-based analysis"""
        # Use basic tokenization to discover meaningful word patterns
        word_freq = {}
        word_pairs = {}

        for text in training_texts:
            # Use basic tokenization to get candidate words
            basic_tokens = self._basic_tok(text)

            for i, token in enumerate(basic_tokens):
                # Only consider alphanumeric tokens
                if token and len(token) >= 2:
                    word_freq[token] = word_freq.get(token, 0) + 1

                # Count word pairs for context learning
                if i < len(basic_tokens) - 1:
                    next_token = basic_tokens[i + 1]
                    if token and next_token and len(token) >= 2 and len(next_token) >= 2:
                        pair = f"{token} {next_token}"
                        word_pairs[pair] = word_pairs.get(pair, 0) + 1

        # Learn meaningful tokens (conservative approach)
        self.learned_tokenizer = set()
        self.max_token_len = 10  # Allow longer word-based tokens

        # Add frequent meaningful words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_words[:200]:  # Top 200 words
            if freq > len(training_texts) * 0.005:  # At least 0.5% frequency
                self.learned_tokenizer.add(word)

        # Add frequent word pairs (phrases)
        sorted_pairs = sorted(word_pairs.items(), key=lambda x: x[1], reverse=True)

        for pair, freq in sorted_pairs[:50]:  # Top 50 phrases
            if freq > len(training_texts) * 0.002:  # At least 0.2% frequency
                self.learned_tokenizer.add(pair)

        # Show what we learned
        if self.learned_tokenizer:
            # Categorize learned patterns
            single_words = [t for t in self.learned_tokenizer if ' ' not in t]
            phrases = [t for t in self.learned_tokenizer if ' ' in t]

            print(f"📚 Learned {len(self.learned_tokenizer)} tokenization patterns")
            print(f"   📝 Single words: {len(single_words)}")
            print(f"   💬 Phrases: {len(phrases)}")

            # Show diverse examples
            examples = []
            if single_words:
                examples.extend(sorted(single_words)[:5])
            if phrases:
                examples.extend(sorted(phrases)[:3])
            if examples:
                print(f"   Examples: {', '.join(examples)}")
        else:
            print("📚 Using basic tokenization patterns")

        # If we learned too few patterns, fall back to basic tokenization
        if len(self.learned_tokenizer) < 10:
            print("⚠ Insufficient patterns learned, using basic tokenization")
            self.learned_tokenizer = None

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
        hue, saturation, brightness = self._delta_to_color(delta)
        sign, exponent, mantissa = self._extract_float_bits(delta)

        return f"CE1{{Δ={delta:.6f}; sign={sign}; exp={exponent}; mantissa={mantissa:.6f}; color=hsb({hue:.1f}°, {saturation:.3f}, {brightness:.3f})}}"

    def allele_to_sentence(self, q: np.ndarray) -> str:
        if q is None or not self.vocab:
            return ""
        w, x, y, z = map(float, q)
        length = int(6 + 10 * min(abs(w), 1.0)); richness = min(abs(x), 1.0); rhythm = min(abs(y), 1.0); theme = min(abs(z), 1.0)
        N = len(self.vocab); cutoff = max(5, int(N * (0.3 + 0.5 * (1 - richness))))
        head = self.vocab[:cutoff]; tail = self.vocab[-cutoff:]
        words: List[str] = []
        for i in range(length):
            pool = head if random.random() < 0.7 else tail
            wtok = random.choice(pool)
            if theme > 0.5 and i == length // 2: wtok = random.choice(tail)
            words.append(wtok)
            if rhythm > 0.7 and random.random() < 0.2 and i < length-1: words.append(wtok)
        sent = ' '.join(words).strip(); sent = sent[:1].upper() + sent[1:]
        if not sent.endswith('.'): sent += '.'
        return sent

    def generate_pheno_text(self, seed_text: str, banks: Optional[List[List[np.ndarray]]], finals: Optional[List[np.ndarray]], sentences: int = 3) -> str:
        chosen = self.select_starting_alleles_from_prompt(seed_text, banks)
        qs = [q for q in chosen if q is not None] or ( [finals[-1]] if finals else [] )
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
    def _content_features(content: str) -> Dict[str, float]:
        length = len(content); lines = content.count('\n')
        if length == 0: return {'length': 0, 'lines': 0, 'entropy': 0.0, 'structure_ratio': 0.0}
        c = Counter(content); entropy = -sum((cnt/length)*math.log2(cnt/length) for cnt in c.values())
        return {'length': float(length), 'lines': float(lines), 'entropy': float(entropy), 'structure_ratio': float(lines/max(length,1))}

    def _symbol_from_content(self, content: str) -> str:
        f = self._content_features(content)
        return 'carry' if f['entropy']>4.0 else ('borrow' if f['structure_ratio']>0.1 else 'drift')

    @staticmethod
    def _amp_from_features(feat: Dict[str, float]) -> float:
        return 0.2 + 1.2 * (0.5 * (min(feat['length']/2000.0,1.0) + min(feat['entropy']/6.0,1.0)))

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

    def _write_learned_template(self, learned: Dict, mem_hash: str, vocab: List[str], bigrams: Optional[Dict[str, Dict[str,int]]]=None, unigrams: Optional[Dict[str,int]]=None, training_files: Optional[List[List[str]]]=None):
        # Compose TOML text manually to avoid extra deps
        lines: List[str] = []
        lines.append('# Learned Fractal Markov Allele Template')
        lines.append('[name]')
        lines.append('slug = "fractal_markov_learned"')
        lines.append('version = "0.1.0"')
        lines.append('')
        lines.append('[levels]')
        lines.append(f'count = {self.levels}')
        lines.append('')
        lines.append('[stochastic]')
        lines.append('seed = 1337')
        lines.append(f'noise = {self.noise}')
        lines.append('')
        lines.append('[fractal.scales]')
        for k in range(self.levels):
            lines.append(f'l{k} = {self.scales[k]}')
        lines.append('')
        lines.append('[fractal]')
        lines.append(f'self_similarity = {self.self_similarity}')
        lines.append(f'branch_factor = {self.branch_factor}')
        lines.append('')
        lines.append('[alphabet]')
        syms = ', '.join([f'"{s}"' for s in self.symbols])
        lines.append(f'symbols = [{syms}]')
        lines.append('')
        lines.append('[priors]')
        for i, s in enumerate(self.symbols):
            row = ', '.join([f'{t} = {float(learned["priors"][i,j]):.6f}' for j, t in enumerate(self.symbols)])
            lines.append(f'{s} = {{ {row} }}')
        lines.append('')
        lines.append('[emissions]')
        for s in self.symbols:
            ax = self.emissions.get(s, {}).get('axis', 'z')
            lo, hi = learned['emissions'][s]
            lines.append(f'{s} = {{ axis = "{ax}", amplitude = [{lo:.4f}, {hi:.4f}] }}')
        lines.append('')
        lines.append('[seed]')
        lines.append(f'alpha = {self.alpha}')
        lines.append(f'epsilon0 = {self.epsilon0}')
        lines.append(f'beta = {self.beta}')
        lines.append(f'kappa = {self.kappa}')
        lines.append(f'delta_spawn = {self.delta_spawn}')
        lines.append(f'basis_weights = [{", ".join(str(float(x)) for x in self.basis_weights)}]')
        lines.append('')
        lines.append('[cache]')
        lines.append(f'mem_hash = "{mem_hash}"')
        lines.append('built_by = "ce1_fractal_markov_engine.py"')
        # Store the precise training files used (git-style tracking)
        if training_files:
            items = ', '.join([f'["{n}","{h}",{sz}]' for n,h,sz in training_files])
            lines.append(f'files = [{items}]')

        # Store learned tokenizer if available
        if hasattr(self, 'learned_tokenizer') and self.learned_tokenizer:
            tokenizer_items = ', '.join(f'"{tok}"' for tok in sorted(self.learned_tokenizer))
            lines.append(f'learned_tokenizer = [{tokenizer_items}]')
            lines.append(f'max_token_len = {getattr(self, "max_token_len", 3)}')

        # Store learned color phase alleles if available
        if self.color_phase_alleles:
            lines.append('[color_phases]')
            for i, phase in enumerate(self.color_phase_alleles):
                phase_items = ', '.join(f'"{k}" = {v:.6f}' for k, v in phase.items() if isinstance(v, (int, float)))
                lines.append(f'phase_{i} = {{ {phase_items} }}')
            lines.append('')

        lines.append('')
        if vocab:
            top = vocab[:256]
            words = ', '.join([f'"{w}"' for w in top])
            lines.append('[vocab]')
            lines.append(f'words = [{words}]')
            lines.append('')
        if unigrams:
            ug_top = sorted(unigrams.items(), key=lambda kv: kv[1], reverse=True)[:5000]
            ug_items = ', '.join([f'["{k}",{v}]' for k,v in ug_top])
            lines.append('[markov]')
            lines.append(f'unigrams = [{ug_items}]')
            lines.append('')
        if bigrams:
            triples: List[tuple] = []
            for a, d in bigrams.items():
                for b, c in d.items():
                    triples.append((a, b, int(c)))
            triples.sort(key=lambda t: t[2], reverse=True)
            triples = triples[:8000]
            bg_items = ', '.join([f'["{a}","{b}",{c}]' for a,b,c in triples])
            if 'markov' not in ''.join(lines):
                lines.append('[markov]')
            lines.append(f'bigrams = [{bg_items}]')
            lines.append('')
        out_path = os.path.join(os.path.dirname(self.template_path), 'fractal_markov_alleles.learned.toml')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return out_path

    def _assess_file_information_content(self, file_path: str) -> Dict[str, float]:
        """Quick assessment of a file's information content for JIT excavation"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if getattr(self, 'assess_sample_chars', None):
                    content = f.read(int(self.assess_sample_chars))
                else:
                    content = f.read()

            if not content.strip():
                return {'total_score': 0.0}

            # Basic content metrics
            length = len(content)
            lines = len(content.split('\n'))
            chars_per_line = length / max(lines, 1)

            # Token diversity metrics
            basic_tokens = self._basic_tok(content)
            unique_tokens = len(set(basic_tokens))
            token_diversity = unique_tokens / max(len(basic_tokens), 1)

            # Character-level entropy
            char_counts = Counter(content)
            char_entropy = -sum((count/length) * math.log(count/length) for count in char_counts.values())

            # Word frequency distribution (power law detection)
            word_freq = {}
            for token in basic_tokens:
                if len(token) >= 2:
                    word_freq[token] = word_freq.get(token, 0) + 1

            if word_freq:
                freq_values = sorted(word_freq.values(), reverse=True)
                # Calculate how closely it follows Zipf's law
                zipf_score = 0
                for i, freq in enumerate(freq_values[:100]):  # Check top 100 words
                    expected = freq_values[0] / (i + 1)
                    zipf_score += abs(freq - expected) / expected
                zipf_score = 1.0 / (1.0 + zipf_score)  # Convert to similarity score
            else:
                zipf_score = 0.0

            # N-gram diversity (detects pattern richness)
            ngram_diversity = 0
            for n in [2, 3, 4]:
                ngrams = [content[i:i+n] for i in range(len(content)-n+1)]
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)
                if total_ngrams > 0:
                    ngram_diversity += unique_ngrams / total_ngrams
            ngram_diversity /= 3  # Average across n-gram sizes

            # Structural complexity
            structural_score = min(1.0, lines / 50.0)  # Prefer files with moderate structure

            # Combine metrics into overall score
            total_score = (
                token_diversity * 0.25 +      # Pattern diversity
                char_entropy * 0.20 +         # Information entropy
                zipf_score * 0.20 +           # Linguistic structure
                ngram_diversity * 0.20 +      # Sequence complexity
                structural_score * 0.15       # Document structure
            )

            return {
                'total_score': total_score,
                'length': length,
                'token_diversity': token_diversity,
                'char_entropy': char_entropy,
                'zipf_score': zipf_score,
                'ngram_diversity': ngram_diversity,
                'structural_score': structural_score
            }

        except Exception as e:
            return {'total_score': 0.0}

    def _prioritize_files_jit(self, file_paths: List[str], top_percentage: float = 0.3) -> Tuple[List[str], List[str]]:
        """Prioritize files using just-in-time excavation assessment"""
        print(f"🎯 JIT Assessment: Analyzing {len(file_paths)} files for information content...")

        file_scores = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            scores = self._assess_file_information_content(file_path)
            file_scores.append((file_name, file_path, scores))

        # Sort by information content score
        file_scores.sort(key=lambda x: x[2]['total_score'], reverse=True)

        # Select top percentage of files
        num_high_value = max(1, int(len(file_paths) * top_percentage))
        high_value_files = [fp for _, fp, _ in file_scores[:num_high_value]]
        remaining_files = [fp for _, fp, _ in file_scores[num_high_value:]]

        # Show assessment results
        print(f"   📊 High-value files ({len(high_value_files)}):")
        for i, (_, _, scores) in enumerate(file_scores[:5]):  # Show top 5
            file_name = os.path.basename(file_scores[i][1])
            score = scores['total_score']
            print(f"      {i+1}. {file_name}: {score:.3f}")

        if len(file_scores) > 5:
            print(f"      ... and {len(high_value_files) - 5} more")

        return high_value_files, remaining_files

    def _extract_patterns_from_high_value_files(self, high_value_files: List[str]) -> Dict[str, Any]:
        """Extract key patterns from high-value files for bootstrap learning"""
        print(f"🔍 Extracting bootstrap patterns from {len(high_value_files)} high-value files...")

        # Quick pattern extraction for seeding
        seed_patterns = {
            'frequent_words': set(),
            'common_phrases': set(),
            'structural_patterns': []
        }

        word_freq_global = {}
        phrase_freq_global = {}

        for file_path in high_value_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                tokens = self._basic_tok(content)

                # Word frequencies
                for i, token in enumerate(tokens):
                    if len(token) >= 2:
                        word_freq_global[token] = word_freq_global.get(token, 0) + 1

                    # Simple phrase detection
                    if i < len(tokens) - 1:
                        next_token = tokens[i + 1]
                        if len(token) >= 2 and len(next_token) >= 2:
                            phrase = f"{token} {next_token}"
                            phrase_freq_global[phrase] = phrase_freq_global.get(phrase, 0) + 1

            except Exception as e:
                continue

        # Select high-frequency patterns for seeding
        sorted_words = sorted(word_freq_global.items(), key=lambda x: x[1], reverse=True)
        sorted_phrases = sorted(phrase_freq_global.items(), key=lambda x: x[1], reverse=True)

        # Take top patterns for seeding
        seed_patterns['frequent_words'] = set(word for word, freq in sorted_words[:50])
        seed_patterns['common_phrases'] = set(phrase for phrase, freq in sorted_phrases[:20])

        print(f"   🎯 Bootstrap patterns: {len(seed_patterns['frequent_words'])} words, "
              f"{len(seed_patterns['common_phrases'])} phrases")

        return seed_patterns

    def _create_assessment_context(self, high_value_files: List[str]) -> Dict[str, Any]:
        """Create comprehensive assessment context from high-value files"""
        total_files = len(high_value_files)
        if total_files == 0:
            return {'avg_info_content': 0.5, 'linguistic_complexity': 0.5}

        # Aggregate metrics from high-value files
        total_info_content = 0.0
        total_entropy = 0.0
        total_diversity = 0.0
        total_zipf = 0.0

        for file_path in high_value_files:
            scores = self._assess_file_information_content(file_path)
            total_info_content += scores['total_score']
            total_entropy += scores['char_entropy']
            total_diversity += scores['token_diversity']
            total_zipf += scores['zipf_score']

        # Calculate averages
        avg_info_content = total_info_content / total_files
        avg_entropy = total_entropy / total_files
        avg_diversity = total_diversity / total_files
        avg_zipf = total_zipf / total_files

        # Derive linguistic complexity
        linguistic_complexity = min(1.0, (avg_entropy / 6.0) * (avg_diversity) * (avg_zipf + 0.5))

        # Pattern diversity based on n-gram diversity
        pattern_diversity = min(1.0, avg_diversity * linguistic_complexity)

        assessment_context = {
            'avg_info_content': avg_info_content,
            'avg_entropy': avg_entropy,
            'avg_diversity': avg_diversity,
            'avg_zipf': avg_zipf,
            'linguistic_complexity': linguistic_complexity,
            'pattern_diversity': pattern_diversity,
            'high_value_files_count': total_files
        }

        print(f"📊 Assessment Context: Info={avg_info_content:.3f}, "
              f"Entropy={avg_entropy:.1f}, Linguistic={linguistic_complexity:.3f}")

        return assessment_context

    def train_from_directory(self, directory='.', blend=0.5, include_hidden=False, write_template=True, incremental_files=None, excavation_iterations=3, enable_jit_excavation=True):
        """
        Train from directory with iterative tokenization-markov excavation.
        incremental_files: dict of {filename: (old_hash, new_hash)} for changed files
        excavation_iterations: number of tokenization-markov refinement cycles
        """
        start_time = time.time()
        time_budget = getattr(self, 'max_secs', None)
        files = []
        training_file_info = []  # Track files actually used for training

        for name in sorted(os.listdir(directory)):
            if not include_hidden and name.startswith('.'):
                continue
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                # Only include likely text sources
                if not any(path.lower().endswith(ext) for ext in ('.py', '.md', '.txt', '.toml', '.tid', '.json', '.cfg', '.ini')):
                    continue
                files.append(path)
        symbols_seq = []
        amp_by_symbol: Dict[str, List[float]] = {s: [] for s in self.symbols}
        amp_seq: List[float] = []
        word_counts: Dict[str,int] = {}
        doc_freq: Dict[str,int] = {}
        bigrams: Dict[str, Dict[str,int]] = {}

        # Determine which files to process
        files_to_process = files
        if incremental_files:
            # Only process changed/new files
            files_to_process = []
            for path in files:
                file_name = os.path.basename(path)
                if file_name in incremental_files:
                    files_to_process.append(path)
            print(f"🔄 Incremental: processing {len(files_to_process)} changed files")

        # JIT Excavation: Prioritize high-value files
        high_value_files = []
        remaining_files = []
        bootstrap_patterns = None
        file_info = []  # Track file info for reporting
        assessment_context = None  # Assessment context for adaptive training

        if enable_jit_excavation and files_to_process:
            print(f"⚡ JIT Excavation: Assessing {len(files_to_process)} files for optimal training order...")
            high_value_files, remaining_files = self._prioritize_files_jit(files_to_process, top_percentage=0.3)

            # Extract bootstrap patterns and create assessment context
            if high_value_files:
                bootstrap_patterns = self._extract_patterns_from_high_value_files(high_value_files)

                # Create assessment context from high-value files
                assessment_context = self._create_assessment_context(high_value_files)

                # Seed initial tokenization with bootstrap patterns
                if bootstrap_patterns:
                    self.learned_tokenizer = set()
                    self.learned_tokenizer.update(bootstrap_patterns['frequent_words'])
                    self.learned_tokenizer.update(bootstrap_patterns['common_phrases'])
                    print(f"🌱 Seeded tokenization with {len(self.learned_tokenizer)} bootstrap patterns")

        # Iterative tokenization-markov excavation
        if files_to_process:
            print(f"🏗️ Starting {excavation_iterations}-iteration excavation process...")

            # Load training texts with JIT prioritization
            training_texts = []

            if high_value_files:
                # Load high-value files first
                print(f"🔥 Priority loading: {len(high_value_files)} high-value files first...")
                for path in self._progress_iter(high_value_files, "loading high-value"):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            training_texts.append(content)
                            file_info.append((os.path.basename(path), 'high_value', len(content)))
                            if time_budget and (time.time() - start_time) > time_budget:
                                print("⏱ stopping early: time budget reached during high-value load")
                                break
                    except:
                        pass

                # Load remaining files
                if remaining_files:
                    print(f"📚 Loading remaining {len(remaining_files)} files...")
                    for path in self._progress_iter(remaining_files, "loading remaining"):
                        try:
                            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                                training_texts.append(content)
                                file_info.append((os.path.basename(path), 'standard', len(content)))
                                if time_budget and (time.time() - start_time) > time_budget:
                                    print("⏱ stopping early: time budget reached during remaining load")
                                    break
                        except:
                            pass
            else:
                # Standard loading without JIT prioritization
                for path in self._progress_iter(files_to_process, "loading"):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='replace') as f:
                            if getattr(self, 'train_sample_chars', None):
                                content = f.read(int(self.train_sample_chars))
                            else:
                                content = f.read()
                            training_texts.append(content)
                            file_info.append((os.path.basename(path), 'standard', len(content)))
                            if time_budget and (time.time() - start_time) > time_budget:
                                print("⏱ stopping early: time budget reached during load")
                                break
                    except:
                        pass

            if training_texts:
                # Iterative excavation: tokenization ↔ markov refinement ↔ color phase clustering
                for iteration in range(excavation_iterations):
                    print(f"\n🏗️ Iteration {iteration + 1}/{excavation_iterations}")
                    if time_budget and (time.time() - start_time) > time_budget:
                        print("⏱ stopping early: time budget reached before iteration work; doing minimal pass")
                        # Minimal pass over a small slice of already-loaded texts so we record some observations
                        mini_texts = training_texts[: min(20, len(training_texts))]
                        for text_content in mini_texts:
                            try:
                                s = self._symbol_from_content(text_content)
                                symbols_seq.append(s)
                                feat = self._content_features(text_content)
                                amp = self._amp_from_features(feat)
                                amp_seq.append(amp)
                                if s in amp_by_symbol:
                                    amp_by_symbol[s].append(amp)

                                # Tokenize a prefix to keep cost bounded
                                toks = self._tok(text_content[:2000])
                                for tok in toks:
                                    if len(tok) >= 1:
                                        word_counts[tok] = word_counts.get(tok, 0) + 1
                                for t in set(toks):
                                    doc_freq[t] = doc_freq.get(t, 0) + 1
                                for a, b in zip(toks[:-1], toks[1:]):
                                    d = bigrams.get(a)
                                    if d is None:
                                        d = {}
                                        bigrams[a] = d
                                    d[b] = d.get(b, 0) + 1
                            except Exception:
                                continue
                        break

                    # Learn/refine tokenization with assessment context
                    self._learn_tokenization_excavation(training_texts, iteration, assessment_context)

                    # Process files with current tokenization
                    print("📊 Processing files with current tokenization...")
                    symbols_seq = []
                    amp_by_symbol = {s: [] for s in self.symbols}
                    amp_seq = []
                    word_counts = {}
                    doc_freq = {}
                    bigrams = {}

                    for path in self._progress_iter(files_to_process, f"excavation-{iteration + 1}"):
                        try:
                            with open(path, 'rb') as f:
                                content = f.read()

                            # Track files
                            if content:
                                file_name = os.path.basename(path)
                                file_size = len(content)
                                file_hash = hashlib.sha256(content).hexdigest()[:16]
                                training_file_info.append([file_name, file_hash, str(file_size)])

                            # Process with current tokenization
                            text_content = content.decode('utf-8', errors='replace')
                            s = self._symbol_from_content(text_content)
                            symbols_seq.append(s)
                            feat = self._content_features(text_content)
                            amp = self._amp_from_features(feat)
                            amp_seq.append(amp)
                            if s in amp_by_symbol:
                                amp_by_symbol[s].append(amp)

                            # Tokenize and analyze with current learned tokenizer
                            toks = self._tok(text_content)
                            for tok in toks:
                                if len(tok) >= 1:
                                    word_counts[tok] = word_counts.get(tok, 0) + 1

                            for t in set(toks):
                                doc_freq[t] = doc_freq.get(t, 0) + 1

                            # Bigram analysis
                            for a, b in zip(toks[:-1], toks[1:]):
                                d = bigrams.get(a)
                                if d is None:
                                    d = {}
                                    bigrams[a] = d
                                d[b] = d.get(b, 0) + 1

                        except Exception as e:
                            continue
                        if time_budget and (time.time() - start_time) > time_budget:
                            print("⏱ stopping early: time budget reached during excavation")
                            break

                    # Update model with this iteration's results
                    if word_counts:
                        self.vocab = [w for w,_ in sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:500]]
                        self.unigram_counts = word_counts
                    if bigrams:
                        self.bigram_counts = bigrams

                    # Build transition matrices for next iteration
                    self.transitions_per_level = self._build_level_kernels()

                    # Refine alphabet based on learned patterns
                    self._refine_alphabet_for_iteration(word_counts, iteration)

                    # Color phase clustering for geometric transformation capability
                    if iteration == excavation_iterations - 1:  # Do this on final iteration
                        print(f"🎨 Analyzing color phases for geometric transformations...")
                        color_samples = self._collect_color_samples_during_training(training_texts)
                        if len(color_samples) >= 8:
                            n_clusters = min(8, len(color_samples) // 10)  # Adaptive cluster count
                            color_phases, cluster_centers = self._analyze_color_phases_clustering(color_samples, n_clusters)

                            if color_phases:
                                self.color_phase_alleles = color_phases
                                self.color_cluster_centers = cluster_centers
                                print(f"   📊 Discovered {len(color_phases)} color phase alleles")
                                for i, phase in enumerate(color_phases):
                                    print(f"      Phase {i}: {phase['sample_count']} samples, "
                                          ".2f"                                          ".2f"
                                          ".2f")
                        else:
                            print(f"   ⚠️  Not enough color samples ({len(color_samples)}) for clustering")

        # Final model update with all iterations' results
        if word_counts:
            self.vocab = [w for w,_ in sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:500]]
            self.unigram_counts = word_counts
        if bigrams:
            self.bigram_counts = bigrams
        if len(symbols_seq) >= 2:
            learnedP = self._learn_transition_matrix(symbols_seq)
            self.priors = (1.0 - blend) * self.priors + blend * learnedP
            self.priors /= self.priors.sum(axis=1, keepdims=True)
            self.transitions_per_level = self._build_level_kernels()
        # learn emission amplitude ranges
        learned_emissions: Dict[str, List[float]] = {}
        all_amp = []
        for s in self.symbols:
            arr = np.array(amp_by_symbol.get(s, []), dtype=float)
            if arr.size == 0:
                lo, hi = 0.4, 1.0
            else:
                lo = float(np.percentile(arr, 25))
                hi = float(np.percentile(arr, 75))
                lo = max(0.2, min(lo, 1.4))
                hi = max(lo + 0.05, min(hi, 1.5))
                all_amp.extend(arr.tolist())
            learned_emissions[s] = [lo, hi]
            if s not in self.emissions:
                self.emissions[s] = {}
            self.emissions[s]['amplitude'] = [lo, hi]
        if all_amp:
            arr = np.array(all_amp, dtype=float)
            std = float(np.std(arr))
            self.alpha = float(np.clip(0.05 + 0.4 * std, 0.05, 0.5))
            self.epsilon0 = float(np.clip(0.5 + 2.0 * std, 0.5, 2.0))
            self.beta = float(np.clip(1.0 / (std + 1e-3), 0.5, 5.0))
        if amp_seq:
            mu = 0.0
            abs_deltas = []
            alpha = self.alpha
            for a in amp_seq:
                mu = (1.0 - alpha) * mu + alpha * a
                abs_deltas.append(abs(a - mu))
            abs_deltas_arr = np.array(abs_deltas, dtype=float)
            self.delta_spawn = float(np.clip(np.percentile(abs_deltas_arr, 85), 0.01, 1.5))
            std_delta = float(np.std(abs_deltas_arr))
            self.kappa = float(np.clip(0.15 + 1.2 * std_delta, 0.05, 1.0))
        learned = {'priors': self.priors, 'emissions': learned_emissions}
        out_path = None
        if write_template:
            # Use the precise training file info instead of directory hash
            training_hash = hashlib.sha256(str(sorted(training_file_info)).encode()).hexdigest()
            out_path = self._write_learned_template(learned, training_hash, self.vocab, self.bigram_counts, self.unigram_counts, training_files=training_file_info)

        # JIT Excavation Summary
        if enable_jit_excavation and high_value_files:
            print(f"\n🎯 JIT Excavation Summary:")
            print(f"   High-value files processed: {len(high_value_files)}")
            print(f"   Remaining files processed: {len(remaining_files)}")
            bootstrap_count = len(self.learned_tokenizer) if self.learned_tokenizer and bootstrap_patterns else 0
            print(f"   Bootstrap patterns seeded: {bootstrap_count}")

            if 'file_info' in locals() and file_info:
                high_value_count = sum(1 for _, ftype, _ in file_info if ftype == 'high_value')
                total_chars = sum(length for _, _, length in file_info)
                print(f"   Training data composition: {high_value_count}/{len(file_info)} high-value files")
                print(f"   Total characters processed: {total_chars:,}")

        timed_out = bool(time_budget and (time.time() - start_time) > time_budget)
        return {
            'files': len(files),
            'observations': len(symbols_seq),
            'priors': self.priors.copy(),
            'emissions': learned_emissions,
            'alpha': self.alpha,
            'epsilon0': self.epsilon0,
            'beta': self.beta,
            'delta_spawn': self.delta_spawn,
            'kappa': self.kappa,
            'template_written': out_path,
            'color_phases_learned': len(self.color_phase_alleles),
            'jit_excavation_enabled': enable_jit_excavation,
            'high_value_files': len(high_value_files) if enable_jit_excavation else 0,
            'timed_out': timed_out,
        }

    def load_learned_if_cached(self, directory='.') -> tuple[bool, Optional[Dict[str, tuple[str, str]]]]:
        learned_path = os.path.join(os.path.dirname(self.template_path), 'fractal_markov_alleles.learned.toml')
        if not os.path.isfile(learned_path):
            return False, None
        try:
            with open(learned_path, 'rb') as f:
                L = toml.load(f)
        except Exception:
            return False, None
        cache_info = L.get('cache') or {}
        mem_expected = cache_info.get('mem_hash')
        files_expected = cache_info.get('files')
        if not mem_expected:
            return False, None

        # per-file validation using precise training files
        if isinstance(files_expected, list):
            changed_files: Dict[str, tuple[str, str]] = {}

            # Check if the exact training files that were used still exist and are unchanged
            for file_info in files_expected:
                if len(file_info) != 3:
                    return False, None  # Invalid cache format

                file_name, expected_hash, expected_size = file_info
                file_path = os.path.join(directory, file_name)

                if not os.path.isfile(file_path):
                    # File was deleted - mark as changed (empty new hash)
                    changed_files[file_name] = (expected_hash, "")
                    continue

                try:
                    with open(file_path, 'rb') as f:
                        current_content = f.read()
                    current_hash = hashlib.sha256(current_content).hexdigest()[:16]
                    current_size = str(len(current_content))

                    # Ensure consistent types for comparison (TOML converts numeric strings to ints)
                    expected_size_str = str(expected_size)
                    size_match = current_size == expected_size_str

                    if current_hash != expected_hash or not size_match:
                        # File changed - track old and new hashes
                        changed_files[file_name] = (expected_hash, current_hash)
                except Exception:
                    return False, None  # Can't read training file

            if changed_files:
                # Some files changed - return info about what changed
                return False, changed_files

            # All training files are unchanged
            return True, None
        else:
            # Fallback to old method if no training files recorded
            mem_current = _hash_directory_mem(directory)
            if mem_expected != mem_current:
                return False, None
        priors_table = L.get('priors') or {}
        self.priors = self._build_prior_from_dict(priors_table)
        em = L.get('emissions') or {}
        for s in self.symbols:
            if s in em and isinstance(em[s], dict) and 'amplitude' in em[s]:
                if s not in self.emissions:
                    self.emissions[s] = {}
                self.emissions[s]['amplitude'] = list(map(float, em[s]['amplitude']))
        sd = L.get('seed') or {}
        self.alpha = float(sd.get('alpha', self.alpha))
        self.epsilon0 = float(sd.get('epsilon0', self.epsilon0))
        self.beta = float(sd.get('beta', self.beta))
        self.kappa = float(sd.get('kappa', self.kappa))
        self.delta_spawn = float(sd.get('delta_spawn', self.delta_spawn))
        bw = sd.get('basis_weights')
        if isinstance(bw, list) and len(bw) >= 3:
            self.basis_weights = np.array([float(bw[0]), float(bw[1]), float(bw[2])], dtype=float)
            self.basis_weights /= max(self.basis_weights.sum(), 1e-9)
        vocab_words = (L.get('vocab') or {}).get('words')
        if isinstance(vocab_words, list):
            self.vocab = [str(w) for w in vocab_words]
        # restore markov
        markov = L.get('markov') or {}
        ug_list = markov.get('unigrams')
        bg_list = markov.get('bigrams')
        if isinstance(ug_list, list):
            self.unigram_counts = {str(k): int(v) for k, v in ug_list}
        if isinstance(bg_list, list):
            big: Dict[str, Dict[str,int]] = {}
            for a, b, c in bg_list:
                d = big.get(a)
                if d is None:
                    d = {}
                    big[a] = d
                d[str(b)] = int(c)
            self.bigram_counts = big

        # Load learned tokenizer if available
        if 'learned_tokenizer' in L.get('cache', {}):
            tokenizer_list = L['cache']['learned_tokenizer']
            if isinstance(tokenizer_list, list):
                self.learned_tokenizer = set(tokenizer_list)
                self.max_token_len = int(L['cache'].get('max_token_len', 3))

        # Load color phase alleles if available
        color_phases = L.get('color_phases', {})
        if color_phases:
            self.color_phase_alleles = []
            for key, phase_dict in color_phases.items():
                if key.startswith('phase_') and isinstance(phase_dict, dict):
                    # Convert string values back to numeric
                    numeric_phase = {}
                    for k, v in phase_dict.items():
                        if isinstance(v, str):
                            try:
                                numeric_phase[k] = float(v)
                            except ValueError:
                                numeric_phase[k] = v
                        else:
                            numeric_phase[k] = v
                    self.color_phase_alleles.append(numeric_phase)
            print(f"🎨 Loaded {len(self.color_phase_alleles)} color phase alleles")

        self.transitions_per_level = self._build_level_kernels()
        return True, None

    @staticmethod
    def _file_hashes(directory: str) -> List[List[str]]:
        out: List[List[str]] = []
        for name in sorted(os.listdir(directory)):
            if name.startswith('.'):
                continue
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                h = hashlib.sha256(data).hexdigest()[:16]
                out.append([name, h, str(len(data))])
            except Exception:
                continue
        return out


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


def main():
    parser = argparse.ArgumentParser(prog='ce1_fractal_markov_engine', add_help=True)
    parser.add_argument('seed', nargs='?', default=None, help='prompt/seed text (positional)')
    parser.add_argument('-g', '--gen', type=int, default=None, help='algebra TEXT tokens')
    parser.add_argument('-p', '--pheno', type=int, default=None, help='phenotype sentences')
    parser.add_argument('-m', '--markov', type=int, default=None, help='markov text length')
    parser.add_argument('-L', '--listen', default=True,action='store_true', help='interpret phenotype as program updates')
    parser.add_argument('-R', '--retrain', action='store_true', help='force retraining')
    parser.add_argument('-N', '--no-train', action='store_true', help='skip training (use cache/defaults)')
    parser.add_argument('-t', '--tags', default='P', help='output tags: L(levels) e(emit) q(quat) M(markov) P(pheno) c(color) H(hybrid)')
    parser.add_argument('--color-phases', action='store_true', help='show color phase transformation demonstration')
    parser.add_argument('--full', action='store_true', help='do a full multi-iteration train (overrides time budget)')
    parser.add_argument('--profile', action='store_true', help='profile training and print top hotspots (cProfile)')
    parser.add_argument('--profile-secs', type=float, default=None, help='statistical wall-time sampling for N seconds')
    parser.add_argument('--max-secs', type=float, default=10.0, help='stop training after this many seconds (wall clock)')
    args = parser.parse_args()

    # Speed knob env: CE1_FAST=1 implies bounded defaults already
    fast_env = os.environ.get('CE1_FAST', '')
    FAST = fast_env not in ('', '0', 'false', 'False')
    QUICK = bool(FAST)

    tpl = os.path.join(os.path.dirname(__file__), 'templates', 'fractal_markov_alleles.toml')
    engine = FractalMarkovAlleleEngine(tpl)
    # Optional wall-clock budget
    max_secs_env = os.environ.get('CE1_MAX_SECS')
    if args.max_secs or max_secs_env:
        try:
            engine.max_secs = float(args.max_secs or max_secs_env)
        except Exception:
            engine.max_secs = None
    if QUICK:
        # Reduce runtime cost when CE1_FAST is set
        engine.planned_steps = max(4, min(engine.planned_steps, 8))
    in_dir = os.path.dirname(__file__)

    # load cache or handle cache miss
    used_cache = False
    changed_files = None
    if not args.retrain:
        used_cache, changed_files = engine.load_learned_if_cached(in_dir)
        if used_cache:
            print('✓ using learned template cache (mem hash match)')
        else:
            if changed_files:
                print(f'⚠ cache miss - {len(changed_files)} training files changed')
                if not args.no_train:
                    print('  → incremental retraining available')
                else:
                    print('  → skipping training (--no-train specified)')
            else:
                print('⚠ cache miss - learned template is stale')
                if not args.no_train:
                    print('  → use --retrain to update the cache')
                else:
                    print('  → skipping training (--no-train specified)')

    # Train if explicitly requested or if files changed and training not disabled
    # Train only when cache is stale and training is allowed, or on explicit retrain/full
    should_train = (args.retrain or args.full or ((not args.no_train) and (not used_cache)))
    if should_train:
        if changed_files:
            print(f"🔄 incremental training {len(changed_files)} changed files...")
        else:
            print("🔄 training fractal markov model...")

        if args.profile or args.profile_secs:
            import cProfile, pstats, io
            pr = None
            sampler_thread = None
            sampler_counts = None
            if args.profile:
                pr = cProfile.Profile(); pr.enable()
            if args.profile_secs:
                import threading, time, sys
                from collections import Counter
                main_ident = threading.main_thread().ident
                sampler_counts = Counter(); stop_flag = {'stop': False}
                duration = float(max(0.1, args.profile_secs)); interval = 0.01
                def pick_frame(fr):
                    stack = []
                    while fr:
                        stack.append(fr); fr = fr.f_back
                    for f in reversed(stack):
                        fn = f.f_code.co_filename
                        if fn and fn.endswith('.py'):
                            return f
                    return stack[-1] if stack else None
                def sample_loop():
                    end = time.time() + duration
                    while time.time() < end and not stop_flag['stop']:
                        fr = sys._current_frames().get(main_ident)
                        if fr:
                            f = pick_frame(fr)
                            if f:
                                key = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}"
                                sampler_counts[key] += 1
                        time.sleep(interval)
                sampler_thread = threading.Thread(target=sample_loop, daemon=True); sampler_thread.start()

            # Bounded by default unless --full is set
            if args.full:
                info = engine.train_from_directory(in_dir, incremental_files=changed_files)
            else:
                # One iteration with wall-clock budget
                engine.max_secs = float(args.max_secs or 10.0)
                engine.assess_sample_chars = 20000
                engine.train_sample_chars = 20000
                info = engine.train_from_directory(in_dir, incremental_files=changed_files, excavation_iterations=1)

            if pr:
                pr.disable(); s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats('cumtime'); ps.print_stats(30)
                print("\nPROFILE (top 30 by cumtime):\n" + s.getvalue())
            if sampler_thread:
                stop_flag['stop'] = True; sampler_thread.join()
                total = sum(sampler_counts.values()) or 1
                top = sorted(sampler_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
                print("\nSAMPLE PROFILE (wall-time, top 20):")
                for k, c in top:
                    pct = 100.0 * c / total
                    print(f"  {k}  {c} samples  {pct:.1f}%")
        else:
            if args.full:
                info = engine.train_from_directory(in_dir, incremental_files=changed_files)
            else:
                engine.max_secs = float(args.max_secs or 10.0)
                engine.assess_sample_chars = 20000
                engine.train_sample_chars = 20000
                info = engine.train_from_directory(in_dir, incremental_files=changed_files, excavation_iterations=1)
        print(f"✅ trained on {info['observations']} observations from {info['files']} files in .in")

    # run kernel
    out = engine.run(steps=engine.planned_steps)
    banks, finals, color_trail = out['banks'], out['final_q'], out['color_trail']

    # seed/prompt
    seed_text = args.seed or ''
    changes: Optional[Dict[str, float]] = None  # Initialize changes variable

    # optional listen cycle
    if args.listen and seed_text:
        if changes and isinstance(changes, dict):
            print('\nAPPLIED:')
            for k, v in changes.items():
                print(f'  {k} -> {v:.4f}')
        out = engine.run(steps=engine.planned_steps)
        banks, finals, _ = out['banks'], out['final_q'], out['color_trail']

    # outputs controlled by tags
    tags = set(args.tags)

    if 'M' in tags and seed_text:
        mtxt = engine.generate_text(seed_text, style="markov", length=args.markov or 60)
        print('\nMARKOV:')
        print(mtxt)

    if 'H' in tags and seed_text:
        htxt = engine.generate_text(seed_text, banks, finals, style="hybrid", length=150)
        print('\nHYBRID:')
        print(htxt)

    changes = None
    if 'P' in tags and seed_text:
        ptxt = engine.generate_text(seed_text, banks, finals, style="pheno", length=(args.pheno or 3)*20)
        changes = args.listen and engine.interpret_pheno_and_apply(ptxt)
        print('\nPHENO:')
        print(ptxt)
        
    if changes:
        print('\nAPPLIED:')
        for k, v in changes.items():
            print(f'  {k} -> {v:.4f}')
        out = engine.run(steps=engine.planned_steps)
        banks, finals, _ = out['banks'], out['final_q'], out['color_trail']

    if 'M' in tags and seed_text:
        mtxt = engine.generate_text(seed_text, style="markov", length=args.markov or 60)
        print('\nMARKOV:')
        print(mtxt)

    if 'q' in tags:
        q_final = finals[-1] if finals else np.array([1.0, 0.0, 0.0, 0.0])
        print(f"\nq_final={engine._format_q(q_final)}")

    if 'L' in tags:
        print('\nLEVELS:')
        for k, alleles in enumerate(banks):
            if alleles:
                glyphs = [engine._format_allele_as_glyph(q) for q in alleles[:5]]  # Show first 5
                glyph_str = ''.join(glyphs)
                print(f'L{k}={len(alleles)} [{glyph_str}]')
            else:
                print(f'L{k}={len(alleles)} []')

    if 'c' in tags and color_trail:
        print('\nCE1 Color Trail:')
        for k, level_colors in enumerate(color_trail):
            print(f'L{k}: {level_colors[:3]}...' if len(level_colors) > 3 else f'L{k}: {level_colors}')

    if 'e' in tags:
        D = engine.levels
        q_final = finals[-1] if finals else np.array([1.0, 0.0, 0.0, 0.0])
        mem_hash = _hash_directory_mem(in_dir)
        m = int(mem_hash[:8], 16) % (2**D)
        passport = hashlib.sha256(mem_hash.encode()).hexdigest()[:16]
        print('\nCE1c emit:')
        print(f'  passport={passport} ; D={D} ; q_final=[{q_final[0]:.3f},{q_final[1]:.3f},{q_final[2]:.3f},{q_final[3]:.3f}] ; mode={engine.mode}')
        print(f'  m(mod 2^D)={m}')

    # Color phase demonstration
    if args.color_phases:
        demo = engine.demonstrate_color_phase_transformation()
        print(f'\n{demo}')

        # Show color phase statistics
        phase_stats = engine.get_color_phase_statistics()
        if phase_stats['total_phases'] > 0:
            print(f"\n📊 Color Phase Statistics:")
            print(f"   Total phases learned: {phase_stats['total_phases']}")
            for phase in phase_stats['phases'][:3]:  # Show first 3
                print(f"   Phase {phase['id']}: {phase['sample_count']} samples, "
                      ".1%")

    # Cache is already saved by training method with precise file tracking


if __name__ == '__main__':
    main()