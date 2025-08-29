from dataclasses import dataclass, field, asdict
import os
import dataclasses
from typing import Dict, List, Set, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import pickle
import msgpack
import zipfile
import io
from collections.abc import Buffer
from deflate import extract_zip_deflate_streams
from aspire import Opix, stable_str_hash, stable_bytes_hash
import re

import quaternion


def grammar_healer_hook(pairs):
    """
    A msgpack hook that intelligently converts dictionary-like structures into
    the appropriate Counter or defaultdict types for the grammar, and heals
    old list-based keys into hashable tuples.
    """
    healed_pairs = []
    for key, value in pairs:
        match key:
            case list() as l:
                healed_pairs.append((tuple(l), value))
            case _:
                healed_pairs.append((key, value))

    # Heuristic for bigram_counts outer dict: keys are tuples
    if healed_pairs and all(isinstance(k, tuple) for k, v in healed_pairs):
        return defaultdict(Counter, {k: Counter(v) for k, v in healed_pairs})

    # Heuristic for Counter objects: values are numbers
    if healed_pairs and all(isinstance(v, (int, float)) for k, v in healed_pairs):
        return Counter(dict(healed_pairs))

    return dict(healed_pairs)


# --- Configuration Dataclasses ---

@dataclass
class SeedConfig:
    alpha: float = 0.1
    epsilon0: float = 1.0
    beta: float = 1.0
    kappa: float = 0.25
    delta_spawn: float = 0.05

@dataclass
class FractalConfig:
    self_similarity: float = 0.6
    scales: Dict[str, float] = field(default_factory=dict)

@dataclass
class EngineConfig:
    levels: int = 4
    symbols: List[str] = field(default_factory=lambda: ["carry", "borrow", "drift"])
    priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    seed: SeedConfig = field(default_factory=SeedConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    stochastic_seed: int = 1337

# --- Learned Grammar ---

@dataclass
class Grammar:
    unigram_counts: Counter = field(default_factory=Counter)
    bigram_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    byte_classes: Dict[int, int] = field(default_factory=dict)
    semantic_alleles: Dict[str, np.ndarray] = field(default_factory=dict)


# --- Genome: The Complete State of a Learner ---

@dataclass
class Genome:
    config: EngineConfig
    grammar: Grammar
    extra_assets: Dict[str, bytes] = field(default_factory=dict)
    compressed_assets: Dict[str, Dict] = field(default_factory=dict)
    alerts: Opix = field(default_factory=Opix)
    # Registry keys (lowercase, .mpk for msgpack)
    reg_cfg: str = 'registry/config.mpk'
    reg_gram: str = 'registry/grammar.mpk'
    reg_opx: str = 'registry/opx.mpk'
    reg_opt: str = 'registry/opt.mpk'
    reg_sym: str = 'registry/sym.mpk'
    reg_wor: str = 'registry/worlds.mpk'

    def save(self, path: str):
        """Saves the genome to a zip archive."""
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Write all assets first
            for name, data in self.extra_assets.items():
                if isinstance(data, Buffer):
                    zf.writestr(name, data)

            # Now, explicitly write the canonical config and grammar
            config_dict = asdict(self.config)
            zf.writestr(self.reg_cfg, msgpack.packb(
                config_dict, use_bin_type=True) or b'')

            grammar_dict = dataclasses.asdict(self.grammar)
            zf.writestr(self.reg_gram, msgpack.packb(
                grammar_dict, use_bin_type=True) or b'')

    @classmethod
    def from_path(cls, path: str) -> 'Genome':
        """Loads a genome from a zip archive."""
        all_assets, compressed_assets = extract_zip_deflate_streams(path)
        alerts = Opix()
        config = EngineConfig()
        grammar = Grammar()

        # Config: Try to load from canonical path, otherwise use default.
        try:
            config_bytes = all_assets.pop(cls.reg_cfg)
            config_dict = msgpack.unpackb(config_bytes, raw=False)
            seed_config = SeedConfig(**config_dict.get('seed', {}))
            fractal_config = FractalConfig(**config_dict.get('fractal', {}))
            config = EngineConfig(
                levels=config_dict.get('levels', 4),
                symbols=config_dict.get(
                    'symbols', ["carry", "borrow", "drift"]),
                priors=config_dict.get('priors', {}),
                seed=seed_config,
                fractal=fractal_config,
                stochastic_seed=config_dict.get('stochastic_seed', 1337)
            )
        except (KeyError, ValueError, msgpack.ExtraData, msgpack.FormatError):
            alerts['C!'] += 1  # Signal missing or corrupt config

        # Grammar: Try to load from canonical path, otherwise use default.
        try:
            grammar_bytes = all_assets.pop(cls.reg_gram)
            grammar_dict = msgpack.unpackb(
                grammar_bytes, raw=False, object_pairs_hook=grammar_healer_hook)
            if not isinstance(grammar_dict, dict):
                raise ValueError('Grammar file is not a valid dictionary.')

            ug = grammar_dict.get('unigram_counts', Counter())
            bg = grammar_dict.get('bigram_counts', defaultdict(Counter))
            bc = grammar_dict.get('byte_classes', {})
            grammar = Grammar(
                unigram_counts=ug,
                bigram_counts=bg,
                byte_classes={int(k): v for k, v in bc.items()}
            )
        except (KeyError, ValueError, msgpack.ExtraData, msgpack.FormatError):
            alerts['Γ'] += 1  # Signal missing or corrupt grammar

        # Clean up any remaining legacy paths to create a clean asset list
        legacy_paths = ['config.msgpack', 'grammar.msgpack',
                        'registry/config.msgpack', 'registry/grammar.msgpack']
        for p in legacy_paths:
            all_assets.pop(p, None)

        extra_assets = all_assets

        return cls(config=config, grammar=grammar, extra_assets=extra_assets, compressed_assets=compressed_assets, alerts=alerts)

    # --- Aspirate carry (module-owned shape) ---
    def carry(self) -> Opix:
        """Genome-owned compact carry: assets and alerts folded into an Opix."""
        c = Opix()

        # Count non-meta assets
        asset_names = [
            n for n in self.extra_assets.keys()
            if not n.startswith('registry/')
        ]
        c['§'] += len(asset_names)

        # Grammar footprint and signature
        ug = self.grammar.unigram_counts
        bg = self.grammar.bigram_counts
        alleles = self.grammar.semantic_alleles

        c['μ'] += sum(ug.values())
        c['ν'] += len(ug)
        c['β'] += sum(len(row) for row in bg.values())
        c['ψ'] += len(alleles)

        # Lightweight signature over top-32 unigrams
        top = sorted(ug.items(), key=lambda kv: kv[1], reverse=True)[:32]
        sig_src = ''.join(f"{k}:{v}," for k, v in top)
        sig = stable_str_hash(sig_src) if top else 0
        sig_hex = f"{sig & 0xFFFFFFFF:08x}"
        c[f"◇{sig_hex}"] += 1

        # Overlay existing alerts
        c = c.overlay_with(self.alerts)

        return c

    # --- Shallow grammar builder (fallback when grammar is absent/empty) ---
    def build_shallow_grammar(self, max_bytes_per_asset: int = 32768) -> None:
        """Builds a minimal grammar from asset contents and names (shallow)."""
        word_re = re.compile(r"[A-Za-z0-9_]+")
        unig = self.grammar.unigram_counts
        bigr = self.grammar.bigram_counts
        # From file contents
        for name, data in self.extra_assets.items():
            if not data:
                continue
            try:
                text = data[:max_bytes_per_asset].decode(
                    'utf-8', errors='ignore')
            except Exception:
                continue
            tokens = word_re.findall(text)
            prev = None
            for tok in tokens:
                unig[tok] += 1
                if prev is not None:
                    bigr[prev][tok] += 1
                prev = tok
        # From file names (path parts and stems)
        for name in self.extra_assets.keys():
            parts = re.split(r"[/\\]+", name)
            for p in parts:
                stem = re.sub(r"\.[^.]+$", "", p)
                if stem:
                    unig[stem] += 1
        # Ensure defaults if still empty
        if not unig:
            unig.update({"carry": 1, "borrow": 1, "drift": 1})

    # --- Registry helpers (store under 'registry/' in ZIP) ---
    def registry_list(self, prefix: str = 'registry/') -> List[str]:
        return sorted([k for k in self.extra_assets.keys() if k.startswith(prefix)])

    def registry_get(self, name: str, *, text: bool = False, encoding: str = 'utf-8') -> Optional[Any]:
        key = name if name.startswith('registry/') else f'registry/{name}'
        blob = self.extra_assets.get(key)
        if blob is None:
            return None
        if text:
            try:
                return blob.decode(encoding)
            except Exception:
                return None
        return blob

    def registry_set(self, name: str, data: Any, *, encoding: str = 'utf-8') -> None:
        key = name if name.startswith('registry/') else f'registry/{name}'
        if isinstance(data, str):
            payload = data.encode(encoding)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            payload = bytes(data)
        else:
            raise TypeError('registry_set expects str or bytes-like data')
        self.extra_assets[key] = payload

    def registry_delete(self, name: str) -> bool:
        key = name if name.startswith('registry/') else f'registry/{name}'
        return self.extra_assets.pop(key, None) is not None

    @staticmethod
    def _create_byte_sphere() -> np.ndarray:
        """Creates 256 points distributed evenly on a unit sphere using Fibonacci lattice."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
        for i in range(256):
            y = 1 - (i / 255.0) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    @staticmethod
    def _create_perceptual_archetypes(n_classes: int) -> np.ndarray:
        """Creates n_classes archetype vectors. For n=8, uses corners of a cube."""
        if n_classes == 8:
            # Normalized corners of a cube
            s = 1.0 / np.sqrt(3)
            return np.array([
                [s, s, s], [-s, s, s], [s, -s, s], [s, s, -s],
                [-s, -s, s], [-s, s, -s], [s, -s, -s], [-s, -s, -s]
            ])
        else: # Fallback for other numbers of classes
            points = []
            phi = np.pi * (3. - np.sqrt(5.))
            for i in range(n_classes):
                y = 1 - (i / float(n_classes - 1)) * 2
                radius = np.sqrt(1 - y * y)
                theta = phi * i
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                points.append([x, y, z])
            return np.array(points)

    @staticmethod
    def _apply_quat_rotation(points: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotates a set of 3D points by a quaternion."""
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        # Represent points as pure quaternions
        points_quat = np.zeros((points.shape[0], 4))
        points_quat[:, 1:] = points
        
        # Apply rotation: p' = q * p * q_conj
        rotated_points = np.zeros_like(points_quat)
        for i, p in enumerate(points_quat):
            # manual quaternion multiplication: q * p
            qp = np.array([
                q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
                q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
                q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1],
                q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
            ])
            # (q * p) * q_conj
            qp_qconj = np.array([
                qp[0]*q_conj[0] - qp[1]*q_conj[1] - qp[2]*q_conj[2] - qp[3]*q_conj[3],
                qp[0]*q_conj[1] + qp[1]*q_conj[0] + qp[2]*q_conj[3] - qp[3]*q_conj[2],
                qp[0]*q_conj[2] - qp[1]*q_conj[3] + qp[2]*q_conj[0] + qp[3]*q_conj[1],
                qp[0]*q_conj[3] + qp[1]*q_conj[2] - qp[2]*q_conj[1] + qp[3]*q_conj[0]
            ])
            rotated_points[i] = qp_qconj
            
        return rotated_points[:, 1:]

    @staticmethod
    def allele_to_byte_classes(q: np.ndarray, n_classes: int = 8) -> Dict[int, int]:
        """Generates a byte class mapping from a single quaternion (allele)."""
        byte_sphere = Genome._create_byte_sphere()
        archetypes = Genome._create_perceptual_archetypes(n_classes)
        
        # The allele rotates the entire perceptual space
        rotated_sphere = Genome._apply_quat_rotation(byte_sphere, q)
        
        # Classify each byte by finding the closest archetype in the new orientation
        byte_classes = {}
        for i, byte_vec in enumerate(rotated_sphere):
            # Compute dot products to find the archetype with the highest alignment
            alignments = np.dot(archetypes, byte_vec)
            best_class = np.argmax(alignments)
            byte_classes[i] = int(best_class)
            
        return byte_classes

    @staticmethod
    def blend(g1: 'Genome', g2: 'Genome', factor: float) -> 'Genome':
        """Creates a new Genome by blending two parent genomes."""
        factor = np.clip(factor, 0.0, 1.0)
        
        # --- Blend EngineConfig ---
        c1, c2 = g1.config, g2.config
        
        # Blend SeedConfig
        s1, s2 = c1.seed, c2.seed
        new_seed = SeedConfig(
            alpha=(1-factor)*s1.alpha + factor*s2.alpha,
            epsilon0=(1-factor)*s1.epsilon0 + factor*s2.epsilon0,
            beta=(1-factor)*s1.beta + factor*s2.beta,
            kappa=(1-factor)*s1.kappa + factor*s2.kappa,
            delta_spawn=(1-factor)*s1.delta_spawn + factor*s2.delta_spawn,
        )

        # Blend FractalConfig
        f1, f2 = c1.fractal, c2.fractal
        new_scales = {k: (1-factor)*f1.scales.get(k,0) + factor*f2.scales.get(k,0) for k in set(f1.scales) | set(f2.scales)}
        new_fractal = FractalConfig(
            self_similarity=(1-factor)*f1.self_similarity + factor*f2.self_similarity,
            scales=new_scales
        )

        # Blend Priors
        priors1 = c1.priors; priors2 = c2.priors
        symbols = sorted(list(set(c1.symbols) | set(c2.symbols)))
        new_priors = {}
        for s_from in symbols:
            new_priors[s_from] = {}
            row1 = priors1.get(s_from, {})
            row2 = priors2.get(s_from, {})
            for s_to in symbols:
                p1 = row1.get(s_to, 0.0); p2 = row2.get(s_to, 0.0)
                new_priors[s_from][s_to] = (1-factor)*p1 + factor*p2
        
        new_config = EngineConfig(
            levels=int((1-factor)*c1.levels + factor*c2.levels),
            symbols=symbols,
            priors=new_priors,
            seed=new_seed,
            fractal=new_fractal,
            stochastic_seed=int((1-factor)*c1.stochastic_seed + factor*c2.stochastic_seed)
        )

        # --- Blend Grammar ---
        gram1, gram2 = g1.grammar, g2.grammar
        
        # Blend unigrams
        new_unigrams = Counter()
        for k in set(gram1.unigram_counts) | set(gram2.unigram_counts):
            c1 = gram1.unigram_counts.get(k, 0); c2 = gram2.unigram_counts.get(k, 0)
            new_unigrams[k] = int((1-factor)*c1 + factor*c2)

        # Blend bigrams
        new_bigrams = defaultdict(Counter)
        for t1 in set(gram1.bigram_counts) | set(gram2.bigram_counts):
            for t2 in set(gram1.bigram_counts.get(t1, {})) | set(gram2.bigram_counts.get(t1, {})):
                c1 = gram1.bigram_counts.get(t1, {}).get(t2, 0)
                c2 = gram2.bigram_counts.get(t1, {}).get(t2, 0)
                new_bigrams[t1][t2] = int((1-factor)*c1 + factor*c2)
        
        # For byte classes, inherit from one parent (simplest approach)
        new_byte_classes = gram1.byte_classes if factor < 0.5 else gram2.byte_classes

        new_grammar = Grammar(
            unigram_counts=new_unigrams,
            bigram_counts=new_bigrams,
            byte_classes=new_byte_classes
        )

        # For extra assets, just inherit from the first parent for simplicity
        new_extra_assets = g1.extra_assets.copy()

        return Genome(config=new_config, grammar=new_grammar, extra_assets=new_extra_assets)
