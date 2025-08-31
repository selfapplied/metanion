from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict, namedtuple
import msgpack
import zipfile
from collections.abc import Buffer
from zip import extract_zip_deflate_streams, ZipFlexManifest, FileOp, blit_to_reflex_bytes
from aspire import Opix, stable_str_hash, apply_ops, build_ops_from_specs
import re
import numpy as np


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

    # Prefer explicit kind-tags if present
    try:
        kinds = [v for k, v in healed_pairs if k == '__kind__']
        if kinds:
            kind = kinds[0]
            if kind == 'bigrams':
                rows = []
                for k, v in healed_pairs:
                    if k == 'rows' and isinstance(v, list):
                        rows = v
                        break
                out = defaultdict(Counter)
                for entry in rows:
                    try:
                        k, row = entry
                        out[k] = Counter(row)
                    except Exception:
                        continue
                return out
    except Exception:
        pass

    # Heuristic for bigram_counts outer dict: keys are tuples
    if healed_pairs and all(isinstance(k, tuple) for k, v in healed_pairs):
        return defaultdict(Counter, {k: Counter(v) for k, v in healed_pairs})

    # Heuristic for Counter objects: values are numbers
    if healed_pairs and all(isinstance(v, (int, float)) for k, v in healed_pairs):
        return Counter(dict(healed_pairs))

    return dict(healed_pairs)


# --- Configuration Converted to NamedTuples ---
SeedConfig = namedtuple('SeedConfig', 'alpha epsilon0 beta kappa delta_spawn', defaults=(
    0.1, 1.0, 1.0, 0.25, 0.05))
FractalConfig = namedtuple(
    'FractalConfig', 'self_similarity scales', defaults=(0.6, {}))
EngineConfig = namedtuple('EngineConfig', 'levels symbols priors seed fractal stochastic_seed', defaults=(
    4, ["carry", "borrow", "drift"], {}, SeedConfig(), FractalConfig(), 1337))

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

    def to_manifest(self) -> ZipFlexManifest:
        """Creates a ZipFlexManifest representing the entire genome."""
        ops = []
        # Add all extra assets to the manifest
        msgpack_bytes = msgpack.packb(self.extra_assets, use_bin_type=True) or b''
        # for name, data in self.extra_assets.items():
        #     payload = data if isinstance(
        #         data, bytes) else str(data).encode('utf-8')
        #     ops.append(FileOp(name, payload))

        # # Serialize and add the config
        # ops.append(FileOp(self.reg_cfg, msgpack.packb(
        #     self.config, use_bin_type=True) or b''))

        # # Serialize and add the grammar
        # ug = dict(self.grammar.unigram_counts)
        # rows = [[k, dict(v)] for k, v in self.grammar.bigram_counts.items()]
        # bg = {'__kind__': 'bigrams', 'rows': rows}
        # gram_obj = {
        #     'unigram_counts': ug,
        #     'bigram_counts': bg,
        #     'byte_classes': self.grammar.byte_classes,
        # }
        ops.append(FileOp(self.reg_gram, msgpack_bytes))
        return ZipFlexManifest(ops, comment=b'Eonyx Genome')

    def save(self, path: str):
        """Saves the genome to a Reflex archive using the manifest."""
        from pathlib import Path

        manifest = self.to_manifest()
        data = blit_to_reflex_bytes(manifest)
        Path(path).write_bytes(data)

    @classmethod
    def from_path(cls, path: str) -> 'Genome':
        """Loads a genome from a zip archive, signaling errors via alerts."""
        alerts = Opix()
        all_assets, compressed_assets = extract_zip_deflate_streams(
            path, alerts)

        # --- Config Loading ---
        config_bytes = all_assets.pop(cls.reg_cfg, None)
        if config_bytes:
            config_dict = None
            try:
                config_dict = msgpack.unpackb(config_bytes, raw=False)
            except (msgpack.ExtraData, msgpack.FormatError, msgpack.UnpackValueError):
                alerts['C!'] += 1  # Config corrupt

            if isinstance(config_dict, dict):
                seed_conf = SeedConfig(**config_dict.get('seed', {}))
                fractal_conf = FractalConfig(**config_dict.get('fractal', {}))
                config = EngineConfig(
                    levels=config_dict.get('levels', 4),
                    symbols=config_dict.get(
                        'symbols', ["carry", "borrow", "drift"]),
                    priors=config_dict.get('priors', {}),
                    seed=seed_conf,
                    fractal=fractal_conf,
                    stochastic_seed=config_dict.get('stochastic_seed', 1337)
                )
            else:
                config = EngineConfig()
                if config_dict is not None:
                    alerts['C?'] += 1  # Config is wrong type
        else:
            alerts['C∅'] += 1  # Config not found
            config = EngineConfig()

        # --- Grammar Loading ---
        grammar_bytes = all_assets.pop(cls.reg_gram, None)
        if grammar_bytes:
            grammar_dict = None
            try:
                grammar_dict = msgpack.unpackb(
                    grammar_bytes, raw=False, object_pairs_hook=grammar_healer_hook)
            except (msgpack.ExtraData, msgpack.FormatError, msgpack.UnpackValueError):
                alerts['Γ!'] += 1  # Grammar corrupt

            if isinstance(grammar_dict, dict):
                grammar = Grammar(
                    unigram_counts=grammar_dict.get(
                        'unigram_counts', Counter()),
                    bigram_counts=grammar_dict.get(
                        'bigram_counts', defaultdict(Counter)),
                    byte_classes={int(k): v for k, v in grammar_dict.get(
                        'byte_classes', {}).items()}
                )
            else:
                grammar = Grammar()
                if grammar_dict is not None:
                    alerts['Γ?'] += 1  # Grammar is wrong type
        else:
            alerts['Γ∅'] += 1  # Grammar not found
            grammar = Grammar()

        # Clean up any remaining legacy paths to create a clean asset list
        legacy_paths = ['config.msgpack', 'grammar.msgpack',
                        'registry/config.msgpack', 'registry/grammar.msgpack']
        for p in legacy_paths:
            all_assets.pop(p, None)

        return cls(config=config, grammar=grammar, extra_assets=all_assets, compressed_assets=compressed_assets, alerts=alerts)

    # --- Aspirate carry (compact status) ---
    def carry(self) -> Opix:
        c = Opix()
        c['§'] += sum(1 for n in self.extra_assets if not n.startswith('registry/'))
        ug = self.grammar.unigram_counts
        bg = self.grammar.bigram_counts
        alleles = self.grammar.semantic_alleles
        c['μ'] += sum(ug.values())
        c['ν'] += len(ug)
        c['β'] += sum(len(row) for row in bg.values())
        c['ψ'] += len(alleles)
        top = sorted(ug.items(), key=lambda kv: kv[1], reverse=True)[:32]
        sig = stable_str_hash(
            ''.join(f"{k}:{v}," for k, v in top)) if top else 0
        c[f"◇{sig & 0xFFFFFFFF:08x}"] += 1
        return c.overlay_with(self.alerts)

    def build_shallow_grammar(self, max_bytes_per_asset: int = 32768) -> None:
        word_re = re.compile(r"[A-Za-z0-9_]+")
        unig = self.grammar.unigram_counts
        bigr = self.grammar.bigram_counts
        for _, data in self.extra_assets.items():
            if not data:
                continue
            try:
                text = data[:max_bytes_per_asset].decode(
                    'utf-8', errors='ignore')
            except Exception:
                continue
            prev = None
            for tok in word_re.findall(text):
                unig[tok] += 1
                if prev is not None:
                    bigr[prev][tok] += 1
                prev = tok
        for name in self.extra_assets.keys():
            for p in re.split(r"[/\\]+", name):
                stem = re.sub(r"\.[^.]+$", "", p)
                if stem:
                    unig[stem] += 1
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

    def apply_viral_ops(self) -> None:
        """Load and apply minimal Opix ops from registry/opx.mpk to grammar counts."""
        try:
            blob = self.registry_get(self.reg_opx)
            if not isinstance(blob, (bytes, bytearray)):
                return
            obj = msgpack.unpackb(bytes(blob), raw=False)
            if not isinstance(obj, dict):
                return
            uni_specs = obj.get('unigram_ops', [])
            bi_specs = obj.get('bigram_ops', [])
            if uni_specs:
                ops = build_ops_from_specs(uni_specs)
                self.grammar.unigram_counts = apply_ops(
                    self.grammar.unigram_counts, ops)
            if bi_specs:
                opsb = build_ops_from_specs(bi_specs)
                bg_new = defaultdict(Counter)
                for k, row in self.grammar.bigram_counts.items():
                    bg_new[k] = apply_ops(row, opsb)
                self.grammar.bigram_counts = bg_new
        except Exception:
            pass

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
        """Vectorized rotation via 3x3 matrix derived from quaternion q=[w,x,y,z]."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])
        return points @ R.T

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

        # --- Blend Grammar (renormalized largest-remainder) ---
        def _renorm_merge(a: Counter, b: Counter, t: float) -> Counter:
            if not a and not b:
                return Counter()
            w = 1.0 - t
            keys = set(a.keys()) | set(b.keys())
            # target total mass
            target = int(round(w * sum(a.values()) + t * sum(b.values())))
            vals = {k: (w * float(a.get(k, 0)) + t * float(b.get(k, 0)))
                    for k in keys}
            s = float(sum(vals.values()))
            if s <= 0.0 or target <= 0:
                return Counter()
            scale = target / s
            floors: Dict[str, int] = {}
            fracs = []
            acc = 0
            for k, v in vals.items():
                x = v * scale
                f = int(np.floor(x + 1e-12))
                floors[k] = f
                acc += f
                fracs.append((x - f, k))
            out = Counter(floors)
            rem = max(0, target - acc)
            fracs.sort(reverse=True)
            for i in range(rem):
                if i < len(fracs):
                    out[fracs[i][1]] += 1
            # prune zeros
            for k in list(out.keys()):
                if out[k] <= 0:
                    del out[k]
            return out

        gram1, gram2 = g1.grammar, g2.grammar
        new_unigrams = _renorm_merge(
            gram1.unigram_counts, gram2.unigram_counts, float(factor))

        new_bigrams = defaultdict(Counter)
        all_src = set(gram1.bigram_counts.keys()) | set(
            gram2.bigram_counts.keys())
        for t1 in all_src:
            row1 = gram1.bigram_counts.get(t1, Counter())
            row2 = gram2.bigram_counts.get(t1, Counter())
            merged = _renorm_merge(row1, row2, float(factor))
            if merged:
                new_bigrams[t1] = merged
        
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
