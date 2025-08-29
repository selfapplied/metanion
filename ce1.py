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

from aspire import Opix, stable_str_hash
from genome import Genome, SeedConfig, FractalConfig, EngineConfig
import quaternion
from resonance import BlockEvent, get_rotation_from_block
from deflate import BlockProcessor

# --- Function-table bitmask (inputs) ---
IN_NAME  = 1 << 0
IN_BYTES = 1 << 1
IN_TEXT  = 1 << 2

# --- Simple Logger ---
logger = logging.getLogger("ce1_core")
logger.setLevel(logging.WARNING)

# Simple console handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Branch State (Branch-cuts over grammar space) ---
@dataclass
class BranchState:
    sheet: int = 0
    cuts: int = 0
    winding: int = 0
    progress: float = 0.0

# --- Block Processing Results ---
@dataclass
class BlockResult:
    final_q: np.ndarray
    final_qs: List[np.ndarray]
    color_trail: List[List[Dict]]
    mean_surprise: float = 0.0
    std_surprise: float = 0.0

# --- Core Engine ---


class CE1Core:
    """
    A strange recombinator that traverses the fractal environment of a zip file,
    constantly reconfiguring itself to parse the DEFLATE stream of its own source code.
    """
    def __init__(self, genome: Genome):
        self.genome = genome
        self.cfg = genome.config
        self.grammar = genome.grammar
        # Minted symbols tracking
        self.minted_symbols: List[str] = []
        self.last_minted_symbol: Optional[str] = None

        # --- Opcode Sourcing ---
        # Look for custom operators in the genome and patch the core methods.
        # Initialize opcode metadata before loader uses it
        self.op_param_counts = {}
        self.fn_masks = {}
        self._load_opcodes()

        # -- Core Parameters --
        self.levels = self.cfg.levels
        self.symbols = self.cfg.symbols # These are the abstract states now
        self.S = len(self.symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        self.idx_to_symbol = {i: s for s, i in self.symbol_to_idx.items()}

        # -- Dynamics Parameters --
        self.alpha = self.cfg.seed.alpha
        self.epsilon0 = self.cfg.seed.epsilon0
        self.beta = self.cfg.seed.beta
        self.kappa = self.cfg.seed.kappa
        self.delta_spawn = self.cfg.seed.delta_spawn

        # -- Fractal & Transition Parameters --
        self.scales = []
        scales_dict = self.cfg.fractal.scales
        for k in range(self.levels):
            self.scales.append(float(scales_dict.get(f"l{k}", 1.0)))
        
        self.self_similarity = self.cfg.fractal.self_similarity
        
        priors = np.zeros((self.S, self.S), dtype=float)
        for i, s_from in enumerate(self.symbols):
            row = self.cfg.priors.get(s_from, {})
            for j, s_to in enumerate(self.symbols):
                priors[i, j] = float(row.get(s_to, 1.0 / self.S))
            priors[i] /= max(priors[i].sum(), 1e-9)
        self.priors = priors
        self.transitions_per_level = self._build_level_kernels()

        # -- State Variables --
        self.mu_rho = np.zeros(self.levels, dtype=float)
        self.surprise_potential = np.zeros(self.levels, dtype=float)
        
        self.rng = np.random.RandomState(self.cfg.stochastic_seed)
        self.mode = 1 - 2 * (self.cfg.stochastic_seed % 2)

        # --- Branch Cuts State ---
        self.branch = [BranchState() for _ in range(self.levels)]
        # Per-level thresholds introduce asymmetry; simple progression
        base_th = 0.3
        self.branch_thresholds = [base_th * (1.0 + 0.05 * k) for k in range(self.levels)]
        self.color_trail: List[List[Dict]] = [[] for _ in range(self.levels)]
        self._proj_cache: Dict[int, np.ndarray] = {}
        self._length_beads: Dict[int, np.ndarray] = {}
        # removed event bus to keep core compact
        self.error_state: List[bool] = [False for _ in range(self.levels)]
        self.op_param_counts: Dict[str, int] = {}
        
        # Block processor is stateless, no need to instantiate

        # --- Worlds registry (typemasks, completeness) ---
        # Define minimal bit taxonomy for typemask coverage
        self.WORLD_BITS: Dict[str, int] = {
            'block:stored': 1 << 0,
            'block:static': 1 << 1,
            'block:dynamic': 1 << 2,
            'ev:literal': 1 << 3,
            'ev:match': 1 << 4,
            'ev:phase': 1 << 5,
            'ev:atom': 1 << 6,
            'ev:summary': 1 << 7,
        }
        self.worlds: Dict = self._load_worlds()
        # Module-owned carry for CE1: minted symbols (∑), cycles (∞)
        self.carry_ce1 = Opix()
        # Select an active world based on policy at startup
        try:
            self._select_active_world()
            self._save_worlds()
        except Exception:
            pass

    def _load_opcodes(self):
        """Checks the genome for bytecode opcodes and patches core methods."""
        
        # --- Patch Genome.blend ---
        blend_op_path = 'ops/simple_blend.pyc'
        if blend_op_path in self.genome.extra_assets:
            logger.info(f"Found custom blend opcode at '{blend_op_path}'. Patching method.")
            
            op_bytes = self.genome.extra_assets[blend_op_path]
            
            # Create a module from the bytecode
            spec = importlib.util.spec_from_loader('custom_blend_op', loader=None)
            if spec is None:
                logger.warning("Failed to create spec for custom_blend_op; skipping opcode load.")
                return
            module = importlib.util.module_from_spec(spec)
            
            # The bytecode needs to be unmarshaled and then executed
            import marshal
            code_obj = marshal.loads(op_bytes[16:]) # Skip the 16-byte .pyc header
            exec(code_obj, module.__dict__)
            
            # --- Invariant Check & Patch ---
            if hasattr(module, 'blend'):
                custom_blend = module.blend
                # Check the function signature
                sig = inspect.signature(custom_blend)
                expected_params = ['g1', 'g2', 'factor']
                if list(sig.parameters.keys()) == expected_params:
                    Genome.blend = staticmethod(custom_blend)
                    logger.info("Custom blend method passed invariant checks and was patched.")
                    self.op_param_counts['blend'] = len(sig.parameters)
                    self.fn_masks['blend'] = 0  # no external inputs
                else:
                    logger.warning(f"Opcode 'blend' has an invalid signature: {sig}. Using default method.")
            else:
                logger.warning(f"Opcode '{blend_op_path}' does not have a 'blend' function.")
        else:
            self.op_param_counts['blend'] = 3
            self.fn_masks['blend'] = 0


    def _build_level_kernels(self) -> List[np.ndarray]:
        kernels = []
        base = self.priors
        for k in range(self.levels):
            M = self.self_similarity * base + (1.0 - self.self_similarity) * (base @ base)
            M = np.clip(M * self.scales[k], 1e-9, None)
            M /= M.sum(axis=1, keepdims=True)
            kernels.append(M)
            base = M
        return kernels

    def _get_length_bead(self, L: int) -> np.ndarray:
        Lc = int(max(0, min(15, L)))
        bead = self._length_beads.get(Lc)
        if bead is not None:
            return bead
        # Note: This is a more complex rotation not covered by the basic resonance module
        # and remains part of the core engine's specialized logic for now.
        axis = self._get_project3(16) @ np.pad(np.eye(1, 16, Lc).ravel(), (0,0))

        # We can still use the generic quaternion constructor from the new module
        bead = quaternion.axis_angle_quat(axis, 1.0)
        self._length_beads[Lc] = bead
        return bead

    def _bead_conjugate(self, q: np.ndarray, bead: np.ndarray, lam: float) -> np.ndarray:
        # This logic is also highly specific to the core engine's state machine.
        # It can be simplified to use the generic quaternion operations.
        b_conj = quaternion.quat_conj(bead)
        q_conj = quaternion.quat_mul(quaternion.quat_mul(b_conj, q), bead)

        # A full slerp implementation is still needed for this specialized operation.
        # It will remain here for now.
        return quaternion.quat_slerp(q, q_conj, float(np.clip(lam, 0.0, 1.0)))
    
    # --- Fast Text Ingestion to Bootstrap Grammar ---
    def _tokenize_text(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        return tokens

    def _tokenize_bytes(self, name: str, data: bytes) -> List[str]:
        # Default tokenizer if no opcode provided
        mask = self.fn_masks.get('tokenize', IN_NAME | IN_BYTES | IN_TEXT)
        text = None
        if mask & IN_TEXT:
            try:
                text = data.decode('utf-8', errors='ignore')
            except Exception:
                text = ''
        if mask & IN_BYTES and not (mask & IN_TEXT):
            # bytes-only op; convert bytes to safe token sequence (hex pairs)
            hexpairs = ' '.join(f"{b:02x}" for b in data[:4096])
            return hexpairs.split()
        # default path uses text
        return self._tokenize_text(text or '')

    def _ingest_text_asset(self, name: str, data: bytes):
        if not data or len(data) < 8:
            return
        # Heuristic: ensure it's mostly printable
        printable = sum(1 for b in data if 32 <= b <= 126 or b in (10,13,9))
        if printable / max(len(data), 1) < 0.6:
            return
        tokens = self._tokenize_bytes(name, data)
        if len(tokens) < 2:
            return
        # Update unigrams and bigrams
        for t in tokens:
            self.grammar.unigram_counts[t] += 1
            if t not in self.grammar.semantic_alleles:
                self.grammar.semantic_alleles[t] = np.array([1.0, 0.0, 0.0, 0.0])
        for a, b in zip(tokens[:-1], tokens[1:]):
            self.grammar.bigram_counts[a][b] += 1

    # --- Branch-cut Operators ---
    def _get_project3(self, n: int) -> np.ndarray:
        """
        Returns a deterministic 3xN projection matrix (rows orthonormal) seeded by cfg.stochastic_seed.
        """
        if n in self._proj_cache:
            return self._proj_cache[n]
        rng = np.random.RandomState(self.cfg.stochastic_seed + n)
        M = rng.normal(size=(3, n))
        # Orthonormalize rows
        for i in range(3):
            for j in range(i):
                denom = (M[j] @ M[j]) + 1e-12
                M[i] -= M[j] * ((M[i] @ M[j]) / denom)
            norm = float(np.linalg.norm(M[i]))
            M[i] /= max(norm, 1e-12)
        self._proj_cache[n] = M
        return M

    def _rotation_from_inverted_diff(self, external_tree: Dict, internal_counts: Dict) -> Tuple[np.ndarray, float]:
        """
        Build quaternion rotation from inverted discrepancy between external atom distribution
        and internal grammar distribution. Returns (q_rotation, angle_scalar).
        """
        # Normalize keys to strings to avoid mixed-type sorting (e.g., int Huffman symbols vs str tokens)
        all_syms = sorted({str(k) for k in external_tree.keys()} | {str(k) for k in internal_counts.keys()})
        if not all_syms:
            return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
        index = {s: i for i, s in enumerate(all_syms)}
        e = np.zeros(len(all_syms), dtype=float)
        g = np.zeros(len(all_syms), dtype=float)
        te = float(sum(external_tree.values())) or 1.0
        tg = float(sum(internal_counts.values())) or 1.0
        for s, c in external_tree.items():
            e[index[str(s)]] = float(c) / te
        for s, c in internal_counts.items():
            g[index[str(s)]] = float(c) / tg
        eps = 1e-9
        r = np.log((g + eps) / (e + eps))
        angle = float(np.linalg.norm(r))
        if angle == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
        # Project to 3D axis deterministically
        M = self._get_project3(len(all_syms))
        axis3 = M @ r
        axis_norm = float(np.linalg.norm(axis3))
        if axis_norm == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0]), angle
        axis3 /= axis_norm
        q_rot = quaternion.axis_angle_quat(axis3, angle)
        return q_rot, angle

    def _push_color(self, level: int, angle: float):
        # Map branch state to HSV-like info
        st = self.branch[level]
        hue = 0.0 if (st.sheet % 2 == 0) else 0.66
        s = float(np.clip(angle, 0.0, 1.0))
        v = float(np.clip(0.5 + 0.1 * abs(st.winding), 0.0, 1.0))
        self.color_trail[level].append({'h': hue, 's': s, 'v': v})
        # event bus removed

    def _enter_branch(self, level: int, q: np.ndarray, angle: float, q_rot: np.ndarray, direction: int = 1) -> np.ndarray:
        st = self.branch[level]
        # Nonlinear shaping of angle and level parity bias
        shaped = math.tanh(angle)
        # Per-level threshold
        th = self.branch_thresholds[level]
        st.progress += shaped
        # Save remainder beyond threshold; allow multiple cuts in one update
        if st.progress >= th:
            cuts = int(st.progress // th)
            st.progress -= cuts * th
            st.sheet ^= (cuts % 2)
            st.cuts += cuts
            st.winding += cuts * (1 if direction >= 0 else -1)
        # Odd/even level asymmetry: conjugation order
        if (level % 2) == 1:
            q_new = quaternion.quat_norm(quaternion.quat_mul(q, q_rot))
        else:
            q_new = quaternion.quat_norm(quaternion.quat_mul(q_rot, q))
        # Embed a lightweight tag in the mantissa of z to mark progression
        tag = self._event_tag12({'type': 'progress', 'level': level, 'angle': shaped, 'dir': direction})
        q_new = self._embed_tag_in_q(q_new, tag, component=3)
        self._push_color(level, shaped)
        return q_new

    def _recover_from_invalid_huffman(self, stream: ConstBitStream, level: int, q: np.ndarray) -> np.ndarray:
        """Delegate to unified branch_cut with byte-skip."""
        return self.branch_cut(stream, level, q, reason="invalid_huffman", do_skip=True)

    def _malformed_branch(self, level: int, reason: str, q: np.ndarray) -> np.ndarray:
        """Delegate to unified branch_cut without byte-skip."""
        return self.branch_cut(None, level, q, reason=reason, do_skip=False)

    def branch_cut(self, stream: Optional[ConstBitStream], level: int, q: np.ndarray, reason: str, do_skip: bool) -> np.ndarray:
        """Unified branch-cut: optional byte-align/skip, forced identity rotation, error flag, log."""
        if do_skip and stream is not None:
            misalign = int((-stream.pos) % 8)
            if misalign > 0:
                stream.pos += misalign
            if stream.pos + 8 <= len(stream):
                stream.pos += 8
        angle = self.branch_thresholds[level] + 1e-3
        q = self._enter_branch(level, q, angle, np.array([1.0, 0.0, 0.0, 0.0]), direction=1)
        self.error_state[level] = True
        logger.warning(f"Branch cut: {reason}.")
        return q

    def excite_by_length_hist(self, q: np.ndarray, lit_len_code_lengths: Dict, angle: float) -> np.ndarray:
        lengths = np.array(list(lit_len_code_lengths.values()), dtype=int)
        if lengths.size == 0:
            return q
        bins = np.bincount(np.clip(lengths, 0, 15), minlength=16)
        total = int(bins.sum())
        closure = float(np.count_nonzero(bins)) / 16.0
        scale = float(np.clip(1.0 - closure, 0.0, 1.0))
        q_blend = q.copy()
        for Lc, cnt in enumerate(bins):
            if cnt <= 0:
                continue
            bead = self._get_length_bead(Lc)
            lam = float(np.clip(angle * (Lc + 1) / 12.0, 0.0, 1.0)) * scale * (cnt / max(total, 1))
            q_blend = self._bead_conjugate(q_blend, bead, lam)
        return quaternion.quat_norm(q_blend)

    def apply_dynamic_tree(self, q: np.ndarray, lit_len_code_lengths: Dict, internal_counts: Dict) -> Tuple[np.ndarray, float, Optional[int]]:
        external_counts = dict(
            lit_len_code_lengths) if lit_len_code_lengths else {}
        q_rot, angle = self._rotation_from_inverted_diff(external_counts, internal_counts)
        q = self._enter_branch(level=0, q=q, angle=angle, q_rot=q_rot)
        dom_L = None
        if lit_len_code_lengths:
            q = self.excite_by_length_hist(q, lit_len_code_lengths, angle)
            lengths = np.array(list(lit_len_code_lengths.values()), dtype=int)
            if lengths.size:
                vals, cnts = np.unique(np.clip(lengths, 0, 15), return_counts=True)
                dom_L = int(vals[np.argmax(cnts)])
        return q, angle, dom_L

    def _ensure_symbol_exists(self, symbol: str):
        """Dynamically add new symbols to the system."""
        if symbol not in self.symbol_to_idx:
            # Add to symbols list
            self.symbols.append(symbol)
            new_idx = len(self.symbols) - 1
            
            # Add to mappings
            self.symbol_to_idx[symbol] = new_idx
            self.idx_to_symbol[new_idx] = symbol
            
            # Expand transition matrices for all levels
            for level in range(self.levels):
                old_size = self.transitions_per_level[level].shape[0]
                new_size = old_size + 1
                
                # Create new expanded matrix
                new_matrix = np.zeros((new_size, new_size))
                new_matrix[:old_size, :old_size] = self.transitions_per_level[level]
                
                # Initialize new row/column with uniform probabilities
                new_matrix[old_size, :] = 1.0 / new_size
                new_matrix[:, old_size] = 1.0 / new_size
                
                self.transitions_per_level[level] = new_matrix
            
            self.S = len(self.symbols)
            # logger.info(f"Added new symbol: '{symbol}' (total: {self.S})")

    def _emit_step(self, level: int, q: np.ndarray, state_symbol: str) -> str:
        # Ensure symbol exists before trying to use it
        self._ensure_symbol_exists(state_symbol)
        next_symbol, _alleles, _color, _delta = self.step(
            level, q, state_symbol)
        return next_symbol


    # --- Steganographic event tags in float mantissa ---
    def _event_tag12(self, event: Dict) -> int:
        s = f"{event.get('type','')}|{event.get('level','')}|{event.get('reason','')}|{event.get('hlit','')}|{event.get('hdist','')}|{event.get('angle','')}"
        h = hashlib.sha1(s.encode('utf-8')).digest()
        return int.from_bytes(h[:2], 'little') & 0x0FFF

    def _embed_tag_in_q(self, q: np.ndarray, tag12: int, component: int = 3) -> np.ndarray:
        v = float(q[component])
        u = np.frombuffer(np.float64(v).tobytes(), dtype=np.uint64)[0]
        u = (u & ~np.uint64(0x0FFF)) | np.uint64(tag12 & 0x0FFF)
        v2 = np.frombuffer(np.uint64(u).tobytes(), dtype=np.float64)[0]
        q2 = q.copy()
        q2[component] = v2
        return q2

    def _extract_tag12(self, q: np.ndarray, component: int = 3) -> int:
        v = float(q[component])
        u = np.frombuffer(np.float64(v).tobytes(), dtype=np.uint64)[0]
        return int(u & np.uint64(0x0FFF))

    # --- Worlds registry helpers ---
    def _default_worlds(self) -> Dict:
        return {
            'version': 1,
            'active': 'default',
            'worlds': {
                'default': {
                    'name': 'default',
                    'typemask': 0,
                    'typemask_size': int(len(self.WORLD_BITS)),
                    'types_seen': {},
                    'size': 0,
                    'color': (0.0, 0.0, 0.0),
                    'complete': False,
                    'last_at': 0,
                }
            }
        }

    def _load_worlds(self) -> Dict:
        try:
            blob = self.genome.registry_get(self.genome.reg_wor)
            if isinstance(blob, (bytes, bytearray)):
                try:
                    obj = msgpack.unpackb(
                        bytes(blob), raw=False, strict_map_key=False)
                except TypeError:
                    obj = msgpack.unpackb(bytes(blob), raw=False)
                if isinstance(obj, dict) and 'worlds' in obj:
                    for w in obj.get('worlds', {}).values():
                        if 'typemask_size' not in w:
                            w['typemask_size'] = int(len(self.WORLD_BITS))
                    if 'active' not in obj:
                        obj['active'] = 'default'
                    return obj
        except Exception:
            pass
        return self._default_worlds()

    def _save_worlds(self) -> None:
        try:
            data = msgpack.packb(self.worlds, use_bin_type=True)
            self.genome.registry_set(self.genome.reg_wor, data)
        except Exception:
            pass

    def _active_world(self) -> Dict:
        name = self.worlds.get('active', 'default')
        worlds = self.worlds.setdefault('worlds', {})
        if name not in worlds:
            worlds[name] = {
                'name': name,
                'typemask': 0,
                'typemask_size': int(len(self.WORLD_BITS)),
                'types_seen': {},
                'size': 0,
                'color': (0.0, 0.0, 0.0),
                'complete': False,
                'last_at': 0,
                'glyph': '',
            }
        return worlds[name]

    def _world_coverage(self, w: Dict) -> int:
        mask = int((w or {}).get('typemask', 0))
        c = 0
        m = mask
        while m:
            m &= m - 1
            c += 1
        return c

    def _select_active_world(self) -> None:
        """Programmatically choose active world by policy.
        Priority: complete worlds by (-size, -last_at), then incomplete by (-coverage, -size, -last_at).
        """
        worlds = self.worlds.setdefault('worlds', {})
        if not worlds:
            self.worlds['active'] = 'default'
            self._active_world()
            return
        items = list(worlds.items())
        complete = [(n, w)
                    for n, w in items if bool((w or {}).get('complete'))]
        if complete:
            n, _w = sorted(complete, key=lambda it: (
                -int((it[1] or {}).get('size', 0)), -
                int((it[1] or {}).get('last_at', 0))
            ))[0]
            self.worlds['active'] = n
            return
        # else incomplete: use coverage then size then recency
        n, _w = sorted(items, key=lambda it: (
            -self._world_coverage(it[1] or {}), -int((it[1] or {}
                                                      ).get('size', 0)), -int((it[1] or {}).get('last_at', 0))
        ))[0]
        self.worlds['active'] = n

    def _sanitize_name(self, s: str, max_len: int = 32) -> str:
        s2 = ''.join(ch if (ch.isalnum() or ch in ('-', '_'))
                     else '-' for ch in s)
        s2 = s2.strip('-_')
        if not s2:
            return 'world'
        return s2[:max_len]

    def _generate_world_name(self, context: str) -> str:
        """Generate a readable world name from grammar and context (GPT-like naming by the VM)."""
        ctx_tokens = [t.lower()
                      for t in re.findall(r"[A-Za-z0-9_]+", context or '')]
        ug = self.genome.grammar.unigram_counts
        minted = self.minted_symbols if hasattr(self, 'minted_symbols') else []
        # score: freq + context overlap + minted bonus

        def score(tok: str) -> float:
            base = float(ug.get(tok, 0))
            overlap = 1.0 if tok.lower() in ctx_tokens else 0.0
            minted_bonus = 2.0 if tok in minted else 0.0
            return base + 3.0 * overlap + minted_bonus
        candidates = sorted(
            ug.keys(), key=lambda t: score(t), reverse=True)[:16]
        # prefer minted or contextual tokens first
        primaries = [t for t in minted if t in ug] + \
            [t for t in candidates if t.lower() in ctx_tokens]
        secondaries = [t for t in candidates if t not in primaries]
        parts = []
        if primaries:
            parts.append(primaries[0])
        if secondaries:
            parts.append(secondaries[0])
        if not parts:
            parts = [context or 'world']
        name = '-'.join(parts)
        return self._sanitize_name(name)

    def ensure_world_for_asset(self, asset_name: str) -> None:
        """Map an asset to a program-decided world; create/switch automatically."""
        key = (asset_name or '').split('/')[:1][0] or 'root'
        wname = self._generate_world_name(key)
        worlds = self.worlds.setdefault('worlds', {})
        if wname not in worlds:
            worlds[wname] = {
                'name': wname,
                'typemask': 0,
                'typemask_size': int(len(self.WORLD_BITS)),
                'types_seen': {},
                'size': 0,
                'color': (0.0, 0.0, 0.0),
                'complete': False,
                'last_at': 0,
                'glyph': '',
            }
        self.worlds['active'] = wname

    def _world_set_bit(self, bit_key: str) -> None:
        w = self._active_world()
        bit = int(self.WORLD_BITS.get(bit_key, 0))
        if bit:
            w['typemask'] = int(w.get('typemask', 0)) | bit
            # recompute completeness via popcount vs typemask_size
            mask = int(w['typemask'])
            coverage = 0
            m = mask
            while m:
                m &= m - 1
                coverage += 1
            size = int(w.get('typemask_size', len(self.WORLD_BITS)))
            was_complete = bool(w.get('complete', False))
            now_complete = bool(coverage >= size)
            w['complete'] = now_complete
            # Assign a stable glyph when world becomes complete
            if now_complete and not was_complete:
                if not w.get('glyph'):
                    w['glyph'] = self._choose_world_glyph(
                        w.get('name', 'default'))

    def _world_bump_type(self, key: str, n: int = 1) -> None:
        w = self._active_world()
        ts = w.setdefault('types_seen', {})
        ts[key] = int(ts.get(key, 0)) + int(n)

    def save_worlds(self) -> None:
        self._save_worlds()

    def _choose_world_glyph(self, name: str) -> str:
        """Pick a stable single-character glyph for a completed world."""
        # Curated monochrome-friendly set: digits, uppercase, a few math letters
        glyphs = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            'αβγδεζηθλμνξπρστυφχψω'
            '◇◆○●□■△▲▽▼◇'
        )
        if not name:
            name = 'world'
        idx = int(abs(stable_str_hash(str(name))) % len(glyphs))
        return glyphs[idx]

    def reconstruct_trace(self, transformed_state: Dict, level: int = 0) -> List[Dict]:
        qs: List[np.ndarray] = transformed_state.get('final_qs', [])
        trail_levels: List[List[Dict]] = transformed_state.get('color_trail', [])
        trail = trail_levels[level] if (trail_levels and level < len(trail_levels)) else []
        if not qs:
            return []
        steps = min(len(trail), max(len(qs) - 1, 0))
        out: List[Dict] = []
        for i in range(steps):
            q = qs[i+1]
            tag = self._extract_tag12(q, component=3)
            c = trail[i]
            out.append({'i': i, 'level': level, 'tag12': tag, 'q': q.tolist(), 'h': c.get('h'), 's': c.get('s'), 'v': c.get('v')})
        return out

    # --- Minimal helpers to unblock step() ---
    def _choose_next_symbol(self, q: np.ndarray, delta: float, base_probs: np.ndarray) -> Tuple[str, float]:
        probs = np.array(base_probs, dtype=float)
        probs = probs / max(probs.sum(), 1e-12)
        idx = int(self.rng.choice(len(probs), p=probs))
        return self.idx_to_symbol[idx], float(probs[idx])

    def _delta_to_color(self, delta: float) -> Dict:
        s = float(np.clip(abs(float(delta)), 0.0, 1.0))
        v = float(np.clip(0.5 + 0.5 * np.tanh(abs(float(delta))), 0.0, 1.0))
        return {'s': s, 'v': v}
    
    def step(self, level: int, q: np.ndarray, state_symbol: str):
        """
        Calculates the per-level consequences of a given state `q`.
        This is an observer function; it does not modify `q`.
        """
        # --- State-Modulated Surprise ---
        # The main state `q` now acts as the "partial sort quaternion",
        # modulating the system's sensitivity to surprise based on the Dynamic Law.
        energy, symmetry, entropy, drift = abs(q[0]), abs(q[1]), abs(q[2]), q[3] # Use abs for robustness
        base_potential = self.surprise_potential[level]
        
        sensitivity = (energy + symmetry) / (1.0 + entropy)
        modulated_potential = base_potential * sensitivity + drift

        # The creative impulse comes from this modulated potential.
        amp = np.tanh(modulated_potential)
        mu = (1.0 - self.alpha) * self.mu_rho[level] + self.alpha * amp
        self.mu_rho[level] = mu
        delta = amp - mu

        logger.debug(f"L{level} | State: {state_symbol}, Delta: {delta:.4f}, Pot: {modulated_potential:.4f}")

        # --- Particle Spawning ---
        # Alleles are "born" from the state, but this no longer modifies the core state `q` directly.
        alleles = []
        if abs(delta) > self.delta_spawn:
            evn = float(np.linalg.norm(q[1:4]))
            e_vec_norm = q[1:4] / max(evn, 1e-9)
            q_inj = q + self.kappa * abs(delta) * np.array([0.0, *e_vec_norm])
            alleles.append(q_inj)
            self.surprise_potential[level] = 0.0 # Discharge surprise
        
        # --- Abstract State Transition ---
        base_probs = self.transitions_per_level[level][self.symbol_to_idx[state_symbol]]
        next_symbol, transition_prob = self._choose_next_symbol(q, delta, base_probs)
        
        # Accumulate surprise (inverse of probability)
        surprise = (1.0 - transition_prob) * 0.1 # Scaled
        self.surprise_potential[level] += surprise
        
        color_info = self._delta_to_color(delta)
        return next_symbol, alleles, color_info, delta

    def resonate_with_block(self, q: np.ndarray, payload: Dict) -> np.ndarray:
        """Applies quaternion physics for a 'block' event."""
        kind = payload.get('type')

        if kind == 'dynamic':
            self._world_set_bit('block:dynamic')
            self._world_bump_type('block:dynamic')

            event = BlockEvent(
                kind='dynamic',
                lit_len_code_lengths=payload.get('lit_len_code_lengths', {})
            )
            rotation_q = get_rotation_from_block(event, self.genome)
            if rotation_q is not None:
                # derive angle from rotation quaternion w component
                w = float(rotation_q[0])
                w = float(np.clip(w, -1.0, 1.0))
                ang = float(2.0 * math.acos(w))
                q = self._enter_branch(
                    level=0, q=q, angle=ang, q_rot=rotation_q, direction=1)

        elif kind == 'static':
            self._world_set_bit('block:static')
            self._world_bump_type('block:static')
            axis = np.array([1.0, 0.0, 0.0])
            ang = 0.05
            rotation_q = quaternion.axis_angle_quat(axis, ang)
            q = self._enter_branch(level=0, q=q, angle=float(
                abs(ang)), q_rot=rotation_q, direction=1)
        else:  # stored or unknown
            self._world_set_bit('block:stored')
            self._world_bump_type('block:stored')
        return q

    # --- Online Learning and Execution ---
    def run_and_learn_asset(self, asset_name: str, time_budget_secs: Optional[float] = None) -> Dict:
        """
        Processes a single, compressed asset from the genome, inhabiting its fractal structure.
        """
        start_time = time.time()
        # Ensure a program-decided world for this asset
        try:
            self.ensure_world_for_asset(asset_name)
            self._save_worlds()
        except Exception:
            pass
        
        # Initialize block processor for this asset
        block_processor = BlockProcessor(asset_name, self.genome, self._ingest_text_asset)
        
        # The main state quaternion `q` now represents the VM's current "understanding"
        # or its active "Huffman tree" for the current context.
        q = np.array([1.0, 0.0, 0.0, 0.0]) 
        final_qs: List[np.ndarray] = [q.copy()]
        symbol = 'drift' # The abstract state machine
        all_deltas = []
        angle = 0.0
        
        # Consume unified event stream
        last_glyphs: Optional[str] = None
        last_counts = None
        surp = Opix()
        minted_in_asset = 0
        events_in_asset = 0
        for ev_type, payload in block_processor.events():
            # --- Time check per event ---
            if time_budget_secs and (time.time() - start_time) > time_budget_secs:
                logger.warning("Time budget exceeded. Halting.")
                break
            events_in_asset += 1
            if ev_type == 'block' and isinstance(payload, dict):
                q = self.resonate_with_block(q, payload)
            elif ev_type == 'atom' and isinstance(payload, dict):
                self._world_set_bit('ev:atom')
                self._world_bump_type('ev:atom')
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
                self._world_set_bit('ev:literal')
                self._world_bump_type('ev:literal')
                surp['s'] += 1
            elif ev_type == 'match':
                self._world_set_bit('ev:match')
                self._world_bump_type('ev:match')
                surp['Δ'] += 1
            elif ev_type == 'phase':
                self._world_set_bit('ev:phase')
                self._world_bump_type('ev:phase')
                surp['S'] += 1
            elif ev_type == 'summary' and isinstance(payload, dict):
                self._world_set_bit('ev:summary')
                self._world_bump_type('ev:summary')
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
        try:
            w = self._active_world()
            w['size'] = int(w.get('size', 0)) + int(events_in_asset)
            w['color'] = tuple(float(x) for x in hsv)
            w['last_at'] = int(time.time())
        except Exception:
            pass
        stats = {'mean_surprise': 0.0, 'std_surprise': 0.0,
                 'atoms': minted_in_asset, 'glyphs': glyphs, 'glyph_counts': (last_counts or {}), 'color': hsv}
        # inject CE1 carry into result for potential rendering
        stats['carry_ce1'] = dict(self.carry_ce1)
        return {'final_q': q, 'final_qs': final_qs, 'color_trail': self.color_trail, 'stats': stats}

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
