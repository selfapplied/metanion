import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import namedtuple
import quaternion
from bitstring import ConstBitStream
import logging
import math

logger = logging.getLogger(__name__)

# --- Branch State ---
BranchState = namedtuple(
    'BranchState', ['sheet', 'cuts', 'winding', 'progress'])

# --- Branch Cut Operator ---
class BranchCutter:
    def __init__(self, levels: int, stochastic_seed: int):
        self.levels = levels
        self.stochastic_seed = stochastic_seed
        self.branch = [BranchState(
            sheet=0, cuts=0, winding=0, progress=0.0) for _ in range(self.levels)]
        base_th = 0.3
        self.branch_thresholds = [
            base_th * (1.0 + 0.05 * k) for k in range(self.levels)]
        self.error_state: List[bool] = [False for _ in range(self.levels)]
        self._proj_cache: Dict[int, np.ndarray] = {}
        self._length_beads: Dict[int, np.ndarray] = {}

    def project3(self, n: int) -> np.ndarray:
        if n in self._proj_cache:
            return self._proj_cache[n]
        rng = np.random.RandomState(self.stochastic_seed + n)
        M = rng.normal(size=(3, n))
        for i in range(3):
            for j in range(i):
                denom = (M[j] @ M[j]) + 1e-12
                M[i] -= M[j] * ((M[i] @ M[j]) / denom)
            norm = float(np.linalg.norm(M[i]))
            M[i] /= max(norm, 1e-12)
        self._proj_cache[n] = M
        return M

    def length_bead(self, L: int) -> np.ndarray:
        Lc = int(max(0, min(15, L)))
        bead = self._length_beads.get(Lc)
        if bead is not None:
            return bead
        axis = self.project3(16) @ np.pad(np.eye(1, 16, Lc).ravel(), (0, 0))
        bead = quaternion.axis_angle_quat(axis, 1.0)
        self._length_beads[Lc] = bead
        return bead

    def bead_conjugate(self, q: np.ndarray, bead: np.ndarray, lam: float) -> np.ndarray:
        b_conj = quaternion.quat_conj(bead)
        q_conj = quaternion.quat_mul(quaternion.quat_mul(b_conj, q), bead)
        return quaternion.quat_slerp(q, q_conj, float(np.clip(lam, 0.0, 1.0)))

    def enter(self, level: int, q: np.ndarray, angle: float, q_rot: np.ndarray, direction: int = 1) -> Tuple[np.ndarray, BranchState, float]:
        st = self.branch[level]
        shaped = math.tanh(angle)
        th = self.branch_thresholds[level]
        
        new_progress = st.progress + shaped
        new_sheet, new_cuts, new_winding = st.sheet, st.cuts, st.winding
        
        if new_progress >= th:
            cuts_to_make = int(new_progress // th)
            new_progress -= cuts_to_make * th
            new_sheet ^= (cuts_to_make % 2)
            new_cuts += cuts_to_make
            new_winding += cuts_to_make * (1 if direction >= 0 else -1)
        
        new_branch_state = BranchState(
            sheet=new_sheet, cuts=new_cuts, winding=new_winding, progress=new_progress)
        
        if (level % 2) == 1:
            q_new = quaternion.quat_norm(quaternion.quat_mul(q, q_rot))
        else:
            q_new = quaternion.quat_norm(quaternion.quat_mul(q_rot, q))
        
        # Return shaped_angle so the ColorEngine can use it
        return q_new, new_branch_state, shaped

    def cut(self, stream: Optional[ConstBitStream], level: int, q: np.ndarray, reason: str, do_skip: bool) -> Tuple[np.ndarray, BranchState, float]:
        if do_skip and stream is not None:
            misalign = int((-stream.pos) % 8)
            if misalign > 0:
                stream.pos += misalign
            if stream.pos + 8 <= len(stream):
                stream.pos += 8
        angle = self.branch_thresholds[level] + 1e-3
        q_new, new_branch_state, shaped_angle = self.enter(level, q, angle, np.array([1.0, 0.0, 0.0, 0.0]), direction=1)
        self.branch[level] = new_branch_state
        self.error_state[level] = True
        logger.warning(f"Branch cut: {reason}.")
        return q_new, new_branch_state, shaped_angle
