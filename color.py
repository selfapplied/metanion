import numpy as np
from typing import List, Dict, Tuple
from collections import namedtuple
from branch import BranchState # We need this to determine hue

class ColorEngine:
    def __init__(self, levels: int):
        self.levels = levels
        self.trail: List[List[Dict]] = [[] for _ in range(self.levels)]

    def push(self, level: int, angle: float, branch_state: BranchState):
        """Pushes a new color event onto the trail for a given level."""
        hue = 0.0 if (branch_state.sheet % 2 == 0) else 0.66
        s = float(np.clip(angle, 0.0, 1.0))
        v = float(np.clip(0.5 + 0.1 * abs(branch_state.winding), 0.0, 1.0))
        self.trail[level].append({'h': hue, 's': s, 'v': v})

    def from_delta(self, delta: float) -> Dict:
        """Generates a color dict from a delta value."""
        s = float(np.clip(abs(float(delta)), 0.0, 1.0))
        v = float(np.clip(0.5 + 0.5 * np.tanh(abs(float(delta))), 0.0, 1.0))
        return {'s': s, 'v': v}
        
    def get_trail(self) -> List[List[Dict]]:
        """Returns the current color trail."""
        return self.trail

    def clear_trail(self):
        """Clears the color trail for a new run."""
        self.trail = [[] for _ in range(self.levels)]
