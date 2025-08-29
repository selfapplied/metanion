import numpy as np
from typing import Callable, Any, List, Dict, Tuple, Optional
from functools import reduce
from quaternion import shape_curvature
from dataclasses import dataclass
from aspire import Opix

# Core types
Gene = Callable[[], Any]
Boundary = Callable[[Any], List[int]]
Genome = List[Gene]

@dataclass(frozen=True)
class Wave:
    data: np.ndarray

# Boundary operators
class Boundaries:
    @staticmethod
    def every(n: int) -> Boundary:
        return lambda x: list(range(0, len(x) if hasattr(x, '__len__') else 1, n))
    
    @staticmethod
    def at(pattern: str) -> Boundary:
        import re
        return lambda s: [m.start() for m in re.finditer(pattern, s)]
    
    @staticmethod
    def where(pred: Callable) -> Boundary:
        return lambda x: [i for i, v in enumerate(x) if pred(v)]
    
    @staticmethod
    def transitions(pred: Callable) -> Boundary:
        return lambda x: [i for i in range(1, len(x)) if pred(x[i-1], x[i])]
    
    @staticmethod
    def combine(*bounds: Boundary) -> Boundary:
        return lambda x: sorted(set().union(*(b(x) for b in bounds)))
    
    @staticmethod
    def quaternion(q_seq: List[np.ndarray]) -> Boundary:
        """Use quaternion curvature for boundaries"""
        def boundary_fn(x):
            if len(x) < 3:
                return []
            q_vecs = np.array([q[1:4] for q in x])
            curvatures = shape_curvature(q_vecs)
            threshold = np.percentile(curvatures, 85)
            return [i for i, c in enumerate(curvatures) if c > threshold]
        return boundary_fn

# Gene creators
def gene(expr: Callable, boundary: Optional[Boundary] = None) -> Gene:
    return lambda: (expr(), boundary)

def color(h: float, s: float = 0.7, v: float = 0.8) -> Gene:
    boundary = Boundaries.transitions(lambda a, b: abs(a-b) > 0.1)
    return gene(lambda: (h, s, v), boundary)

def quat_color(h: float, s: float, v: float, q: np.ndarray) -> Gene:
    """Color gene with quaternion-aware boundaries"""
    boundary = Boundaries.quaternion([q])
    return gene(lambda: (h, s, v), boundary)

def regulator(target: Gene, modifier: Callable, boundary: Optional[Boundary] = None) -> Gene:
    return gene(lambda: modifier(target()[0]), boundary or target()[1])

def promoter(activator: Callable, boundary: Optional[Boundary] = None) -> Gene:
    return gene(activator, boundary)

# Expression and recombination
def express(gene: Gene) -> Any:
    return gene()

def recombine(gene1: Gene, gene2: Gene) -> Gene:
    val1, bounds1 = express(gene1)
    val2, bounds2 = express(gene2)
    
    if not bounds1 or not bounds2:
        return gene1
    
    seq1 = val1 if hasattr(val1, '__len__') else [val1]
    seq2 = val2 if hasattr(val2, '__len__') else [val2]
    
    split1 = bounds1[0] % len(seq1) if bounds1 else 0
    split2 = bounds2[0] % len(seq2) if bounds2 else 0
    
    recombinant = seq1[:split1] + seq2[split2:]
    new_boundary = Boundaries.combine(bounds1, bounds2)
    
    return gene(lambda: (recombinant, new_boundary))

# Color phase system
class ColorPhases:
    def __init__(self):
        self.phases: List[Gene] = []
        self.q_history: List[np.ndarray] = []
        self.alerts: Opix = Opix()
    
    def add_phase(self, phase: Gene):
        self.phases.append(phase)
        self.alerts['🎨'] += 1  # Track phase additions
    
    def update_quaternion(self, q: np.ndarray):
        self.q_history.append(q.copy())
        if len(self.q_history) > 100:  # Keep last 100
            self.q_history.pop(0)
    
    def get_color(self, delta: float, q: np.ndarray) -> Tuple[float, float, float]:
        if not self.phases:
            # Fallback: simple HSV mapping
            s = float(np.clip(abs(delta), 0.0, 1.0))
            v = float(np.clip(0.5 + 0.5 * np.tanh(abs(delta)), 0.0, 1.0))
            h = 0.0 if (q[0] > 0) else 0.66
            return (h, s, v)

        # Update quaternion history
        self.update_quaternion(q)
        
        # Find best phase based on quaternion state
        best_phase = self._select_phase(q)
        if best_phase:
            color_val, _ = express(best_phase)
            
            # Apply delta modulation
            h, s, v = color_val
            s = float(np.clip(s * abs(delta), 0.0, 1.0))
            v = float(np.clip(v * (0.5 + 0.5 * np.tanh(abs(delta))), 0.0, 1.0))
            
            return (h, s, v)
        
        # Fallback if no phase selected
        s = float(np.clip(abs(delta), 0.0, 1.0))
        v = float(np.clip(0.5 + 0.5 * np.tanh(abs(delta)), 0.0, 1.0))
        h = 0.0 if (q[0] > 0) else 0.66
        return (h, s, v)
    
    def _select_phase(self, q: np.ndarray) -> Optional[Gene]:
        if len(self.q_history) < 3:
            return self.phases[0] if self.phases else None
        
        # Use quaternion curvature to select phase
        curvatures = shape_curvature(np.array([q[1:4] for q in self.q_history[-10:]]))
        if len(curvatures) > 0:
            curvature_level = np.mean(curvatures)
            phase_idx = int(curvature_level * len(self.phases)) % len(self.phases)
            return self.phases[phase_idx]
        
        return self.phases[0] if self.phases else None
    
    def learn_from_samples(self, samples):
        """Learn color phases from (delta, quaternion, color) samples"""
        if len(samples) < 3:
            return
        
        # Group samples by quaternion similarity
        for delta, q, (h, s, v) in samples:
            # Create quaternion-aware color gene
            phase = color(h, s, v)  # Use regular color function for now
            self.add_phase(phase)
        
        self.alerts['📚'] += len(samples)  # Track learning activity
    
    def get_status(self) -> Opix:
        """Get current status as Opix object"""
        status = Opix()
        status['phases'] = len(self.phases)
        status['history'] = len(self.q_history)
        status['alerts'] = self.alerts
        return status

# Example usage
if __name__ == '__main__':
    # Create color phases
    phases = ColorPhases()
    
    # Add some base phases
    red = color(0.0, 0.9, 0.9)
    blue = color(0.66, 0.9, 0.9)
    green = color(0.33, 0.9, 0.9)
    
    phases.add_phase(red)
    phases.add_phase(blue)
    phases.add_phase(green)
    
    # Test recombination
    purple = recombine(red, blue)
    phases.add_phase(purple)
    
    # Test quaternion integration
    q = np.array([1.0, 0.1, 0.2, 0.3])
    color_result = phases.get_color(0.5, q)
    
    # Use Opix-based output
    status = phases.get_status()
    print(f"Status: {status}")
    print(f"Color: {color_result}")
