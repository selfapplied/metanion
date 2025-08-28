import math
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from functools import partial

# --- Function-table bitmask (inputs) ---
IN_NAME  = 1 << 0
IN_BYTES = 1 << 1
IN_TEXT  = 1 << 2
IN_INT   = 1 << 3
IN_FLOAT = 1 << 4

# --- Type Adapters (Curried Functions) ---
def str_to_int(s: str) -> int:
    """String to int via hash/mod."""
    return hash(s) % (1 << 32)

def str_to_float(s: str) -> float:
    """String to float via embedding."""
    return float(hash(s)) / (1 << 32)

def int_to_float(n: int) -> float:
    """Int to float via log/scale."""
    return math.log(max(abs(n), 1)) * (1 if n >= 0 else -1)

def float_to_int(x: float) -> int:
    """Float to int via bin/round."""
    return int(round(x))

# --- Bridge Functions ---
def bridge_str_int(f: Callable[[int], Any]) -> Callable[[str], Any]:
    """Bridge string input to int function."""
    return lambda s: f(str_to_int(s))

def bridge_str_float(f: Callable[[float], Any]) -> Callable[[str], Any]:
    """Bridge string input to float function."""
    return lambda s: f(str_to_float(s))

def bridge_int_float(f: Callable[[float], Any]) -> Callable[[int], Any]:
    """Bridge int input to float function."""
    return lambda n: f(int_to_float(n))

def bridge_float_int(f: Callable[[int], Any]) -> Callable[[float], Any]:
    """Bridge float input to int function."""
    return lambda x: f(float_to_int(x))

# --- Core Operators (Curried Functions) ---
def gap_op(q: np.ndarray, bins: np.ndarray, closure: float, q_rot: np.ndarray, level: int, eps: float) -> np.ndarray:
    """Core gap operator: bead blending, rotation, mantissa tagging."""
    # This will be wired to _excite_by_bins
    return q

def mod_op(modulus: int) -> Callable[[int], int]:
    """Modulus operation for angles, progress, lengths."""
    return lambda n: n % modulus

def token_op(token_a: str, token_b: str) -> str:
    """Token manipulation and grammar operations."""
    return f"{token_a}{token_b}"

# --- Operator Registry (Function-based) ---
class OpRegistry:
    """Lightweight registry for curried operators."""
    
    def __init__(self):
        self.ops: Dict[str, Callable] = {}
        self.masks: Dict[str, int] = {}
        self._setup_default_ops()
    
    def _setup_default_ops(self):
        """Initialize default operators with their masks."""
        # Core operators
        self.register("gap", gap_op, IN_FLOAT, 6)
        self.register("mod16", mod_op(16), IN_INT, 1)
        self.register("mod360", mod_op(360), IN_INT, 1)
        self.register("token", token_op, IN_TEXT | IN_NAME, 2)
        
        # Bridged operators - use empty arrays/values as placeholders
        self.register("gap_str", bridge_str_float(lambda x: gap_op(np.array([1.0, 0.0, 0.0, 0.0]), np.array([]), 0.0, np.array([1.0, 0.0, 0.0, 0.0]), 0, 0.0)), IN_TEXT, 1)
        self.register("mod_str", bridge_str_int(mod_op(16)), IN_TEXT, 1)
    
    def register(self, name: str, op: Callable, mask: int, arity: int):
        """Register an operator function with its mask and arity."""
        self.ops[name] = op
        self.masks[name] = mask
    
    def can_accept(self, name: str, inputs: List) -> bool:
        """Check if operator can accept the given inputs."""
        if name not in self.ops:
            return False
        
        mask = self.masks[name]
        if len(inputs) != self.ops[name].__code__.co_argcount:
            return False
        
        for inp in inputs:
            if isinstance(inp, str) and not (mask & (IN_TEXT | IN_NAME)):
                return False
            elif isinstance(inp, int) and not (mask & IN_INT):
                return False
            elif isinstance(inp, float) and not (mask & IN_FLOAT):
                return False
            elif isinstance(inp, bytes) and not (mask & IN_BYTES):
                return False
        return True
    
    def execute(self, name: str, inputs: List) -> Any:
        """Execute an operator by name."""
        if name not in self.ops:
            raise ValueError(f"Unknown operator: {name}")
        return self.ops[name](*inputs)
    
    def find_compatible(self, inputs: List) -> List[str]:
        """Find all operators that can accept the given inputs."""
        compatible = []
        for name in self.ops:
            if self.can_accept(name, inputs):
                compatible.append(name)
        return compatible
    
    def route(self, inputs: List, preferred_type: str = None) -> Optional[str]:
        """Route inputs to the best operator name."""
        compatible = self.find_compatible(inputs)
        if not compatible:
            return None
        
        if preferred_type:
            # Prefer operators of the specified type
            typed_ops = [name for name in compatible if preferred_type in name.lower()]
            if typed_ops:
                return typed_ops[0]
        
        return compatible[0]

# --- Convenience Functions ---
def create_float_op(f: Callable[[float], Any]) -> Callable[[Any], Any]:
    """Create a float operator that bridges other types."""
    return lambda x: f(float(x))

def create_int_op(f: Callable[[int], Any]) -> Callable[[Any], Any]:
    """Create an int operator that bridges other types."""
    return lambda x: f(int(x))

def create_string_op(f: Callable[[str], Any]) -> Callable[[Any], Any]:
    """Create a string operator that bridges other types."""
    return lambda x: f(str(x))

# --- Composition ---
def compose(f: Callable, g: Callable) -> Callable:
    """Function composition: compose(f, g)(x) = f(g(x))."""
    return lambda x: f(g(x))

def curry(f: Callable, *args, **kwargs) -> Callable:
    """Partial application for currying."""
    return partial(f, *args, **kwargs)
