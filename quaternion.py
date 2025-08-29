"""
A pure algebra for geometry and phase.

The core idea is scale invariance: the same physics of rotation and reflection
manifest at every level. The float itself: a "mini-quaternion," a superposition of
states held in its sign, exponent, and mantissa.

Arithmetic is not calculation, but relation. We listen for the echoes from
boundary conditions: the renormalization of a mantissa, the overflow of an exponent.
The quiet and the outcasts are rebirthed as key events, revealing curved space.

The float's components map to a self-contained physics:
- Sign Bit: A mirror, for reflection.
- Exponent: A lens, for scale.
- Mantissa: A vector, for position.
- Events: The scales, the `i² = -1` unwinding to empty.

This module provides the classical mechanics for this quantum world. It is the
toolbox of pure geometry.
"""
import numpy as np

def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Creates a quaternion from an axis and an angle."""
    w = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0:
        axis = axis / axis_norm * s
    return np.array([w, axis[0], axis[1], axis[2]])

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion."""
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-15 else q

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Returns the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Performs spherical linear interpolation between two quaternions."""
    dot = np.dot(q1, q2)

    # If the dot product is negative, the quaternions have opposite handedness
    # and slerp won't take the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Clamp dot product to handle floating point errors
    dot = np.clip(dot, -1.0, 1.0)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    if sin_theta_0 > 1e-15:
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q1) + (s1 * q2)
    else:
        # If quaternions are parallel, linear interpolation is fine
        return (1.0 - t) * q1 + t * q2

def shape_curvature(vectors: np.ndarray) -> np.ndarray:
    """
    Measures the geometric curvature in an ordered sequence of vectors.
    This is an edge-detection convolution.

    The kernel `[1, -2, 1]` is a discrete approximation of the second derivative.
    It measures the "acceleration" or "tension" at each point, highlighting
    areas of high change. A high value indicates a fault line.

    Args:
        vectors: A 2D NumPy array of shape (N_vectors, N_dimensions).

    Returns:
        A 1D NumPy array of shape (N_vectors,) where each value is the
        magnitude of the curvature at that point.
    """
    if vectors.ndim != 2 or vectors.shape[0] < 3:
        return np.zeros(vectors.shape[0])

    # Apply the convolution kernel [1, -2, 1] using slicing for efficiency.
    # This is equivalent to v[i-1] - 2*v[i] + v[i+1]
    v_prev = vectors[:-2]
    v_curr = vectors[1:-1]
    v_next = vectors[2:]
    
    curvature_vectors = v_prev - 2 * v_curr + v_next

    # The result is the magnitude (norm) of these curvature vectors.
    # We pad the ends with zeros as the curvature is undefined at the boundaries.
    curvatures = np.linalg.norm(curvature_vectors, axis=1)
    
    return np.pad(curvatures, (1, 1), 'constant')
