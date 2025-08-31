import numpy as np
import math

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)

def quat_norm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / max(n, 1e-15)

def axis_angle_quat(axis_vec: np.ndarray, angle: float) -> np.ndarray:
    axis = axis_vec.astype(float)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    h = angle / 2.0
    s = math.sin(h)
    return np.array([math.cos(h), s * axis[0], s * axis[1], s * axis[2]], dtype=float)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    theta_0 = math.acos(dot)
    if theta_0 < 1e-12:
        return (1.0 - t) * q1 + t * q2
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q1 + s1 * q2

def shape_curvature(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] < 3:
        return np.zeros(vectors.shape[0], dtype=float)
    v_prev = vectors[:-2]
    v_curr = vectors[1:-1]
    v_next = vectors[2:]
    curvature_vectors = v_prev - 2.0 * v_curr + v_next
    curv = np.linalg.norm(curvature_vectors, axis=1)
    return np.pad(curv, (1, 1), 'constant')
