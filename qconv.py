"""
Quaternion Convolutional Grammar (scaffolding)

Core ops (no IO):
- measure(payload) -> measurements  # histograms, matches, codebooks, q_energy
- make_bases() -> {seq, tree, s3}   # Kravchuk/LapSpec/WignerD placeholders
- fit_kernels(measurements, bases) -> kernels
- walk(payload, kernels) -> responses {seq, tree, s3}, localizations
- reconcile(primal, dual, responses) -> delta (edits/suggestions)

Transport across blocks:
  K_next = U_k K U_k^{-1}, with U_k = exp(-i H_k) captured by codebook rotations

This is a tight, testable surface you can plug into CE1 flow.
Everything returns plain dicts/arrays; no printing and no files.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from collections import namedtuple

from quaternion import quat_mul, quat_norm


SeqKernel = namedtuple('SeqKernel', 'taps')  # taps: np.ndarray (L,4)


TreeKernel = namedtuple('TreeKernel', 'up down lateral')  # each: np.ndarray (4,)


S3Kernel = namedtuple('S3Kernel', 'bands')  # bands: np.ndarray


# Core typed surfaces
Kernels = namedtuple('Kernels', 'seq tree s3')
Responses = namedtuple('Responses', 'R_seq argmax_seq R_tree R_s3 argmax_s3')
Measurements = namedtuple('Measurements', 'seq_len tree_nodes blocks_q s3_curvature')


Payload = namedtuple('Payload', 'seq tree_nodes blocks_q hist codebooks')
QRoute = namedtuple('QRoute', 'w_seq w_tree w_s3 mode')
RouteResult = namedtuple('RouteResult', 'payload measurements qroute kernels')


def _q(arr_like: Any) -> np.ndarray:
    a = np.array(arr_like, dtype=float)
    if a.shape[-1] != 4:
        raise ValueError('quaternion must be [...,4]')
    return a


def _qconj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def _transport(qU: np.ndarray, qK: np.ndarray) -> np.ndarray:
    """Transport kernel tap qK by unitary rotation qU: qU * qK * qU^{-1}."""
    return quat_mul(quat_mul(qU, qK), _qconj(qU))


def _coerce_payload(obj: Any) -> Payload:
    if isinstance(obj, Payload.__class__) or isinstance(obj, tuple):
        # assume already a Payload-like tuple
        try:
            return Payload(seq=getattr(obj, 'seq', None),
                           tree_nodes=getattr(obj, 'tree_nodes', 0),
                           blocks_q=getattr(obj, 'blocks_q', None),
                           hist=getattr(obj, 'hist', None),
                           codebooks=getattr(obj, 'codebooks', None))
        except Exception:
            pass
    if isinstance(obj, dict):
        return Payload(seq=obj.get('seq'),
                       tree_nodes=int(obj.get('tree_nodes') or 0),
                       blocks_q=obj.get('blocks_q'),
                       hist=obj.get('hist'),
                       codebooks=obj.get('codebooks'))
    # tuple/list by arity: (seq), (seq, tree_nodes), (seq, tree_nodes, blocks_q), ...
    if isinstance(obj, (list, tuple)):
        vals = list(obj)
        fields = ['seq', 'tree_nodes', 'blocks_q', 'hist', 'codebooks']
        # pad/truncate to length 5
        vals = (vals + [None] * 5)[:5]
        return Payload(seq=vals[0],
                       tree_nodes=int(vals[1] or 0),
                       blocks_q=vals[2],
                       hist=vals[3],
                       codebooks=vals[4])
    # fallback: empty
    return Payload(seq=None, tree_nodes=0, blocks_q=None, hist=None, codebooks=None)


def measure(payload: Any) -> Measurements:
    """Minimal, typed measurement extraction (𝕄)."""
    pay = _coerce_payload(payload)
    arr = np.asarray(pay.seq) if pay.seq is not None else np.array([], dtype=int)
    seq_len = int(arr.size)
    tree_nodes = int(pay.tree_nodes or 0)
    blocks_q_arr, s3_curvature = None, 0.0
    Q_obj = pay.blocks_q
    if Q_obj is not None:
        Q = np.asarray(Q_obj, dtype=float)
        if Q.ndim == 2 and Q.shape[1] == 4 and len(Q):
            QQ = np.array([quat_norm(q) for q in Q])
            blocks_q_arr = QQ
            if len(QQ) > 1:
                dqs = np.array([quat_mul(_qconj(QQ[i - 1]), QQ[i]) for i in range(1, len(QQ))])
                ang = 2 * np.arccos(np.clip(dqs[:, 0], -1.0, 1.0))
                s3_curvature = float(np.sum(np.abs(ang)))
    return Measurements(seq_len=seq_len, tree_nodes=tree_nodes, blocks_q=blocks_q_arr, s3_curvature=s3_curvature)


def fit_kernels(measurements: Measurements) -> Kernels:
    """Tiny init macro: make a small real-tap seq kernel; axis tags for tree; flat s3 bands."""
    L = int(min(8, max(3, (measurements.seq_len or 16) // 8)))
    K_seq = SeqKernel(taps=np.pad(np.array([[1.0, 0.0, 0.0, 0.0]]), ((0, L-1), (0, 0))))
    K_tree = TreeKernel(up=_q([1.0, 0.1, 0.0, 0.0]), down=_q([1.0, 0.0, 0.1, 0.0]), lateral=_q([1.0, 0.0, 0.0, 0.1]))
    K_s3 = S3Kernel(bands=np.ones((4,), dtype=float))
    return Kernels(seq=K_seq, tree=K_tree, s3=K_s3)


def walk(payload: Dict[str, Any], kernels: Kernels) -> Responses:
    """Convolution (⋆) over seq/tree/S³; returns Responses with argmax."""
    R_seq = None
    argmax_seq = 0
    R_tree = None
    R_s3 = None
    argmax_s3 = 0
    # Sequence: tap real parts over sequence length as simple response
    seq = payload.get('seq')
    if seq is not None:
        X = np.array(seq, dtype=float)
        taps = kernels.seq.taps
        # crude conv: correlate counts window with tap sum
        tap_mag = float(np.sum(np.abs(taps)))
        win = min(len(X), max(1, taps.shape[0]))
        if win > 0:
            # sliding sum
            csum = np.convolve(np.ones_like(X), np.ones(win), mode='same')
            R_seq = (csum * tap_mag).astype(float)
            argmax_seq = int(np.argmax(R_seq))

    # Tree: simple directional counts proxy
    if 'tree_nodes' in payload:
        n = float(int(payload.get('tree_nodes', 0)))
        k = kernels.tree
        R_tree = np.array([k.up[0]*n, k.down[0]*n, k.lateral[0]*n], dtype=float)

    # S^3: measure alignment to first block quaternion
    Q = payload.get('blocks_q')
    if Q is not None:
        QQ = np.array(Q, dtype=float)
        if QQ.ndim == 2 and QQ.shape[1] == 4 and len(QQ) >= 1:
            q0 = QQ[0]
            # align successive dq angles as response
            dqs = np.array([quat_mul(_qconj(q0), QQ[i]) for i in range(1, len(QQ))])
            ang = 2 * np.arccos(np.clip(dqs[:, 0], -1.0, 1.0))
            R_s3 = np.abs(ang).astype(float)
            argmax_s3 = int(np.argmax(R_s3)) if len(R_s3) else 0

    return Responses(R_seq=R_seq, argmax_seq=argmax_seq, R_tree=R_tree, R_s3=R_s3, argmax_s3=argmax_s3)


def transport_kernels(kernels: Kernels, U: np.ndarray) -> Kernels:
    """Transport (𝕌_U) a kernel pack by unit quaternion U."""
    U = quat_norm(np.array(U, dtype=float))
    taps = kernels.seq.taps.copy()
    for i in range(len(taps)):
        taps[i] = _transport(U, taps[i])
    out_seq = SeqKernel(taps=taps)
    kt = kernels.tree
    out_tree = TreeKernel(up=_transport(U, kt.up),
                          down=_transport(U, kt.down),
                          lateral=_transport(U, kt.lateral))
    out_s3 = kernels.s3
    return Kernels(seq=out_seq, tree=out_tree, s3=out_s3)


def reconcile(primal: Dict[str, Any], dual: Dict[str, Any], responses: Responses) -> List[Tuple[str, str]]:
    """Reconcile (ℛ): return minimal (action, why) tuples from responses."""
    delta: List[Tuple[str, str]] = []
    if responses.R_tree is not None and responses.R_tree[0] > responses.R_tree[1] * 1.2:
        delta.append(('insert_adapter', 'missing-arg signature'))
    if responses.R_s3 is not None and np.max(responses.R_s3) > 0.5:
        delta.append(('rename', 'phase misalignment'))
    return delta


def route(payload: Any, kernels: Optional[Kernels] = None, *, transport: bool = True) -> RouteResult:
    """Unified router ρ+𝕌: type→energy→alignment."""
    p = _coerce_payload(payload)
    m = measure(p)
    qr = quaternion_router(m)
    k_out = transport_kernels(kernels, m.blocks_q[0]) if (transport and kernels is not None and m.blocks_q is not None and len(m.blocks_q) > 0) else kernels
    return RouteResult(payload=p, measurements=m, qroute=qr, kernels=k_out)


def quaternion_router(measurements: Measurements) -> QRoute:
    """Route (ρ): weights from average quaternion vector (|x|,|y|,|z|)."""
    Q = measurements.blocks_q
    if Q is None or len(Q) == 0:
        return QRoute(w_seq=1/3, w_tree=1/3, w_s3=1/3, mode='neutral')
    vec = np.abs(np.mean(Q[:, 1:4], axis=0))  # (x,y,z)
    s = float(np.sum(vec)) or 1.0
    x, y, z = (float(vec[0]/s), float(vec[1]/s), float(vec[2]/s))
    return QRoute(w_seq=x, w_tree=y, w_s3=z, mode='vector')


def route2(payload: Any, kernels: Optional[Kernels] = None, *, transport: bool = True) -> RouteResult:
    """Unified router: type → energy → alignment. (internal alias)"""
    p = _coerce_payload(payload)
    m = measure(p)
    qr = quaternion_router(m)
    k_out = kernels
    if transport and kernels is not None and m.blocks_q is not None and len(m.blocks_q) > 0:
        k_out = transport_kernels(kernels, m.blocks_q[0])
    return RouteResult(payload=p, measurements=m, qroute=qr, kernels=k_out)
