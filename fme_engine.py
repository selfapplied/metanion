from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import random
from quaternion import quat_mul, quat_norm, axis_angle_quat
import fme_color
from collections import namedtuple

EngineResult = namedtuple('EngineResult', 'banks final_q color_trail boundary_events')


def _emit_amp(engine, symbol: str) -> float:
    lo, hi = engine.emissions.get(symbol, {}).get('amplitude', [0.4, 0.9])
    return random.uniform(float(lo), float(hi))

def step(engine, level: int, q: np.ndarray, state_symbol: str):
    i = engine.symbol_to_idx[state_symbol]
    P = engine.transitions_per_level[level][i]
    amp = _emit_amp(engine, state_symbol)
    mu = (1.0 - engine.alpha) * engine.mu_rho[level] + engine.alpha * amp
    engine.mu_rho[level] = mu
    Delta = amp - mu
    x, y, z = q[1], q[2], q[3]
    m = 1.0 if abs(Delta) > abs(x) else 0.0
    chi = 1.0 if Delta >= 0 else -1.0
    e_vec = chi * m * np.array([x, y, z]) + (1.0 - m) * np.array([0.0, 0.0, 1.0])
    r_hat = axis_angle_quat(e_vec, engine.epsilon0 * abs(Delta) * engine.mode)
    q_rot = quat_mul(q, r_hat)
    q_new = quat_norm(q_rot)
    e_norm = float(np.linalg.norm(e_vec))
    e_dir = e_vec / max(e_norm, 1e-9) if e_norm > 1e-9 else np.array([0.0, 0.0, 0.0])
    q_inj = q_rot + engine.kappa * abs(Delta) * np.array([0.0, e_dir[0], e_dir[1], e_dir[2]])
    maxc_raw = float(np.max(np.abs(q_inj)))
    q_new = q_rot
    spill = float(np.dot(q_new, q_new) - 1.0)

    if abs(spill) > 0.01:
        q_new = quat_norm(q_new)
    engine.ledger[level] += spill
    e_dir2 = e_vec / max(float(np.linalg.norm(e_vec)), 1e-9) if float(np.linalg.norm(e_vec)) > 1e-9 else np.array([0.0, 0.0, 0.0])
    alignment = engine.axis_dirs @ e_dir2
    enhanced_alignment = engine.percept(alignment)
    bias = engine.beta * Delta * enhanced_alignment[:len(P)] if len(enhanced_alignment) > len(P) else engine.beta * Delta * enhanced_alignment
    logits = np.log(np.maximum(P, 1e-9)) + bias / max(abs(x), 1e-9)
    logits -= np.max(logits)
    probs = np.exp(logits); probs /= probs.sum()
    j = np.random.choice(engine.S, p=probs)
    symbol = engine.idx_to_symbol[j]
    alleles = []
    if maxc_raw > 1.0 or abs(Delta) > engine.delta_spawn:
        alleles.append(q_inj.copy())

    if abs(Delta) > engine.delta_spawn * 0.1:
        tint_strength = 0.001 * abs(Delta)
        q_new = q_new + tint_strength * q_inj
        q_new = quat_norm(q_new)

    # --- EMA (quaternion fill-up) over vector part ---
    ema_alpha = getattr(engine, 'ema_alpha', 0.1)
    if not hasattr(engine, 'q_ema') or engine.q_ema is None:
        engine.q_ema = [np.zeros(3) for _ in range(engine.levels)]
    ema_prev = engine.q_ema[level]
    vec_part = np.array([q_new[1], q_new[2], q_new[3]])
    ema_now = (1.0 - float(ema_alpha)) * ema_prev + float(ema_alpha) * vec_part
    engine.q_ema[level] = ema_now

    # Fill metric: how much the quaternion "crystallized" (norm of EMA of vector)
    fill_metric = float(np.linalg.norm(ema_now))
    if not hasattr(engine, 'q_fill') or engine.q_fill is None:
        engine.q_fill = [0.0 for _ in range(engine.levels)]
    engine.q_fill[level] = fill_metric

    # Boundary phase detection
    color_info = fme_color.delta_to_color(Delta)
    boundary_events = []
    boundary_delta = getattr(engine, 'boundary_delta', max(getattr(engine, 'delta_spawn', 0.05) * 2.0, 0.05))
    boundary_theta = getattr(engine, 'boundary_theta', 0.85)
    is_boundary = (abs(float(Delta)) > float(boundary_delta)) or (fill_metric > float(boundary_theta))
    if is_boundary:
        evt = {'level': level, 'phase': color_info, 'fill': fill_metric, 'delta': float(Delta)}
        boundary_events.append(evt)
        # Optional hook for live grammar switching
        onb = getattr(engine, 'on_boundary', None)
        if callable(onb):
            try:
                onb(level, color_info)
            except Exception:
                pass
    return symbol, q_new, alleles, color_info, boundary_events

def run(engine, steps=64) -> EngineResult:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    symbol = 'drift'
    bank = [[] for _ in range(engine.levels)]
    finals = []
    color_trail = []
    boundary_events: List[Dict[str, Any]] = []
    for k in range(engine.levels):
        qk = q.copy()
        symk = symbol
        level_colors = []
        for _ in range(steps):
            symk, qk, born, color_info, be = step(engine, k, qk, symk)
            if born:
                bank[k].extend(born)
            level_colors.append(color_info)
            if be:
                boundary_events.extend(be)
        finals.append(qk.copy())
        q = qk
        color_trail.append(level_colors)
    return EngineResult(banks=bank, final_q=finals, color_trail=color_trail, boundary_events=boundary_events)
