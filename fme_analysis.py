from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_color_phases_clustering(engine, color_samples: List[Tuple[float, float, float, float]], n_clusters: int = 8) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """
    Perform cluster analysis on color samples to discover color phase alleles.
    """
    if len(color_samples) < n_clusters:
        return [], np.array([])

    features = []
    for delta, h, s, b in color_samples:
        delta_mag = abs(delta)
        features.append([h, s, b, delta_mag])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=engine.rng_mod)
    clusters = kmeans.fit_predict(features_scaled)

    color_phases = []
    cluster_centers = kmeans.cluster_centers_

    for i in range(n_clusters):
        cluster_samples = [color_samples[j] for j in range(len(color_samples)) if clusters[j] == i]
        if not cluster_samples:
            continue

        deltas = [s[0] for s in cluster_samples]
        hues = [s[1] for s in cluster_samples]
        saturations = [s[2] for s in cluster_samples]
        brightnesses = [s[3] for s in cluster_samples]

        phase_allele = {
            'cluster_id': i,
            'hue_center': float(np.mean(hues)),
            'hue_range': float(np.std(hues)),
            'sat_center': float(np.mean(saturations)),
            'sat_range': float(np.std(saturations)),
            'bright_center': float(np.mean(brightnesses)),
            'bright_range': float(np.std(brightnesses)),
            'delta_center': float(np.mean(deltas)),
            'delta_range': float(np.std(deltas)),
            'sample_count': len(cluster_samples),
            'cluster_weight': len(cluster_samples) / len(color_samples)
        }
        color_phases.append(phase_allele)

    return color_phases, cluster_centers

def get_color_phase_statistics(engine) -> Dict[str, Any]:
    """Get statistics about learned color phase alleles"""
    if not engine.color_phase_alleles:
        return {'total_phases': 0}

    stats = {
        'total_phases': len(engine.color_phase_alleles),
        'phases': []
    }

    for phase in engine.color_phase_alleles:
        phase_stats = {
            'id': phase.get('cluster_id', 0),
            'sample_count': phase.get('sample_count', 0),
            'cluster_weight': phase.get('cluster_weight', 0),
            'delta_range': ".3f",
            'hue_range': ".1f",
            'color_center': ".1f"
        }
        stats['phases'].append(phase_stats)

    return stats

def demonstrate_color_phase_transformation(engine, test_deltas: Optional[List[float]] = None) -> str:
    """
    Demonstrate color phase transformations for geometric template conversion.
    """
    if not engine.color_phase_alleles:
        return "No color phase alleles available for demonstration"

    if test_deltas is None:
        test_deltas = [0.1, 0.5, -0.2, 1.0, -0.8]

    demonstration = ["🎨 Color Phase Transformation Demonstration", "=" * 50]

    for delta in test_deltas:
        std_hue, std_sat, std_bright = engine._delta_to_color(delta)
        phase_hue, phase_sat, phase_bright = engine.transform_color_between_geometries(delta)

        demonstration.append(f"\nΔ = {delta:.3f}:")
        demonstration.append(f"  Standard:  H={std_hue:.1f}° S={std_sat:.3f} B={std_bright:.3f}")
        demonstration.append(f"  Phase-X:   H={phase_hue:.1f}° S={phase_sat:.3f} B={phase_bright:.3f}")

        hue_diff = abs(phase_hue - std_hue)
        sat_diff = abs(phase_sat - std_sat)
        bright_diff = abs(phase_bright - std_bright)
        demonstration.append(f"  ΔTransform: H±{hue_diff:.1f}° S±{sat_diff:.3f} B±{bright_diff:.3f}")

    return '\n'.join(demonstration)
