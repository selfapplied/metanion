import dataclasses
from typing import Dict, Optional

import numpy as np

from genome import Genome
from aspire import stable_str_hash
import quaternion
from emits import em

from sprixel2 import gene


@dataclasses.dataclass
class BlockEvent:
    """A structured representation of a DEFLATE block event."""
    kind: str  # 'dynamic', 'static', 'stored'
    lit_len_code_lengths: Dict[int, int] = dataclasses.field(default_factory=dict)


@gene
@em("seed: int := axis: [3]")
def axis_from_id(seed: int) -> np.ndarray:
    """Creates a normalized 3D vector from a 32-bit integer."""
    rng = np.random.RandomState(seed)
    axis = rng.rand(3) * 2 - 1
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else np.array([0., 0., 1.])


@gene
@em("event: BlockEvent, genome: Genome := rotation: Optional[ndarray]")
def rotation_from_block(event: BlockEvent, genome: Genome) -> Optional[np.ndarray]:
    """
    Determines the quaternion rotation based on a dynamic block event and the
    genome's grammar. Returns the rotation quaternion, or None if no rotation
    should occur.
    """
    if event.kind != 'dynamic' or not event.lit_len_code_lengths:
        return None

    dominant_token, _ = max(
        event.lit_len_code_lengths.items(), key=lambda item: item[1])

    alleles = genome.grammar.semantic_alleles
    allele_bytes = alleles.get(dominant_token)

    if allele_bytes:
        # Use the learned, meaningful allele as the rotation axis.
        axis = np.frombuffer(allele_bytes, dtype=np.float32)
    else:
        # Fallback for rare tokens: hash the token itself
        shape_id = stable_str_hash(str(dominant_token))
        axis = axis_from_id(int(shape_id & 0xFFFFFFFF))

    ang = 0.2  # Dynamic block rotation angle
    return quaternion.axis_angle_quat(axis, ang)


@gene
@em("grammar: Grammar := shadowfold: Dict[str, float]")
def grammar_shadowfold(grammar: "Grammar") -> Dict[str, float]:
    """
    Calculates the conceptual 'shadowfold' of a grammar.

    This is a projection of the grammar's structure, revealing the creases
    and folds in its conceptual space. It is calculated by ordering the
    semantic alleles and then measuring the curvature of the resulting
    vector sequence. High values indicate a 'fold' where meaning shifts
    abruptly, akin to a crease in a cast shadow.
    """
    alleles = grammar.semantic_alleles
    if not alleles or len(alleles) < 3:
        return {}

    # Create a canonical ordering by sorting tokens alphabetically.
    sorted_tokens = sorted(alleles.keys())
    
    # Convert the allele bytes into a structured NumPy array.
    vector_list = [
        np.frombuffer(alleles[token], dtype=np.float32)
        for token in sorted_tokens
    ]
    vectors = np.array(vector_list)

    # Measure the curvature to find the intensity of the folds.
    intensities = quaternion.shape_curvature(vectors)

    return {token: intensity for token, intensity in zip(sorted_tokens, intensities)}
