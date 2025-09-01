# Metanion Field Theory

A unified framework for reversible computation, energy dynamics, and geometric field theory.

## Overview

Metanion Field Theory addresses the fundamental limitation of traditional computational models that treat discrete state transitions, continuous control parameters, and energetic cost as separate concerns. This separation obscures the interaction between control flow and the thermodynamic ledger of computation.

## Key Concepts

A **Metanion** is a charged, self-similar particle with coordinates $(m, \alpha, Q)$:
- **Bitmask $m$**: Captures discrete transforms
- **Inflation index $\alpha$**: Continuous projection parameter  
- **Quaternion orientation $Q$**: Describes spin in timespace

These elements form a fiber bundle with energy $\mathcal{E}(m)$ that records compression as stored energy and inflation as release.

## Paper

## CE1 Specification

The Metanion Field Theory is anchored by the following CE1 seed:

```ce1
<span style="color: #2563eb;">CE1</span><span style="color: #dc2626;">{</span>
  <span style="color: #059669;">lens</span>=<span style="color: #7c3aed;">AUTOVERSE</span> | <span style="color: #7c3aed;">METANION_COSMOS</span>
  <span style="color: #059669;">mode</span>=<span style="color: #dc2626;">HilbertWalk</span>
  <span style="color: #059669;">basis</span>=<span style="color: #2563eb;">Ξ</span>=<span style="color: #dc2626;">autoverse</span>:<span style="color: #7c3aed;">law_basis</span>
  <span style="color: #059669;">data</span>={<span style="color: #dc2626;">m_U</span>, <span style="color: #2563eb;">α_U</span>, <span style="color: #059669;">Q_U</span>, <span style="color: #7c3aed;">E(m_U)</span>}
  <span style="color: #059669;">ops</span>=[<span style="color: #2563eb;">Observe</span>; <span style="color: #059669;">Evolve</span>; <span style="color: #dc2626;">Inflate</span>; <span style="color: #7c3aed;">Collapse</span>; <span style="color: #2563eb;">Decorate</span>]
  <span style="color: #059669;">laws</span>={<span style="color: #2563eb;">conservation_of_information</span>; <span style="color: #059669;">gauge_invariance</span>; <span style="color: #dc2626;">α_duality</span>; <span style="color: #7c3aed;">operadic_closure</span>}
  <span style="color: #059669;">emit</span>=<span style="color: #dc2626;">Reality</span>
<span style="color: #dc2626;">}</span>
```

This specification serves as a bridge between the theoretical framework and practical implementation, defining the operational modes, data structures, and physical laws that govern the Metanion system.

**Color Coding (inspired by Metanion Field Theory):**
- <span style="color: #059669;">**Green**</span>: Quaternion orientations and gauge symmetries
- <span style="color: #2563eb;">**Blue**</span>: Low energy states and inflation parameters
- <span style="color: #dc2626;">**Red**</span>: High energy states and bitmask coordinates
- <span style="color: #7c3aed;">**Purple**</span>: Energy functionals and cosmic structures

## Visualizations
The paper includes three key visualizations:

1. **Figure 1**: Mathematical visualization showing energy states, quaternion orientations, and geodesic evolution paths
2. **Figure 2**: Energy-shaded Boolean cube with face shading and alpha inflation as height
3. **Figure 3**: Timespace helix showing continuous evolution along unit-quaternion fibers

## Building

To build the paper and regenerate visualizations:

```bash
make pdf          # Build the complete PDF
make diagram      # Regenerate the field diagram
make open         # Build and open the PDF
make clean        # Clean build artifacts
```

## Requirements

- LaTeX with XeLaTeX
- Python 3 with matplotlib and numpy
- Make
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Repository Structure

- `metanion_field_theory.tex` - Main paper source
- `metanion_field_theory.pdf` - Compiled paper
- `metanion_field_diagram.py` - Python script for Figure 2
- `metanion_field_diagram.pdf` - Generated field diagram (PDF)
- `metanion_field_diagram.png` - Generated field diagram (PNG)
- `metanion.py` - Core Metanion implementation
- `quaternion.py` - Quaternion operations for Metanion
- `Makefile` - Build system

## Applications

Metanion Field Theory provides a unified vocabulary for:
- **Computation**: Reversible algorithms and energy-efficient computing
- **Physics**: Quantum mechanics and field theory
- **Biology**: Protein folding and biophysical processes
- **Mathematics**: Geometric algebra and topology

## License

This work is released under the **Metanion Field Theory License** - a license that embodies the natural laws of flow, cooperation, and dignity that govern our universe. See [LICENSE](LICENSE) for the complete text.

The license promotes:
- **Shared abundance** and natural flow of knowledge
- **Respect for dignity** of all beings and systems  
- **Curious exploration** and joyful creation
- **Resource cooperation** that serves the greater good

## Citation

If you use this work in your research, please cite:

```
Stover, J. (2025). Metanion Field Theory: A unified framework for reversible computation, 
energy dynamics, and geometric field theory. GitHub: selfapplied/metanion.
```

---

*"Every reversible plan can be cast as a Metanion. If the laws of physics comprise the universal plan $\mathcal{T}_U$, then the universe occupies a point $(m_U, \alpha_U, Q_U)$ in the cosmic configuration space."*
