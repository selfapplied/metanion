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

The complete theoretical framework is presented in:
- **[Metanion Field Theory (PDF)](metanion_field_theory.pdf)** - The main paper
- **[LaTeX Source](metanion_field_theory.tex)** - Source code for the paper

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
