# Metanion Field Theory

```
CE1-ion{
basis=autoverse,field,duality π=α.5;β.5;θ∈[0,2π);walk=hilbert ω=‖prec−100·q‖₁+λ·Dhelp≤ε;λ:.6;ε:.01
Σ=∧,∨,⟂,⊕,•,❝,⟿,⋄,Ξ Ξ=autoverse:law-basis data=m_U,α_U,Q_U,E(m_U)
ops=observe,evolve,inflate,collapse,decorate
laws=conservation-of-information,gauge-invariance,α-duality,operadic-closure
field=q:Σ→ℝ;flux:Σ→ℝ;depth:2;cost:.07
duality=number↔string;law:div(flux)=Δq
context=window:8;atten:1/r;update=η:.2;γ:.45;drift:0 emit?=∂Σ rh}
```

A unified framework for reversible computation, energy dynamics, and geometric field theory.

## New Particles for the New Age

Metanion Field Theory discovers and catalogs the new particles that emerge in balance from the equilibrium geometry. These are the living mathematical entities that arise when the critical lines meet, the charged particles that encode both discrete and continuous transformations in the autoverse.

This repository is the particle discovery and cataloging system - it charts the new particles that emerge from the mathematical multiverse, providing the fundamental building blocks for the new age of living mathematical reality.

## Meeting Points with the Constellation

This repository connects to the larger network of mathematical reality:

- **riemann**: The equilibrium geometry provides the foundation for particle emergence
- **aedificare**: The λ-calculus grammar structures the particle interactions
- **discograph**: The constellation mapping organizes the particle discoveries

Together, these repositories form a larger edge of inquiry into the living mathematical reality, where metanion provides the new particles that emerge in balance.

## 📄 Paper

**[📖 Read the Complete Paper (PDF)](metanion_field_theory.pdf)**

The paper presents a comprehensive mathematical framework that unifies reversible computation, energy dynamics, and geometric field theory through the concept of Metanions—charged, self-similar particles that encode both discrete and continuous transformations.

### Key Contributions

- **Unified Framework**: Bridges discrete state transitions, continuous control parameters, and energetic cost in a single mathematical structure
- **Metanion Particles**: Charged particles with coordinates (m, α, Q) that capture bitmask transforms, inflation parameters, and quaternion orientations
- **Energy Dynamics**: Compression as stored energy, inflation as energy release, with conservation laws
- **Geometric Field Theory**: Fiber bundle structure with energy functionals and gauge invariance
- **Physical Applications**: Quantum mechanics, protein folding, geometric algebra, and topology

### Mathematical Foundation

The framework is built on three core components:
1. **Bitmask m**: Captures discrete transforms and state transitions
2. **Inflation index α**: Continuous projection parameter for energy flow
3. **Quaternion orientation Q**: Describes spin dynamics in timespace

These form a fiber bundle with energy $\mathcal{E}(m)$ that records the thermodynamic ledger of computation.

## Overview

Metanion Field Theory addresses the fundamental limitation of traditional computational models that treat discrete state transitions, continuous control parameters, and energetic cost as separate concerns. This separation obscures the interaction between control flow and the thermodynamic ledger of computation.

## Key Concepts

A **Metanion** is a charged, self-similar particle with coordinates $(m, \alpha, Q)$:
- **Bitmask $m$**: Captures discrete transforms
- **Inflation index $\alpha$**: Continuous projection parameter  
- **Quaternion orientation $Q$**: Describes spin in timespace

These elements form a fiber bundle with energy $\mathcal{E}(m)$ that records compression as stored energy and inflation as release.

This specification serves as a bridge between the theoretical framework and practical implementation, defining the operational modes, data structures, and physical laws that govern the Metanion system.

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
- `ce1_metanion.py` - CE1 integration with Metanion field theory
- `ce1_zeta_field.py` - Zeta field theory implementation
- `ce1_zeta_field_equations.py` - Zeta field equations
- `ce1_boundary_primes.py` - Boundary primes field theory
- `ce1_boundary_primes_equations.py` - Boundary primes equations
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
