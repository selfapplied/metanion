#!/usr/bin/env python3
"""
CE1 ZetaField Equations: Mathematical Diagram

Creates a focused visualization showing the key equations and relationships
in the ZetaField concept:

- ζ(s) is Green's function of multiplicative Laplacian Δ_mult = -t²∂²_t - t∂_t
- Primes act as delta boundary scatterers V(x) = ∑_p δ(x - log p)
- ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))
- CE1 mirror gives spectral symmetry Λ(s) = Λ(1-s)
- Zeros are eigenmodes, primes are boundary scatterers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
import os
import time


def create_zeta_field_equations_diagram(output_file: str = None) -> str:
    """
    Create a focused diagram showing ZetaField equations and relationships.
    """
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f".out/ce1_visualization/zeta_field_equations_{timestamp}.png"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#FAFAFA')
    
    # Colors
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple
        'accent': '#F18F01',       # Orange
        'axis': '#E74C3C',         # Red
        'green': '#27AE60',        # Green
        'text': '#2C3E50',         # Dark blue-gray
        'background': '#FAFAFA'    # Light gray
    }
    
    # Main title
    ax.text(0.5, 0.95, 'CE1 ZetaField: Green\'s Function of Multiplicative Laplacian', 
           ha='center', va='center', fontsize=22, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # Subtitle
    ax.text(0.5, 0.91, 'ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s)); primes as boundary scatterers', 
           ha='center', va='center', fontsize=14, style='italic', 
           color=colors['text'], transform=ax.transAxes)
    
    # 1. Multiplicative Group
    ax.text(0.15, 0.85, 'Multiplicative Group', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['primary'], transform=ax.transAxes)
    
    # Equation box
    group_box = FancyBboxPatch((0.05, 0.75), 0.2, 0.08, boxstyle="round,pad=0.02", 
                              facecolor=colors['primary'], alpha=0.2, 
                              edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(group_box)
    
    ax.text(0.15, 0.79, 'G = R_+^×', 
           ha='center', va='center', fontsize=14, fontweight='bold', 
           color=colors['primary'], transform=ax.transAxes)
    ax.text(0.15, 0.75, 'x = log t; Mellin = Fourier(x)', 
           ha='center', va='center', fontsize=10, 
           color=colors['primary'], transform=ax.transAxes)
    
    # Visual representation - coordinate transformation
    t_values = np.linspace(0.1, 5, 50)
    x_values = np.log(t_values)
    
    # Scale and offset for plotting
    x_offset = 0.05
    y_offset = 0.65
    scale = 0.2
    
    ax.plot(x_offset + scale * (t_values / 5), y_offset + scale * (x_values / 2), 
           '-', color=colors['primary'], linewidth=3, label='x = log t')
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Coordinate\nTransformation', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 2. Multiplicative Laplacian
    ax.text(0.5, 0.85, 'Multiplicative Laplacian', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Equation box
    laplacian_box = FancyBboxPatch((0.35, 0.70), 0.3, 0.12, boxstyle="round,pad=0.02", 
                                  facecolor=colors['secondary'], alpha=0.2, 
                                  edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(laplacian_box)
    
    ax.text(0.5, 0.78, 'Δ_mult = -t²∂²_t - t∂_t', 
           ha='center', va='center', fontsize=14, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    ax.text(0.5, 0.74, 'Eigenfunctions: t^{s-1}', 
           ha='center', va='center', fontsize=12, 
           color=colors['secondary'], transform=ax.transAxes)
    ax.text(0.5, 0.70, 'Eigenvalues: s(1-s)', 
           ha='center', va='center', fontsize=12, 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Visual representation - eigenfunctions
    x_offset = 0.35
    y_offset = 0.55
    scale = 0.3
    
    # Plot eigenfunctions for different s
    s_values = [0.5, 0.5 + 1j, 0.5 + 2j]
    colors_eigen = [colors['primary'], colors['accent'], colors['green']]
    
    for i, s in enumerate(s_values):
        t_plot = np.linspace(0, 1, 30)
        eigenfunction = np.power(t_plot + 0.1, s - 1)
        
        # Plot real part
        ax.plot(x_offset + scale * t_plot, y_offset + scale * np.real(eigenfunction) / 2, 
               '-', color=colors_eigen[i], linewidth=2, alpha=0.8)
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Eigenfunctions\nt^{s-1}', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 3. Prime Pins
    ax.text(0.85, 0.85, 'Prime Pins', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['green'], transform=ax.transAxes)
    
    # Equation box
    pins_box = FancyBboxPatch((0.75, 0.70), 0.2, 0.12, boxstyle="round,pad=0.02", 
                             facecolor=colors['green'], alpha=0.2, 
                             edgecolor=colors['green'], linewidth=2)
    ax.add_patch(pins_box)
    
    ax.text(0.85, 0.78, 'V(x) = ∑_p δ(x - log p)', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['green'], transform=ax.transAxes)
    ax.text(0.85, 0.74, 'Boundary Scatterers', 
           ha='center', va='center', fontsize=10, 
           color=colors['green'], transform=ax.transAxes)
    
    # Visual representation - delta functions
    x_offset = 0.75
    y_offset = 0.55
    scale = 0.2
    
    # Show delta functions at log p
    primes = [2, 3, 5, 7, 11]
    for i, p in enumerate(primes):
        log_p = np.log(p)
        x_pos = x_offset + (log_p / 3.0) * scale
        y_pos = y_offset + scale * 0.5
        
        # Delta function
        ax.plot([x_pos, x_pos], [y_pos, y_pos + 0.1], 
               color=colors['green'], linewidth=3)
        ax.text(x_pos, y_pos - 0.02, f'p={p}', ha='center', va='top', 
               fontsize=8, color=colors['green'], fontweight='bold')
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Delta Functions\nat log p', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 4. Green's Function
    ax.text(0.15, 0.45, 'Green\'s Function', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['accent'], transform=ax.transAxes)
    
    # Equation box
    green_box = FancyBboxPatch((0.05, 0.30), 0.2, 0.12, boxstyle="round,pad=0.02", 
                              facecolor=colors['accent'], alpha=0.2, 
                              edgecolor=colors['accent'], linewidth=2)
    ax.add_patch(green_box)
    
    ax.text(0.15, 0.38, 'ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))', 
           ha='center', va='center', fontsize=10, fontweight='bold', 
           color=colors['accent'], transform=ax.transAxes)
    ax.text(0.15, 0.34, 'Zeta as Green\'s Function', 
           ha='center', va='center', fontsize=10, 
           color=colors['accent'], transform=ax.transAxes)
    
    # Visual representation - Green's function
    x_offset = 0.05
    y_offset = 0.15
    scale = 0.2
    
    # Create complex plane visualization
    sigma = np.linspace(0, 1, 20)
    t = np.linspace(0, 20, 20)
    S, T = np.meshgrid(sigma, t)
    
    # Simplified zeta magnitude
    zeta_mag = 1.0 / (np.abs(S - 0.5) + 0.1)
    
    im = ax.imshow(zeta_mag, extent=[x_offset, x_offset + scale, y_offset, y_offset + scale], 
                  origin='lower', cmap='viridis', alpha=0.7)
    
    # Critical line
    ax.plot([x_offset, x_offset + scale], [y_offset + scale/2, y_offset + scale/2], 
           color=colors['axis'], linewidth=2)
    
    # Zeta zero
    ax.plot(x_offset + scale/2, y_offset + scale/2, 'o', color=colors['accent'], 
           markersize=6, markeredgecolor='white', markeredgewidth=1)
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Green\'s Function\nζ(s)', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 5. Mirror Symmetry
    ax.text(0.5, 0.45, 'Mirror Symmetry', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Equation box
    mirror_box = FancyBboxPatch((0.35, 0.30), 0.3, 0.12, boxstyle="round,pad=0.02", 
                               facecolor=colors['secondary'], alpha=0.2, 
                               edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(mirror_box)
    
    ax.text(0.5, 0.38, 'Λ(s) = Λ(1-s)', 
           ha='center', va='center', fontsize=14, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    ax.text(0.5, 0.34, 'CE1 mirror gives spectral symmetry', 
           ha='center', va='center', fontsize=10, 
           color=colors['secondary'], transform=ax.transAxes)
    ax.text(0.5, 0.30, 'axis = Re s = 1/2', 
           ha='center', va='center', fontsize=10, 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Visual representation - reflection
    x_offset = 0.35
    y_offset = 0.15
    scale = 0.3
    
    s_values = np.linspace(0, 1, 50)
    ax.plot(x_offset + scale * s_values, y_offset + scale * s_values, 
           '-', color=colors['axis'], linewidth=3, alpha=0.8)
    ax.plot(x_offset + scale * s_values, y_offset + scale * (1 - s_values), 
           '--', color=colors['secondary'], linewidth=3, alpha=0.8)
    
    # Fixed point
    ax.plot(x_offset + scale * 0.5, y_offset + scale * 0.5, 'o', color=colors['axis'], 
           markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Spectral Symmetry\ns ↔ 1-s', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # 6. Attractor Eigenmodes
    ax.text(0.85, 0.45, 'Attractor Eigenmodes', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # Equation box
    eigenmodes_box = FancyBboxPatch((0.75, 0.30), 0.2, 0.12, boxstyle="round,pad=0.02", 
                                   facecolor=colors['text'], alpha=0.2, 
                                   edgecolor=colors['text'], linewidth=2)
    ax.add_patch(eigenmodes_box)
    
    ax.text(0.85, 0.38, 'Zeros = Eigenmodes', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    ax.text(0.85, 0.34, 'Primes = Boundary', 
           ha='center', va='center', fontsize=10, 
           color=colors['text'], transform=ax.transAxes)
    ax.text(0.85, 0.30, 'Scatterers', 
           ha='center', va='center', fontsize=10, 
           color=colors['text'], transform=ax.transAxes)
    
    # Visual representation - eigenmodes
    x_offset = 0.75
    y_offset = 0.15
    scale = 0.2
    
    # Create eigenmode visualization
    modes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    for i, mode in enumerate(modes):
        ax.bar(x_offset + i * 0.04, y_offset + scale * mode, 0.03, 
              color=colors['text'], alpha=0.7)
    
    # Highlight dominant mode
    ax.bar(x_offset, y_offset + scale * modes[0], 0.03, 
          color=colors['accent'], alpha=0.8)
    
    ax.text(x_offset + scale/2, y_offset - 0.05, 'Eigenmode\nSpectrum', 
           ha='center', va='top', fontsize=10, color=colors['text'], transform=ax.transAxes)
    
    # Add connecting arrows
    # From group to Laplacian
    ax.annotate('', xy=(0.35, 0.76), xytext=(0.25, 0.76),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From Laplacian to pins
    ax.annotate('', xy=(0.75, 0.76), xytext=(0.65, 0.76),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From pins to Green's function
    ax.annotate('', xy=(0.15, 0.36), xytext=(0.25, 0.36),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From Green's function to mirror
    ax.annotate('', xy=(0.35, 0.36), xytext=(0.25, 0.36),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From mirror to eigenmodes
    ax.annotate('', xy=(0.75, 0.36), xytext=(0.65, 0.36),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add footer
    ax.text(0.5, 0.02, 'ZetaField: ζ(s) is Green\'s function of multiplicative Laplacian with primes as boundary scatterers', 
           ha='center', va='center', fontsize=12, style='italic', 
           color=colors['text'], transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
               facecolor='#FAFAFA', edgecolor='none')
    plt.close()
    
    return output_file


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CE1 ZetaField Equations")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create equations diagram
    output_file = create_zeta_field_equations_diagram(args.output)
    print(f"Generated ZetaField equations diagram: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
