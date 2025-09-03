#!/usr/bin/env python3
"""
CE1 BoundaryPrimes Equations: Mathematical Diagram

Creates a focused visualization showing the key equations and relationships
in the BoundaryPrimes concept:

- Each prime p defines delta boundary at log p
- ζ(s) = ∏(1-p^{-s})^{-1} solves multiplicative Laplace equation
- s↔1-s symmetry with axis = critical line
- Zeros as eigenmodes under prime boundary constraints
- Integer flow determined by prime placement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
import os
import time


def create_boundary_primes_equations_diagram(output_file: str = None) -> str:
    """
    Create a focused diagram showing BoundaryPrimes equations and relationships.
    """
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f".out/ce1_visualization/boundary_primes_equations_{timestamp}.png"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor('#FAFAFA')
    
    # Colors
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple
        'accent': '#F18F01',       # Orange
        'axis': '#E74C3C',         # Red
        'boundary': '#27AE60',     # Green
        'text': '#2C3E50',         # Dark blue-gray
        'background': '#FAFAFA'    # Light gray
    }
    
    # Main title
    ax.text(0.5, 0.95, 'CE1 BoundaryPrimes: Prime Boundaries and Multiplicative Laplace Equation', 
           ha='center', va='center', fontsize=20, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # 1. Prime Boundaries
    ax.text(0.15, 0.85, 'Prime Boundaries', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    
    # Equation box
    boundary_box = FancyBboxPatch((0.05, 0.75), 0.2, 0.08, boxstyle="round,pad=0.02", 
                                 facecolor=colors['boundary'], alpha=0.2, 
                                 edgecolor=colors['boundary'], linewidth=2)
    ax.add_patch(boundary_box)
    
    ax.text(0.15, 0.79, 'Each prime p defines', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    ax.text(0.15, 0.75, 'δ boundary at log p', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    
    # Visual representation
    primes = [2, 3, 5, 7, 11]
    for i, p in enumerate(primes):
        log_p = np.log(p)
        x_pos = 0.05 + (log_p / 3.0) * 0.2  # Scale to fit
        y_pos = 0.70 - i * 0.02
        
        # Boundary line
        ax.plot([x_pos, x_pos], [y_pos, y_pos + 0.015], 
               color=colors['boundary'], linewidth=3)
        ax.text(x_pos, y_pos - 0.01, f'p={p}', ha='center', va='top', 
               fontsize=8, color=colors['boundary'], fontweight='bold')
    
    # 2. Multiplicative Laplace Equation
    ax.text(0.5, 0.85, 'Multiplicative Laplace Equation', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['primary'], transform=ax.transAxes)
    
    # Equation box
    laplace_box = FancyBboxPatch((0.35, 0.70), 0.3, 0.12, boxstyle="round,pad=0.02", 
                                facecolor=colors['primary'], alpha=0.2, 
                                edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(laplace_box)
    
    ax.text(0.5, 0.78, 'ζ(s) = ∏(1-p^{-s})^{-1}', 
           ha='center', va='center', fontsize=14, fontweight='bold', 
           color=colors['primary'], transform=ax.transAxes)
    ax.text(0.5, 0.74, 'solves multiplicative Laplace eq', 
           ha='center', va='center', fontsize=10, 
           color=colors['primary'], transform=ax.transAxes)
    
    # Visual representation - complex plane
    sigma = np.linspace(0, 1, 20)
    t = np.linspace(0, 20, 20)
    S, T = np.meshgrid(sigma, t)
    
    # Create zeta magnitude visualization
    zeta_mag = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s = S[i, j] + 1j * T[i, j]
            # Simplified zeta magnitude
            zeta_mag[i, j] = 1.0 / (abs(s - 0.5) + 0.1)
    
    # Plot in transformed coordinates
    x_offset = 0.35
    y_offset = 0.55
    scale = 0.3
    
    im = ax.imshow(zeta_mag, extent=[x_offset, x_offset + scale, y_offset, y_offset + scale], 
                  origin='lower', cmap='viridis', alpha=0.7)
    
    # Critical line
    ax.plot([x_offset, x_offset + scale], [y_offset + scale/2, y_offset + scale/2], 
           color=colors['axis'], linewidth=3)
    
    # Zeta zero
    ax.plot(x_offset + scale/2, y_offset + scale/2, 'o', color=colors['accent'], 
           markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # 3. Mirror Symmetry
    ax.text(0.85, 0.85, 'Mirror Symmetry', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Equation box
    mirror_box = FancyBboxPatch((0.75, 0.70), 0.2, 0.12, boxstyle="round,pad=0.02", 
                               facecolor=colors['secondary'], alpha=0.2, 
                               edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(mirror_box)
    
    ax.text(0.85, 0.78, 's ↔ 1-s', 
           ha='center', va='center', fontsize=14, fontweight='bold', 
           color=colors['secondary'], transform=ax.transAxes)
    ax.text(0.85, 0.74, 'axis = critical line', 
           ha='center', va='center', fontsize=10, 
           color=colors['secondary'], transform=ax.transAxes)
    
    # Visual representation - reflection
    x_offset = 0.75
    y_offset = 0.55
    scale = 0.2
    
    s_values = np.linspace(0, 1, 50)
    ax.plot(x_offset + scale * s_values, y_offset + scale * s_values, 
           '-', color=colors['axis'], linewidth=3, alpha=0.8)
    ax.plot(x_offset + scale * s_values, y_offset + scale * (1 - s_values), 
           '--', color=colors['secondary'], linewidth=3, alpha=0.8)
    
    # Fixed point
    ax.plot(x_offset + scale * 0.5, y_offset + scale * 0.5, 'o', color=colors['axis'], 
           markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # 4. Attractor Eigenmodes
    ax.text(0.15, 0.45, 'Attractor Eigenmodes', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['accent'], transform=ax.transAxes)
    
    # Equation box
    eigenmodes_box = FancyBboxPatch((0.05, 0.30), 0.2, 0.12, boxstyle="round,pad=0.02", 
                                   facecolor=colors['accent'], alpha=0.2, 
                                   edgecolor=colors['accent'], linewidth=2)
    ax.add_patch(eigenmodes_box)
    
    ax.text(0.15, 0.38, 'Zeros as eigenmodes', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['accent'], transform=ax.transAxes)
    ax.text(0.15, 0.34, 'under constraints', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['accent'], transform=ax.transAxes)
    
    # Visual representation - eigenmodes
    x_offset = 0.05
    y_offset = 0.15
    scale = 0.2
    
    # Create eigenmode visualization
    modes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    for i, mode in enumerate(modes):
        ax.bar(x_offset + i * 0.04, y_offset + scale * mode, 0.03, 
              color=colors['accent'], alpha=0.7)
    
    # Highlight dominant mode
    ax.bar(x_offset, y_offset + scale * modes[0], 0.03, 
          color=colors['primary'], alpha=0.8)
    
    # 5. Integer Flow
    ax.text(0.5, 0.45, 'Integer Flow', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    
    # Equation box
    flow_box = FancyBboxPatch((0.35, 0.30), 0.3, 0.12, boxstyle="round,pad=0.02", 
                             facecolor=colors['boundary'], alpha=0.2, 
                             edgecolor=colors['boundary'], linewidth=2)
    ax.add_patch(flow_box)
    
    ax.text(0.5, 0.38, 'Flow determined by', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    ax.text(0.5, 0.34, 'prime placement', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['boundary'], transform=ax.transAxes)
    
    # Visual representation - flow
    x_offset = 0.35
    y_offset = 0.15
    scale = 0.3
    
    # Show flow for integer 12 = 2² × 3
    primes_flow = [2, 3]
    colors_flow = [colors['primary'], colors['secondary']]
    
    for i, p in enumerate(primes_flow):
        log_p = np.log(p)
        x_pos = x_offset + (log_p / 3.0) * scale
        y_pos = y_offset + scale * 0.5
        
        # Flow arrow
        ax.annotate('', xy=(x_pos + 0.05, y_pos), xytext=(x_pos - 0.05, y_pos),
                   arrowprops=dict(arrowstyle='->', color=colors_flow[i], lw=3))
        ax.text(x_pos, y_pos + 0.02, f'p={p}', ha='center', va='bottom', 
               fontsize=10, color=colors_flow[i], fontweight='bold')
    
    # 6. CE1 Integration
    ax.text(0.85, 0.45, 'CE1 Integration', 
           ha='center', va='center', fontsize=16, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    
    # Equation box
    ce1_box = FancyBboxPatch((0.75, 0.30), 0.2, 0.12, boxstyle="round,pad=0.02", 
                            facecolor=colors['text'], alpha=0.2, 
                            edgecolor=colors['text'], linewidth=2)
    ax.add_patch(ce1_box)
    
    ax.text(0.85, 0.38, 'K(x,y) = δ(y - I·x)', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color=colors['text'], transform=ax.transAxes)
    ax.text(0.85, 0.34, 'Involution Kernel', 
           ha='center', va='center', fontsize=10, 
           color=colors['text'], transform=ax.transAxes)
    
    # Visual representation - CE1 kernel
    x_offset = 0.75
    y_offset = 0.15
    scale = 0.2
    
    # Central involution
    ax.plot(x_offset + scale/2, y_offset + scale/2, 'o', color=colors['axis'], 
           markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.text(x_offset + scale/2, y_offset + scale/2, 'I', ha='center', va='center', 
           fontsize=10, color='white', fontweight='bold')
    
    # Symmetric structures
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for i, angle in enumerate(angles):
        x = x_offset + scale/2 + 0.08 * np.cos(angle)
        y = y_offset + scale/2 + 0.08 * np.sin(angle)
        
        ax.plot(x, y, 's', color=colors['boundary'], markersize=6, alpha=0.8)
        ax.plot([x_offset + scale/2, x], [y_offset + scale/2, y], 
               '--', color=colors['boundary'], alpha=0.5)
    
    # Add connecting arrows
    # From boundaries to Laplace
    ax.annotate('', xy=(0.35, 0.76), xytext=(0.25, 0.76),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From Laplace to symmetry
    ax.annotate('', xy=(0.75, 0.76), xytext=(0.65, 0.76),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From eigenmodes to flow
    ax.annotate('', xy=(0.35, 0.36), xytext=(0.25, 0.36),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # From flow to CE1
    ax.annotate('', xy=(0.75, 0.36), xytext=(0.65, 0.36),
               arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add footer
    ax.text(0.5, 0.02, 'BoundaryPrimes: Prime boundaries create involution structure underlying Riemann zeta function', 
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
    
    parser = argparse.ArgumentParser(description="CE1 BoundaryPrimes Equations")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create equations diagram
    output_file = create_boundary_primes_equations_diagram(args.output)
    print(f"Generated BoundaryPrimes equations diagram: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
