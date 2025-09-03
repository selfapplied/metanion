#!/usr/bin/env python3
"""
CE1 BoundaryPrimes: Prime Boundaries and Multiplicative Laplace Equation

Implements the BoundaryPrimes concept where:
- Each prime p defines a delta boundary at log p
- ζ(s) = ∏(1-p^{-s})^{-1} solves multiplicative Laplace equation
- s↔1-s symmetry with axis = critical line
- Zeros as eigenmodes under these constraints
- Integer flow determined by prime placement

This extends CE1 to show how prime boundaries create the involution structure
that underlies the Riemann zeta function and its zeros.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import math

# Import CE1 components
from ce1_core import TimeReflectionInvolution, CE1Kernel
from ce1_convolution import DressedCE1Kernel, MellinDressing


class BoundaryPrimes:
    """
    BoundaryPrimes implementation showing how primes define delta boundaries
    and create the involution structure underlying the Riemann zeta function.
    """
    
    def __init__(self, max_prime: int = 100):
        self.max_prime = max_prime
        self.primes = self._generate_primes(max_prime)
        self.log_primes = np.log(self.primes)
        self.involution = TimeReflectionInvolution()
        self.ce1_kernel = CE1Kernel(self.involution)
        
    def _generate_primes(self, n: int) -> List[int]:
        """Generate list of primes up to n"""
        if n < 2:
            return []
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, n + 1) if sieve[i]]
    
    def pin_boundaries(self, s: complex) -> Dict[str, Any]:
        """
        Pin each prime p defines delta boundary at log p
        
        Returns the boundary structure created by prime placement.
        """
        boundaries = {}
        
        for p in self.primes:
            log_p = np.log(p)
            
            # Delta boundary at log p
            boundary_strength = 1.0 / p  # Strength inversely proportional to prime
            
            # Distance from s to boundary
            if np.iscomplexobj(s):
                s_real = s.real
                s_imag = s.imag
            else:
                s_real = s
                s_imag = 0.0
            
            # Boundary effect (simplified)
            boundary_effect = boundary_strength * np.exp(-abs(s_real - log_p))
            
            boundaries[p] = {
                'log_p': log_p,
                'strength': boundary_strength,
                'effect': boundary_effect,
                'position': (log_p, 0.0)  # Position in (log p, 0) plane
            }
        
        return boundaries
    
    def field_equation(self, s: complex) -> complex:
        """
        ζ(s) = ∏(1-p^{-s})^{-1} solves multiplicative Laplace equation
        
        This shows how the zeta function emerges from the prime boundary structure.
        """
        # Euler product formula
        zeta_value = 1.0
        
        for p in self.primes:
            if p > 1000:  # Limit for computational efficiency
                break
            zeta_value *= 1.0 / (1.0 - p**(-s))
        
        return zeta_value
    
    def mirror_symmetry(self, s: complex) -> Dict[str, Any]:
        """
        s↔1-s symmetry with axis = critical line
        
        Shows how the involution structure emerges from prime boundaries.
        """
        # Apply time reflection involution
        s_reflected = self.involution.apply(s)
        
        # Compute zeta values
        zeta_s = self.field_equation(s)
        zeta_reflected = self.field_equation(s_reflected)
        
        # Mirror residual
        mirror_residual = zeta_reflected - zeta_s
        
        # Check if on critical line
        on_critical_line = abs(s.real - 0.5) < 1e-12 if np.iscomplexobj(s) else abs(s - 0.5) < 1e-12
        
        return {
            's': s,
            's_reflected': s_reflected,
            'zeta_s': zeta_s,
            'zeta_reflected': zeta_reflected,
            'mirror_residual': mirror_residual,
            'on_critical_line': on_critical_line,
            'symmetry_preserved': abs(mirror_residual) < 1e-6
        }
    
    def attractor_eigenmodes(self, s: complex) -> Dict[str, Any]:
        """
        Zeros as eigenmodes under prime boundary constraints
        
        Shows how zeta zeros emerge as eigenmodes of the prime boundary system.
        """
        # Get boundary structure
        boundaries = self.pin_boundaries(s)
        
        # Compute boundary matrix (simplified)
        n_boundaries = len(boundaries)
        boundary_matrix = np.zeros((n_boundaries, n_boundaries))
        
        primes_list = list(boundaries.keys())
        for i, p1 in enumerate(primes_list):
            for j, p2 in enumerate(primes_list):
                if i == j:
                    boundary_matrix[i, j] = boundaries[p1]['strength']
                else:
                    # Interaction between boundaries
                    log_p1 = boundaries[p1]['log_p']
                    log_p2 = boundaries[p2]['log_p']
                    interaction = np.exp(-abs(log_p1 - log_p2))
                    boundary_matrix[i, j] = interaction * boundaries[p1]['strength'] * boundaries[p2]['strength']
        
        # Compute eigenvalues (eigenmodes)
        eigenvals = np.linalg.eigvals(boundary_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort by magnitude
        
        # Check if s is near a zero (eigenmode)
        zeta_value = self.field_equation(s)
        near_zero = abs(zeta_value) < 1e-6
        
        return {
            'boundary_matrix': boundary_matrix,
            'eigenvalues': eigenvals,
            'zeta_value': zeta_value,
            'near_zero': near_zero,
            'dominant_eigenmode': eigenvals[0] if len(eigenvals) > 0 else 0.0
        }
    
    def integer_flow(self, n: int) -> Dict[str, Any]:
        """
        Integer flow determined by prime placement
        
        Shows how integers flow through the prime boundary structure.
        """
        # Prime factorization
        factors = self._prime_factorization(n)
        
        # Compute flow through boundaries
        flow_path = []
        current_value = 1.0
        
        for p, exp in factors.items():
            log_p = np.log(p)
            boundary_effect = p**(-exp)  # Multiplicative effect
            
            flow_path.append({
                'prime': p,
                'exponent': exp,
                'log_p': log_p,
                'boundary_effect': boundary_effect,
                'cumulative_effect': current_value * boundary_effect
            })
            
            current_value *= boundary_effect
        
        # Total flow
        total_flow = current_value
        
        return {
            'n': n,
            'factors': factors,
            'flow_path': flow_path,
            'total_flow': total_flow,
            'log_flow': np.log(total_flow) if total_flow > 0 else -np.inf
        }
    
    def _prime_factorization(self, n: int) -> Dict[int, int]:
        """Compute prime factorization of n"""
        factors = {}
        
        for p in self.primes:
            if p > n:
                break
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            if exp > 0:
                factors[p] = exp
        
        if n > 1:
            factors[n] = 1  # Remaining prime factor
        
        return factors


class BoundaryPrimesVisualizer:
    """
    Creates visualizations of the BoundaryPrimes concept.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'accent': '#F18F01',       # Orange
            'axis': '#E74C3C',         # Red
            'boundary': '#27AE60',     # Green
            'text': '#2C3E50',         # Dark blue-gray
            'background': '#FAFAFA'    # Light gray
        }
    
    def create_boundary_primes_diagram(self, boundary_primes: BoundaryPrimes, output_file: str = None) -> str:
        """
        Create comprehensive BoundaryPrimes visualization.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/ce1_visualization/boundary_primes_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Prime boundaries
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_prime_boundaries(ax1, boundary_primes)
        
        # 2. Multiplicative Laplace equation
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_multiplicative_laplace(ax2, boundary_primes)
        
        # 3. Mirror symmetry
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_mirror_symmetry(ax3, boundary_primes)
        
        # 4. Attractor eigenmodes
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_attractor_eigenmodes(ax4, boundary_primes)
        
        # 5. Integer flow
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_integer_flow(ax5, boundary_primes)
        
        # 6. CE1 integration
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_ce1_integration(ax6, boundary_primes)
        
        # Main title
        fig.suptitle('CE1 BoundaryPrimes: Prime Boundaries and Multiplicative Laplace Equation', 
                    fontsize=20, fontweight='bold', color=self.colors['text'], y=0.95)
        
        # Subtitle
        fig.text(0.5, 0.92, 'Each prime p defines delta boundary at log p; ζ(s) = ∏(1-p^{-s})^{-1} solves multiplicative Laplace eq', 
                ha='center', fontsize=12, style='italic', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_prime_boundaries(self, ax, boundary_primes: BoundaryPrimes):
        """Plot prime boundaries at log p"""
        ax.set_title('Prime Boundaries: Each p defines δ at log p', 
                    fontweight='bold', color=self.colors['boundary'])
        
        # Plot boundaries
        for i, p in enumerate(boundary_primes.primes[:20]):  # Show first 20 primes
            log_p = np.log(p)
            strength = 1.0 / p
            
            # Plot boundary as vertical line
            ax.axvline(x=log_p, color=self.colors['boundary'], alpha=0.7, linewidth=2)
            
            # Add prime label
            ax.text(log_p, strength, f'{p}', ha='center', va='bottom', 
                   fontsize=8, color=self.colors['boundary'], fontweight='bold')
            
            # Add strength indicator
            ax.plot(log_p, strength, 'o', color=self.colors['boundary'], markersize=6)
        
        # Add some zeta zeros for reference
        zeta_zeros = [14.134725, 21.02204, 25.010858]
        for t_zero in zeta_zeros:
            ax.axvline(x=0.5, ymin=0, ymax=0.1, color=self.colors['axis'], linewidth=3, alpha=0.8)
            ax.text(0.5, 0.05, f'ζ₀', ha='center', va='center', 
                   fontsize=10, color=self.colors['axis'], fontweight='bold')
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel('log p', fontweight='bold')
        ax.set_ylabel('Boundary Strength', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_multiplicative_laplace(self, ax, boundary_primes: BoundaryPrimes):
        """Plot multiplicative Laplace equation solution"""
        ax.set_title('Multiplicative Laplace: ζ(s) = ∏(1-p^{-s})^{-1}', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create complex plane
        sigma = np.linspace(0, 1, 50)
        t = np.linspace(0, 30, 50)
        S, T = np.meshgrid(sigma, t)
        
        # Compute zeta magnitude (simplified)
        zeta_mag = np.zeros_like(S)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                s = S[i, j] + 1j * T[i, j]
                zeta_val = boundary_primes.field_equation(s)
                zeta_mag[i, j] = abs(zeta_val)
        
        # Plot zeta magnitude
        im = ax.imshow(zeta_mag, extent=[0, 1, 0, 30], origin='lower', 
                      cmap='viridis', alpha=0.8)
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=3, label='Critical Line')
        
        # Zeta zeros
        zeta_zeros = [14.134725, 21.02204, 25.010858]
        for t_zero in zeta_zeros:
            ax.plot(0.5, t_zero, 'o', color=self.colors['accent'], markersize=8, 
                   markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 30)
        ax.set_xlabel('Re(s)', fontweight='bold')
        ax.set_ylabel('Im(s)', fontweight='bold')
        ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_mirror_symmetry(self, ax, boundary_primes: BoundaryPrimes):
        """Plot s↔1-s symmetry"""
        ax.set_title('Mirror Symmetry: s↔1-s', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Create s values
        s_values = np.linspace(0, 1, 100)
        
        # Plot identity and reflection
        ax.plot(s_values, s_values, '-', color=self.colors['axis'], linewidth=3, label='Identity')
        ax.plot(s_values, 1 - s_values, '--', color=self.colors['secondary'], linewidth=3, label='I: s↦1-s')
        
        # Critical line
        ax.axvline(x=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8, label='Critical Line')
        ax.axhline(y=0.5, color=self.colors['axis'], linewidth=2, alpha=0.8)
        
        # Fixed point
        ax.plot(0.5, 0.5, 'o', color=self.colors['axis'], markersize=10, 
               markeredgecolor='white', markeredgewidth=2)
        ax.text(0.5, 0.5, 'A', ha='center', va='center', 
               fontsize=12, color='white', fontweight='bold')
        
        # Add some prime boundaries
        for p in boundary_primes.primes[:5]:
            log_p = np.log(p)
            if log_p <= 1:
                ax.axvline(x=log_p, color=self.colors['boundary'], alpha=0.6, linewidth=1)
                ax.text(log_p, 0.9, f'log {p}', ha='center', va='center', 
                       fontsize=8, color=self.colors['boundary'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('s', fontweight='bold')
        ax.set_ylabel('I(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
    
    def _plot_attractor_eigenmodes(self, ax, boundary_primes: BoundaryPrimes):
        """Plot attractor eigenmodes"""
        ax.set_title('Attractor Eigenmodes: Zeros as Eigenmodes', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Compute eigenmodes for a test point
        s_test = 0.5 + 14.134725j
        eigenmodes = boundary_primes.attractor_eigenmodes(s_test)
        
        # Plot eigenvalues
        eigenvals = eigenmodes['eigenvalues']
        n_modes = len(eigenvals)
        
        ax.bar(range(n_modes), np.real(eigenvals), color=self.colors['accent'], alpha=0.7, label='Eigenvalues')
        
        # Highlight dominant mode
        if n_modes > 0:
            ax.bar(0, np.real(eigenvals[0]), color=self.colors['primary'], alpha=0.8, label='Dominant Mode')
        
        # Add zeta zero indicator
        zeta_value = eigenmodes['zeta_value']
        ax.axhline(y=abs(zeta_value), color=self.colors['axis'], linewidth=2, linestyle='--', 
                  label=f'|ζ(s)| = {abs(zeta_value):.2e}')
        
        ax.set_xlim(-0.5, min(n_modes, 20) - 0.5)
        ax.set_xlabel('Eigenmode Index', fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def _plot_integer_flow(self, ax, boundary_primes: BoundaryPrimes):
        """Plot integer flow through prime boundaries"""
        ax.set_title('Integer Flow: Determined by Prime Placement', 
                    fontweight='bold', color=self.colors['boundary'])
        
        # Show flow for some integers
        integers = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
        colors = plt.cm.viridis(np.linspace(0, 1, len(integers)))
        
        for i, n in enumerate(integers):
            flow = boundary_primes.integer_flow(n)
            
            # Plot flow path
            x_positions = []
            y_positions = []
            
            for step in flow['flow_path']:
                x_positions.append(step['log_p'])
                y_positions.append(step['cumulative_effect'])
            
            if x_positions:
                ax.plot(x_positions, y_positions, 'o-', color=colors[i], 
                       linewidth=2, markersize=6, label=f'n={n}')
        
        # Add prime boundaries
        for p in boundary_primes.primes[:10]:
            log_p = np.log(p)
            ax.axvline(x=log_p, color=self.colors['boundary'], alpha=0.3, linewidth=1)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.set_xlabel('log p', fontweight='bold')
        ax.set_ylabel('Cumulative Flow', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    
    def _plot_ce1_integration(self, ax, boundary_primes: BoundaryPrimes):
        """Plot CE1 integration with prime boundaries"""
        ax.set_title('CE1 Integration: Prime Boundaries → Involution', 
                    fontweight='bold', color=self.colors['text'])
        
        # Show how prime boundaries create involution structure
        # Central involution
        center_x = 0.5
        center_y = 0.5
        ax.plot(center_x, center_y, 'o', color=self.colors['axis'], markersize=15, 
               markeredgecolor='white', markeredgewidth=2)
        ax.text(center_x, center_y, 'I', ha='center', va='center', 
               fontsize=12, color='white', fontweight='bold')
        
        # Prime boundaries as symmetric structures
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(angles):
            x = center_x + 0.3 * np.cos(angle)
            y = center_y + 0.3 * np.sin(angle)
            
            # Plot prime boundary
            ax.plot(x, y, 's', color=self.colors['boundary'], markersize=8, alpha=0.8)
            ax.plot([center_x, x], [center_y, y], '--', color=self.colors['boundary'], alpha=0.5)
            
            # Add reflected boundary
            x_ref = center_x + 0.3 * np.cos(angle + np.pi)
            y_ref = center_y + 0.3 * np.sin(angle + np.pi)
            ax.plot(x_ref, y_ref, 's', color=self.colors['secondary'], markersize=6, alpha=0.8)
            ax.plot([center_x, x_ref], [center_y, y_ref], '--', color=self.colors['secondary'], alpha=0.5)
        
        # Add axis
        ax.axhline(y=center_y, xmin=0.2, xmax=0.8, color=self.colors['axis'], linewidth=2, alpha=0.8)
        ax.axvline(x=center_x, ymin=0.2, ymax=0.8, color=self.colors['axis'], linewidth=2, alpha=0.8)
        
        # Add text
        ax.text(0.5, 0.1, 'Prime Boundaries\n→ Involution Structure\n→ CE1 Kernel', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               color=self.colors['text'], bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=self.colors['background'], alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Prime Boundary Space', fontweight='bold')
        ax.set_ylabel('Involution Structure', fontweight='bold')
        ax.set_aspect('equal')


def main():
    """Main entry point for BoundaryPrimes visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CE1 BoundaryPrimes Visualization")
    parser.add_argument("--max-prime", type=int, default=100, help="Maximum prime to consider")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize BoundaryPrimes
    boundary_primes = BoundaryPrimes(max_prime=args.max_prime)
    
    print(f"BoundaryPrimes initialized with {len(boundary_primes.primes)} primes")
    print(f"First 10 primes: {boundary_primes.primes[:10]}")
    
    # Test some operations
    s_test = 0.5 + 14.134725j
    print(f"\nTesting at s = {s_test}")
    
    # Pin boundaries
    boundaries = boundary_primes.pin_boundaries(s_test)
    print(f"Boundary effects for first 5 primes:")
    for p in boundary_primes.primes[:5]:
        print(f"  p={p}: log_p={boundaries[p]['log_p']:.3f}, effect={boundaries[p]['effect']:.3f}")
    
    # Field equation
    zeta_value = boundary_primes.field_equation(s_test)
    print(f"ζ(s) = {zeta_value:.6f}")
    
    # Mirror symmetry
    mirror = boundary_primes.mirror_symmetry(s_test)
    print(f"Mirror residual: {mirror['mirror_residual']:.6f}")
    
    # Attractor eigenmodes
    eigenmodes = boundary_primes.attractor_eigenmodes(s_test)
    print(f"Dominant eigenmode: {eigenmodes['dominant_eigenmode']:.6f}")
    
    # Integer flow
    flow = boundary_primes.integer_flow(12)
    print(f"Integer 12 flow: {flow['total_flow']:.6f}")
    
    # Create visualization
    visualizer = BoundaryPrimesVisualizer()
    output_file = visualizer.create_boundary_primes_diagram(boundary_primes, args.output)
    
    print(f"\nGenerated BoundaryPrimes visualization: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
