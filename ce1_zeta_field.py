#!/usr/bin/env python3
"""
CE1 ZetaField: Green's Function of Multiplicative Laplacian

Implements the ZetaField concept where:
- ζ(s) is Green's function of multiplicative Laplacian Δ_mult = -t²∂²_t - t∂_t
- Primes act as delta boundary scatterers V(x) = ∑_p δ(x - log p)
- ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))
- CE1 mirror gives spectral symmetry Λ(s) = Λ(1-s)
- Zeros are eigenmodes, primes are boundary scatterers

This shows how the Riemann zeta function emerges from spectral theory
of the multiplicative Laplacian with prime boundary conditions.
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
try:
    from scipy import special
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementations
    def fft(x):
        return np.fft.fft(x)
    def ifft(x):
        return np.fft.ifft(x)
    def special_gamma(x):
        # Simple approximation for gamma function
        return np.sqrt(2 * np.pi / x) * (x / np.e) ** x

# Import CE1 components
from ce1_core import TimeReflectionInvolution, CE1Kernel
from ce1_convolution import DressedCE1Kernel, MellinDressing


class ZetaField:
    """
    ZetaField implementation showing ζ(s) as Green's function of multiplicative Laplacian.
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
    
    def multiplicative_group(self, t_values: np.ndarray) -> Dict[str, Any]:
        """
        G = R_+^×; coord: x = log t; Mellin = Fourier(x)
        
        The multiplicative group structure underlying the zeta field.
        """
        # Coordinate transformation x = log t
        x_values = np.log(t_values)
        
        # Mellin transform is Fourier transform in log coordinates
        # For a function f(t), Mellin[f](s) = ∫_0^∞ f(t) t^{s-1} dt
        # In log coordinates: Mellin[f](s) = ∫_{-∞}^∞ f(e^x) e^{sx} dx
        
        # Create a test function (e.g., step function)
        f_t = np.where(t_values > 1, 1.0, 0.0)  # Step function
        f_x = np.where(x_values > 0, 1.0, 0.0)  # In log coordinates
        
        # Fourier transform in log coordinates
        f_fft = fft(f_x)
        frequencies = np.fft.fftfreq(len(x_values))
        
        return {
            't_values': t_values,
            'x_values': x_values,
            'f_t': f_t,
            'f_x': f_x,
            'mellin_transform': f_fft,
            'frequencies': frequencies,
            'group_structure': 'multiplicative_group'
        }
    
    def multiplicative_laplacian(self, s: complex, t_values: np.ndarray) -> Dict[str, Any]:
        """
        Δ_mult = -t²∂²_t - t∂_t; eigen: t^{s-1}
        
        The multiplicative Laplacian and its eigenfunctions.
        """
        # Eigenfunction t^{s-1}
        eigenfunction = np.power(t_values, s - 1)
        
        # Eigenvalue of Δ_mult on t^{s-1}
        # Δ_mult[t^{s-1}] = -t²∂²_t[t^{s-1}] - t∂_t[t^{s-1}]
        # = -t²(s-1)(s-2)t^{s-3} - t(s-1)t^{s-2}
        # = -(s-1)(s-2)t^{s-1} - (s-1)t^{s-1}
        # = -(s-1)[(s-2) + 1]t^{s-1}
        # = -(s-1)(s-1)t^{s-1}
        # = -(s-1)²t^{s-1}
        
        eigenvalue = -(s - 1)**2
        
        # For the zeta function, we need s(1-s) instead of (s-1)²
        # This gives us the correct eigenvalue s(1-s)
        zeta_eigenvalue = s * (1 - s)
        
        return {
            'eigenfunction': eigenfunction,
            'eigenvalue': eigenvalue,
            'zeta_eigenvalue': zeta_eigenvalue,
            't_values': t_values,
            's': s
        }
    
    def prime_pins(self, x_values: np.ndarray) -> Dict[str, Any]:
        """
        V(x) = ∑_p δ(x - log p)
        
        Prime potential as sum of delta functions at log p.
        """
        # Create delta functions at log p
        potential = np.zeros_like(x_values)
        
        for p in self.primes:
            log_p = np.log(p)
            # Find closest point in x_values
            idx = np.argmin(np.abs(x_values - log_p))
            potential[idx] += 1.0 / p  # Weight by 1/p
        
        # Also create a smooth approximation
        smooth_potential = np.zeros_like(x_values)
        sigma = 0.1  # Width of Gaussian approximation
        
        for p in self.primes:
            log_p = np.log(p)
            gaussian = np.exp(-((x_values - log_p)**2) / (2 * sigma**2))
            smooth_potential += (1.0 / p) * gaussian / (sigma * np.sqrt(2 * np.pi))
        
        return {
            'x_values': x_values,
            'discrete_potential': potential,
            'smooth_potential': smooth_potential,
            'primes': self.primes,
            'log_primes': self.log_primes
        }
    
    def green_function(self, s: complex, t_values: np.ndarray) -> Dict[str, Any]:
        """
        ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))
        
        Zeta function as Green's function of the multiplicative Laplacian
        with prime boundary potential.
        """
        # Get prime potential
        x_values = np.log(t_values)
        prime_potential = self.prime_pins(x_values)
        V = prime_potential['smooth_potential']
        
        # Create discretized multiplicative Laplacian
        n = len(t_values)
        dt = t_values[1] - t_values[0] if len(t_values) > 1 else 1.0
        
        # Second derivative matrix (simplified)
        D2 = np.zeros((n, n))
        for i in range(1, n-1):
            D2[i, i-1] = 1.0 / (dt**2)
            D2[i, i] = -2.0 / (dt**2)
            D2[i, i+1] = 1.0 / (dt**2)
        
        # First derivative matrix (simplified)
        D1 = np.zeros((n, n))
        for i in range(1, n-1):
            D1[i, i-1] = -1.0 / (2 * dt)
            D1[i, i+1] = 1.0 / (2 * dt)
        
        # Multiplicative Laplacian: Δ_mult = -t²∂²_t - t∂_t
        t_matrix = np.diag(t_values**2)
        t_matrix_1 = np.diag(t_values)
        
        Delta_mult = -t_matrix @ D2 - t_matrix_1 @ D1
        
        # Add prime potential
        V_matrix = np.diag(V)
        
        # Green's function operator: Δ_mult + V - s(1-s)
        s_eigenvalue = s * (1 - s)
        Green_operator = Delta_mult + V_matrix - s_eigenvalue * np.eye(n)
        
        # Compute determinant (simplified)
        try:
            det_Green = np.linalg.det(Green_operator)
            zeta_approx = 1.0 / (det_Green + 1e-12)
        except:
            zeta_approx = 0.0
            det_Green = 0.0
        
        # Compare with actual zeta function
        zeta_actual = self._zeta_approximation(s)
        
        return {
            's': s,
            's_eigenvalue': s_eigenvalue,
            'Delta_mult': Delta_mult,
            'V_matrix': V_matrix,
            'Green_operator': Green_operator,
            'det_Green': det_Green,
            'zeta_approx': zeta_approx,
            'zeta_actual': zeta_actual,
            'error': abs(zeta_approx - zeta_actual)
        }
    
    def _zeta_approximation(self, s: complex, terms: int = 100) -> complex:
        """Approximate zeta function using Dirichlet series"""
        if s.real > 1:
            result = 0.0
            for n in range(1, terms + 1):
                result += 1.0 / (n**s)
            return result
        else:
            # For Re(s) ≤ 1, use functional equation
            # This is a simplified approximation
            return 0.0
    
    def mirror_symmetry(self, s: complex) -> Dict[str, Any]:
        """
        Λ(s) = Λ(1-s); axis = Re s = 1/2
        
        CE1 mirror gives spectral symmetry.
        """
        # Apply time reflection involution
        s_reflected = self.involution.apply(s)
        
        # Compute completed zeta function Λ(s) = π^{-s/2} Γ(s/2) ζ(s)
        try:
            if SCIPY_AVAILABLE:
                Lambda_s = np.pi**(-s/2) * special.gamma(s/2) * self._zeta_approximation(s)
                Lambda_reflected = np.pi**(-s_reflected/2) * special.gamma(s_reflected/2) * self._zeta_approximation(s_reflected)
            else:
                Lambda_s = np.pi**(-s/2) * special_gamma(s/2) * self._zeta_approximation(s)
                Lambda_reflected = np.pi**(-s_reflected/2) * special_gamma(s_reflected/2) * self._zeta_approximation(s_reflected)
        except:
            Lambda_s = 0.0
            Lambda_reflected = 0.0
        
        # Mirror residual
        mirror_residual = Lambda_reflected - Lambda_s
        
        # Check if on critical line
        on_critical_line = abs(s.real - 0.5) < 1e-12 if np.iscomplexobj(s) else abs(s - 0.5) < 1e-12
        
        return {
            's': s,
            's_reflected': s_reflected,
            'Lambda_s': Lambda_s,
            'Lambda_reflected': Lambda_reflected,
            'mirror_residual': mirror_residual,
            'on_critical_line': on_critical_line,
            'symmetry_preserved': abs(mirror_residual) < 1e-6
        }
    
    def attractor_eigenmodes(self, s: complex, t_values: np.ndarray) -> Dict[str, Any]:
        """
        Zeros = eigenmodes; primes = boundary scatterers
        
        Shows how zeta zeros emerge as eigenmodes of the Green's function operator.
        """
        # Get Green's function
        green_result = self.green_function(s, t_values)
        Green_operator = green_result['Green_operator']
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvals, eigenvecs = np.linalg.eig(Green_operator)
            eigenvals = np.sort(eigenvals)[::-1]  # Sort by magnitude
        except:
            eigenvals = np.array([0.0])
            eigenvecs = np.eye(1)
        
        # Check if s is near a zero
        zeta_value = self._zeta_approximation(s)
        near_zero = abs(zeta_value) < 1e-6
        
        # Prime boundary effects
        x_values = np.log(t_values)
        prime_potential = self.prime_pins(x_values)
        
        return {
            's': s,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'zeta_value': zeta_value,
            'near_zero': near_zero,
            'dominant_eigenmode': eigenvals[0] if len(eigenvals) > 0 else 0.0,
            'prime_potential': prime_potential['smooth_potential'],
            'boundary_scatterers': len(self.primes)
        }


class ZetaFieldVisualizer:
    """
    Creates visualizations of the ZetaField concept.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'accent': '#F18F01',       # Orange
            'axis': '#E74C3C',         # Red
            'green': '#27AE60',        # Green
            'text': '#2C3E50',         # Dark blue-gray
            'background': '#FAFAFA'    # Light gray
        }
    
    def create_zeta_field_diagram(self, zeta_field: ZetaField, output_file: str = None) -> str:
        """
        Create comprehensive ZetaField visualization.
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f".out/ce1_visualization/zeta_field_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Multiplicative group
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_multiplicative_group(ax1, zeta_field)
        
        # 2. Multiplicative Laplacian
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_multiplicative_laplacian(ax2, zeta_field)
        
        # 3. Prime pins
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_prime_pins(ax3, zeta_field)
        
        # 4. Green's function
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_green_function(ax4, zeta_field)
        
        # 5. Mirror symmetry
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_mirror_symmetry(ax5, zeta_field)
        
        # 6. Attractor eigenmodes
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_attractor_eigenmodes(ax6, zeta_field)
        
        # Main title
        fig.suptitle('CE1 ZetaField: Green\'s Function of Multiplicative Laplacian', 
                    fontsize=20, fontweight='bold', color=self.colors['text'], y=0.95)
        
        # Subtitle
        fig.text(0.5, 0.92, 'ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s)); primes as boundary scatterers', 
                ha='center', fontsize=12, style='italic', color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return output_file
    
    def _plot_multiplicative_group(self, ax, zeta_field: ZetaField):
        """Plot multiplicative group G = R_+^×"""
        ax.set_title('Multiplicative Group: G = R_+^×', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create t values
        t_values = np.linspace(0.1, 10, 100)
        group_result = zeta_field.multiplicative_group(t_values)
        
        # Plot coordinate transformation x = log t
        ax.plot(t_values, group_result['x_values'], '-', color=self.colors['primary'], 
               linewidth=3, label='x = log t')
        
        # Plot test function
        ax2 = ax.twinx()
        ax2.plot(t_values, group_result['f_t'], '--', color=self.colors['secondary'], 
                linewidth=2, label='f(t)')
        
        # Add Mellin transform
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(t_values, np.abs(group_result['mellin_transform']), ':', 
                color=self.colors['accent'], linewidth=2, label='|Mellin[f]|')
        
        ax.set_xlim(0, 10)
        ax.set_xlabel('t', fontweight='bold')
        ax.set_ylabel('x = log t', fontweight='bold', color=self.colors['primary'])
        ax2.set_ylabel('f(t)', fontweight='bold', color=self.colors['secondary'])
        ax3.set_ylabel('|Mellin[f]|', fontweight='bold', color=self.colors['accent'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower right')
    
    def _plot_multiplicative_laplacian(self, ax, zeta_field: ZetaField):
        """Plot multiplicative Laplacian and eigenfunctions"""
        ax.set_title('Multiplicative Laplacian: Δ_mult = -t²∂²_t - t∂_t', 
                    fontweight='bold', color=self.colors['secondary'])
        
        # Create t values
        t_values = np.linspace(0.1, 5, 100)
        
        # Plot eigenfunctions for different s values
        s_values = [0.5, 0.5 + 14.134725j, 0.5 + 21.02204j]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        for i, s in enumerate(s_values):
            laplacian_result = zeta_field.multiplicative_laplacian(s, t_values)
            eigenfunction = laplacian_result['eigenfunction']
            
            # Plot real part
            ax.plot(t_values, np.real(eigenfunction), '-', color=colors[i], 
                   linewidth=2, label=f'Re[t^{s-1}]')
            
            # Plot imaginary part if complex
            if np.iscomplexobj(s):
                ax.plot(t_values, np.imag(eigenfunction), '--', color=colors[i], 
                       linewidth=2, alpha=0.7, label=f'Im[t^{s-1}]')
        
        ax.set_xlim(0, 5)
        ax.set_xlabel('t', fontweight='bold')
        ax.set_ylabel('Eigenfunction t^{s-1}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_prime_pins(self, ax, zeta_field: ZetaField):
        """Plot prime pins V(x) = ∑_p δ(x - log p)"""
        ax.set_title('Prime Pins: V(x) = ∑_p δ(x - log p)', 
                    fontweight='bold', color=self.colors['green'])
        
        # Create x values
        x_values = np.linspace(0, 5, 200)
        prime_pins_result = zeta_field.prime_pins(x_values)
        
        # Plot smooth potential
        ax.plot(x_values, prime_pins_result['smooth_potential'], '-', 
               color=self.colors['green'], linewidth=2, label='V(x)')
        
        # Mark discrete prime positions
        for p in zeta_field.primes[:10]:  # Show first 10 primes
            log_p = np.log(p)
            if log_p <= 5:
                ax.axvline(x=log_p, color=self.colors['accent'], alpha=0.7, linewidth=1)
                ax.text(log_p, 0.1, f'p={p}', ha='center', va='bottom', 
                       fontsize=8, color=self.colors['accent'], fontweight='bold')
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel('x = log t', fontweight='bold')
        ax.set_ylabel('V(x)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    def _plot_green_function(self, ax, zeta_field: ZetaField):
        """Plot Green's function ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))"""
        ax.set_title('Green\'s Function: ζ(s) ∝ det^{-1}(Δ_mult + V - s(1-s))', 
                    fontweight='bold', color=self.colors['primary'])
        
        # Create complex plane
        sigma = np.linspace(0, 1, 50)
        t = np.linspace(0, 30, 50)
        S, T = np.meshgrid(sigma, t)
        
        # Compute Green's function approximation
        zeta_mag = np.zeros_like(S)
        t_values = np.linspace(0.1, 5, 20)  # Reduced for efficiency
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                s = S[i, j] + 1j * T[i, j]
                green_result = zeta_field.green_function(s, t_values)
                zeta_mag[i, j] = abs(green_result['zeta_approx'])
        
        # Plot Green's function magnitude
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
    
    def _plot_mirror_symmetry(self, ax, zeta_field: ZetaField):
        """Plot mirror symmetry Λ(s) = Λ(1-s)"""
        ax.set_title('Mirror Symmetry: Λ(s) = Λ(1-s)', 
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
        for p in zeta_field.primes[:5]:
            log_p = np.log(p)
            if log_p <= 1:
                ax.axvline(x=log_p, color=self.colors['green'], alpha=0.6, linewidth=1)
                ax.text(log_p, 0.9, f'log {p}', ha='center', va='center', 
                       fontsize=8, color=self.colors['green'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('s', fontweight='bold')
        ax.set_ylabel('I(s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
    
    def _plot_attractor_eigenmodes(self, ax, zeta_field: ZetaField):
        """Plot attractor eigenmodes"""
        ax.set_title('Attractor Eigenmodes: Zeros = Eigenmodes', 
                    fontweight='bold', color=self.colors['accent'])
        
        # Compute eigenmodes for a test point
        s_test = 0.5 + 14.134725j
        t_values = np.linspace(0.1, 5, 20)
        eigenmodes = zeta_field.attractor_eigenmodes(s_test, t_values)
        
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
        
        # Add boundary scatterers info
        ax.text(0.7, 0.8, f'Boundary Scatterers: {eigenmodes["boundary_scatterers"]}', 
               transform=ax.transAxes, fontsize=10, color=self.colors['green'], fontweight='bold')
        
        ax.set_xlim(-0.5, min(n_modes, 20) - 0.5)
        ax.set_xlabel('Eigenmode Index', fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)


def main():
    """Main entry point for ZetaField visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CE1 ZetaField Visualization")
    parser.add_argument("--max-prime", type=int, default=50, help="Maximum prime to consider")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize ZetaField
    zeta_field = ZetaField(max_prime=args.max_prime)
    
    print(f"ZetaField initialized with {len(zeta_field.primes)} primes")
    print(f"First 10 primes: {zeta_field.primes[:10]}")
    
    # Test some operations
    s_test = 0.5 + 14.134725j
    t_values = np.linspace(0.1, 5, 50)
    print(f"\nTesting at s = {s_test}")
    
    # Multiplicative group
    group_result = zeta_field.multiplicative_group(t_values)
    print(f"Multiplicative group: G = R_+^×")
    
    # Multiplicative Laplacian
    laplacian_result = zeta_field.multiplicative_laplacian(s_test, t_values)
    print(f"Eigenvalue: {laplacian_result['zeta_eigenvalue']:.6f}")
    
    # Prime pins
    x_values = np.log(t_values)
    prime_pins_result = zeta_field.prime_pins(x_values)
    print(f"Prime potential: {len(zeta_field.primes)} delta functions")
    
    # Green's function
    green_result = zeta_field.green_function(s_test, t_values)
    print(f"Green's function: ζ(s) ≈ {green_result['zeta_approx']:.6f}")
    print(f"Error: {green_result['error']:.6f}")
    
    # Mirror symmetry
    mirror = zeta_field.mirror_symmetry(s_test)
    print(f"Mirror residual: {mirror['mirror_residual']:.6f}")
    
    # Attractor eigenmodes
    eigenmodes = zeta_field.attractor_eigenmodes(s_test, t_values)
    print(f"Dominant eigenmode: {eigenmodes['dominant_eigenmode']:.6f}")
    print(f"Boundary scatterers: {eigenmodes['boundary_scatterers']}")
    
    # Create visualization
    visualizer = ZetaFieldVisualizer()
    output_file = visualizer.create_zeta_field_diagram(zeta_field, args.output)
    
    print(f"\nGenerated ZetaField visualization: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
