import numpy as np
import math
from typing import Tuple, List, Dict
from collections import namedtuple

ColorInfo = namedtuple('ColorInfo', 'h s v')

def _extract_float_bits(value: float) -> tuple[int, int, float]:
    """Extract sign, exponent, and mantissa from IEEE 754 float for CE1 color representation"""
    if abs(value) < 1e-9:
        return 0, 0, 0.0
    
    b = np.float64(value).view(np.uint64)
    sign = int((b >> 63) & 1)
    exponent = int((b >> 52) & 0x7FF) - 1023
    mantissa = 1.0 + float(b & 0xFFFFFFFFFFFFF) / (1 << 52)
    
    return sign, exponent, mantissa

def _mantissa_to_triangle_coords(mantissa: float) -> tuple[float, float, float]:
    """Convert mantissa to Sierpiński triangle barycentric coordinates"""
    m_bits = int((mantissa - 1.0) * (1 << 52))
    coords = np.array([0.5, 0.5, 0.0])
    v = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    
    for i in range(26):
        idx = (m_bits >> (2 * i)) & 0b11
        if idx < 3:
            coords = (coords + v[idx]) / 2.0
            
    return (coords[0], coords[1], coords[2])

def _sierpinski_hsv_bead(delta: float, mantissa: float, exponent: int) -> ColorInfo:
    """Map delta to Sierpiński triangle HSV color space"""
    u, v, w = _mantissa_to_triangle_coords(mantissa)
    
    hue_base = (exponent % 12) / 12.0
    hue_shift = u * 0.1
    hue = (hue_base + hue_shift) % 1.0
    
    saturation = 0.6 + v * 0.4
    brightness = 0.5 + w * 0.5
    
    if delta < 0:
        brightness *= 0.8
        
    return ColorInfo(hue, saturation, brightness)

def delta_to_color(delta: float) -> ColorInfo:
    """Convert delta to CE1 color using Sierpiński HSV bead palette"""
    sign, exponent, mantissa = _extract_float_bits(delta)
    return _sierpinski_hsv_bead(delta, mantissa, exponent)

def apply_color_phase_allele(delta: float, phase_allele: Dict[str, float]) -> ColorInfo:
    """Apply a color phase allele to transform a delta value to color."""
    sign, exponent, mantissa = _extract_float_bits(delta)
    
    phase_hue_shift = phase_allele.get('hue_shift', 0.0)
    phase_sat_shift = phase_allele.get('saturation_shift', 0.0)
    phase_bright_shift = phase_allele.get('brightness_shift', 0.0)
    
    hue, sat, bright = _sierpinski_hsv_bead(delta, mantissa, exponent)
    
    phase_hue = (hue + phase_hue_shift) % 1.0
    phase_sat = np.clip(sat + phase_sat_shift, 0.1, 1.0)
    phase_bright = np.clip(bright + phase_bright_shift, 0.1, 1.0)
    
    return ColorInfo(phase_hue, phase_sat, phase_bright)
