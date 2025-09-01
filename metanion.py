#!/usr/bin/env python3
"""
metanion: Bootstrap module for core structures and spec decorator.

This module provides the essential Metanion field theory structures and @em decorator
that other modules can import without creating circular dependencies.

Core structures:
- Metanion: The fundamental particle in our field theory
- @em: The specification decorator for function contracts
- Spec: The specification tuple for pattern matching

This enables proper bootstrap order where modules can register their specs
before the full emits system is loaded.
"""

from __future__ import annotations
from collections import namedtuple
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import inspect
import hashlib

# Core Metanion structures
HilbertPoint = namedtuple('HilbertPoint', ['density', 'entropy', 'depth', 'diversity'])
SpatialEntry = namedtuple('SpatialEntry', ['path', 'data', 'coords', 'quat'])
Transform = namedtuple('Transform', ['f', 'U', 'w', 'c', 'y'])
Metanion = namedtuple('Metanion', ['S', 'T', 'm', 'alpha', 'Q', 'E'])

# Specification structure
Spec = namedtuple('Spec', ['raw', 'inputs', 'outputs', 'mint'])

# Global registries for cross-module access
_pending_specs: List[Tuple[str, str, Callable[..., Any]]] = []  # (func_name, pattern, func)
_genes: Dict[str, List[Spec]] = {}
_mint_seq: int = 1
_minted: Dict[str, int] = {}

def mint(spec: str) -> int:
    """Mint a unique type ID from a spec string."""
    global _mint_seq
    if spec not in _minted:
        _minted[spec] = _mint_seq
        _mint_seq += 1
    return _minted[spec]

def split(raw: str) -> Tuple[str, str]:
    """Split a spec string into inputs and outputs."""
    s = raw.strip()
    if ':=' in s:
        left, right = s.split(':=', 1)
        return left.strip(), right.strip()
    return s, ''

def split_commas(s: str) -> List[str]:
    """Split comma-separated values, respecting brackets."""
    items: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in s:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            item = ''.join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        items.append(tail)
    return items

def parse(s: str) -> Mapping[str, Any]:
    """Parse a spec string into structured data."""
    out: Dict[str, Any] = {}
    if not s:
        return out
    items = split_commas(s)
    for item in items:
        if ':' in item:
            k, v = item.split(':', 1)
            key = k.strip()
            val = v.strip()
            if val.lower() == 'present':
                out[key] = ('present', True)
            elif val.startswith('[') and val.endswith(']'):
                body = val[1:-1].strip()
                arr = [p.strip() for p in body.split(',')] if body else []
                out[key] = ('array', arr)
            else:
                # attempt numeric parse; default to atom
                try:
                    if '.' in val or 'e' in val.lower():
                        num = float(val)
                    else:
                        num = int(val)
                    out[key] = ('number', num)
                except ValueError:
                    out[key] = ('atom', val)
        else:
            # bare token means presence requirement
            out[item] = ('item', True)
    return out

def recog(raw: str) -> Spec:
    """Recognize a spec string and create a Spec object."""
    left, right = split(raw)
    inputs = parse(left)
    outputs = parse(right)
    return Spec(raw, inputs, outputs, mint(raw))

def em(pattern: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    The core @em decorator for function specifications.
    
    This decorator registers function specifications for pattern matching
    and validation. It works both during bootstrap (queuing specs) and
    after full system load (immediate registration).
    """
    spec = recog(pattern)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = getattr(fn, '__name__', 'anon')
        
        # Try to register immediately if emits is ready
        try:
            import emits
            if hasattr(emits, '_emits_ready') and emits._emits_ready:
                # Use the same Spec type as emits
                emits_spec = emits.Spec(spec.raw, spec.inputs, spec.outputs, spec.mint)
                emits.genes[name].append(emits_spec)
            else:
                # Queue for later registration
                _pending_specs.append((name, pattern, fn))
        except ImportError:
            # Emits not ready yet, queue locally
            _pending_specs.append((name, pattern, fn))
        
        return fn

    return decorate

def register_pending_specs():
    """Register all pending specs with the emits system."""
    try:
        import emits
        for func_name, pattern, func in _pending_specs:
            spec = recog(pattern)
            # Use the same Spec type as emits
            emits_spec = emits.Spec(spec.raw, spec.inputs, spec.outputs, spec.mint)
            emits.genes[func_name].append(emits_spec)
        _pending_specs.clear()
    except ImportError:
        # Emits not available, keep queued
        pass

def get_genes() -> Dict[str, List[Spec]]:
    """Get the current gene registry."""
    try:
        import emits
        # Convert emits.Spec to metanion.Spec for consistency
        converted_genes = {}
        for name, specs in emits.genes.items():
            converted_genes[name] = [Spec(s.raw, s.inputs, s.outputs, s.mint) for s in specs]
        return converted_genes
    except ImportError:
        return _genes

# Export the core structures
__all__ = [
    'Metanion', 'HilbertPoint', 'SpatialEntry', 'Transform',
    'Spec', 'em', 'mint', 'recog', 'register_pending_specs', 'get_genes'
]
