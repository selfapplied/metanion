"""
Central style vector and enforcement utilities.

This module provides a single source of truth for style preferences across
text generation and visual rendering. It supports a strict override mode to
force global choices, or a cooperative mode where local choices are allowed
but nudged toward the global vector.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


# Global style state (kept simple and explicit)
_STYLE_VECTOR: Dict[str, Any] = {
    # Text generation
    'text.style': 'choreographed',  # choreographed | pheno | ...
    'text.length': None,            # optional default length

    # Visual rendering
    'visual.style': 'acetyl',       # for sprixel2.mark or similar
    'visual.palette': None,         # function or palette object
}

_STRICT: bool = False


def set_style_vector(preferences: Dict[str, Any]) -> None:
    """Update the global style vector with provided preferences."""
    for k, v in (preferences or {}).items():
        _STYLE_VECTOR[k] = v


def get_style_value(key: str, default: Optional[Any] = None) -> Any:
    """Get a style attribute with optional default."""
    return _STYLE_VECTOR.get(key, default)


def set_strict_enforcement(strict: bool) -> None:
    """Enable or disable strict enforcement globally."""
    global _STRICT
    _STRICT = bool(strict)


def is_strict() -> bool:
    return _STRICT


def resolve_text_style(local_style: Optional[str] = None, length: Optional[int] = None) -> tuple[str, Optional[int]]:
    """Resolve text style and length based on strictness and global vector."""
    global_style = get_style_value('text.style', None)
    global_len = get_style_value('text.length', None)
    if is_strict():
        return (global_style or (local_style or 'choreographed'), global_len if global_len is not None else length)
    # cooperative: prefer local, fall back to global
    style = local_style if local_style is not None else (global_style or 'choreographed')
    final_len = length if length is not None else global_len
    return (style, final_len)


def resolve_visual_style(local_style: Optional[str] = None, palette: Optional[Any] = None) -> tuple[str, Optional[Any]]:
    """Resolve visual style and palette for rendering."""
    g_style = get_style_value('visual.style', None)
    g_pal = get_style_value('visual.palette', None)
    if is_strict():
        return (g_style or (local_style or 'acetyl'), g_pal if g_pal is not None else palette)
    # cooperative
    out_style = local_style if local_style is not None else (g_style or 'acetyl')
    out_pal = palette if palette is not None else g_pal
    return (out_style, out_pal)


__all__ = [
    'set_style_vector', 'get_style_value', 'set_strict_enforcement', 'is_strict',
    'resolve_text_style', 'resolve_visual_style', 'attune_text', 'attune_visual',
]

# Resonant naming: attune_* as preferred API
def attune_text(local_style: Optional[str] = None, length: Optional[int] = None):
    return resolve_text_style(local_style, length)


def attune_visual(local_style: Optional[str] = None, palette: Optional[Any] = None):
    return resolve_visual_style(local_style, palette)


