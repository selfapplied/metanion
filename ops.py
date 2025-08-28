from typing import Iterable, Callable, Optional

# Tiny operator kit that composes with sprixel
from sprixel import (
    signal,
    reflect,
    forms,
    shade,
    splash,
    neon,
    solar,
)

__all__ = ["symmetry", "stage", "mark", "promote"]


def symmetry(motif: Iterable[str] | str, kind: str = "reflect") -> Callable[[int], str]:
    """Minimal symmetry: build a line from motif, then mirror it.

    kind: only "reflect" is implemented for now, by design (keep small).
    """
    band = signal(motif)

    def at(width: int) -> str:
        line = "".join(band(int(width)))
        if kind == "reflect":
            return reflect(line, depth=1, drift=0, quiet=0.6)
        return line

    return at


def stage(*bands: tuple[int, Callable[[int], str]]) -> Callable[[int], str]:
    """Width-gated forms (thin wrapper)."""
    gate = forms(*bands)

    def at(width: int) -> str:
        return gate(width)(width)

    return at


def mark(text: str, style: str = "methyl", level: float = 0.6,
         pal: Callable[[float], tuple[int, int, int]] = solar) -> str:
    """Epigenetic tint:
    - methyl: dim to gray by level
    - acetyl: palette wash (soft)
    """
    if style == "acetyl":
        return "".join(splash(text, pal))
    # default methylation
    return "".join(shade(ch, level) for ch in text)


def promote(target: str, strength: float = 0.9,
            pal: Callable[[float], tuple[int, int, int]] = neon) -> Callable[[str], str]:
    """Promoter: if the target appears, lightly amplify with a palette wash."""

    def apply(text: str) -> str:
        if target and (target.lower() in text.lower()):
            return "".join(splash(text, pal))
        return text

    return apply


