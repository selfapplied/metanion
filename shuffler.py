#!/usr/bin/env python3
"""
shuffler: move/reorder code with awareness and verification.

- AST-guided, text-preserving moves of top-level defs within a file.
- Minimal Gene/Virus abstractions for source transforms with verification.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# --- Utilities ---------------------------------------------------------------

def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding='utf-8')


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def _top_level_defs_with_spans(src: str) -> List[Tuple[str, int, int]]:
    """Return (name, start_lineno, end_lineno) for top-level defs in order.

    Includes function and class definitions. Uses end_lineno to preserve text.
    """
    tree = ast.parse(src)
    spans: List[Tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = getattr(node, 'name', '')
            start = getattr(node, 'lineno', 0)
            end = getattr(node, 'end_lineno', start)
            if name and start and end:
                spans.append((name, int(start), int(end)))
    return spans


def _extract_blocks_by_span(src: str, spans: List[Tuple[str, int, int]]) -> Dict[str, str]:
    lines = src.splitlines(keepends=True)
    blocks: Dict[str, str] = {}
    for name, start, end in spans:
        # Convert 1-based to 0-based indices
        s = max(0, start - 1)
        e = min(len(lines), end)
        blocks[name] = ''.join(lines[s:e])
    return blocks


def move_def(filename: str, name: str, *, before: Optional[str] = None, after: Optional[str] = None) -> bool:
    """Reorder a top-level def/class definition within the file.

    - name: symbol to move
    - before/after: reference symbol to position relative to (mutually exclusive)
    Returns True on edit, False if no-op.
    """
    path = pathlib.Path(filename)
    src = _read(path)
    spans = _top_level_defs_with_spans(src)
    order = [n for n, _, _ in spans]
    if name not in order:
        return False
    if before and after:
        raise ValueError('Specify only one of before/after')
    if before and before not in order:
        return False
    if after and after not in order:
        return False

    # Compute new order
    cur_idx = order.index(name)
    new_order = [x for x in order if x != name]
    if before:
        idx = new_order.index(before)
        new_order.insert(idx, name)
    elif after:
        idx = new_order.index(after) + 1
        new_order.insert(idx, name)
    else:
        # default: move to top
        new_order.insert(0, name)

    if new_order == order:
        return False

    # Rebuild text: stitch blocks in new order, keeping non-def text around them
    # Approach: treat regions outside defs as background and keep as-is by slicing.
    # Build mapping of spans and blocks.
    name_to_span = {n: (s, e) for n, s, e in spans}
    blocks = _extract_blocks_by_span(src, spans)

    # Create a mask over lines that belong to defs
    lines = src.splitlines(keepends=True)
    owned = [False] * len(lines)
    for _, s, e in spans:
        for i in range(max(0, s - 1), min(len(lines), e)):
            owned[i] = True

    background: List[str] = []
    for i, line in enumerate(lines):
        if not owned[i]:
            background.append(line)

    # Stitch: background with placeholders replaced by reordered blocks
    # Simplify: collapse all def regions and insert reordered blocks between background chunks.
    out: List[str] = []
    i = 0
    while i < len(lines):
        if owned[i]:
            # skip contiguous owned region
            j = i
            while j < len(lines) and owned[j]:
                j += 1
            # only insert blocks once at first owned region
            for nm in new_order:
                out.append(blocks[nm])
            i = j
        else:
            out.append(lines[i])
            i += 1

    new_src = ''.join(out)
    if new_src == src:
        return False
    _write(path, new_src)
    return True


# --- Gene / Virus abstractions ----------------------------------------------

VerifyFn = Callable[[str], bool]
ApplyFn = Callable[[str], str]


@dataclass
class Gene:
    name: str
    apply: ApplyFn
    verify: Optional[VerifyFn] = None

    def run(self, src: str) -> str:
        out = self.apply(src)
        if self.verify and not self.verify(out):
            raise ValueError(f"Gene '{self.name}' verification failed")
        return out


@dataclass
class Virus:
    name: str
    genes: List[Gene]

    def run(self, src: str) -> str:
        for g in self.genes:
            src = g.run(src)
        return src


_GENE_REGISTRY: Dict[str, Gene] = {}


def register_gene(gene: Gene) -> None:
    _GENE_REGISTRY[gene.name] = gene


def run_gene_on_file(filename: str, gene_name: str) -> bool:
    path = pathlib.Path(filename)
    if gene_name not in _GENE_REGISTRY:
        return False
    src = _read(path)
    out = _GENE_REGISTRY[gene_name].run(src)
    if out == src:
        return False
    _write(path, out)
    return True


def run_virus_on_file(filename: str, virus: Virus) -> bool:
    path = pathlib.Path(filename)
    src = _read(path)
    out = virus.run(src)
    if out == src:
        return False
    _write(path, out)
    return True


# --- Example builtin genes ---------------------------------------------------

def _strip_trailing_ws(src: str) -> str:
    # Preserve indentation style; only trim end-of-line whitespace
    return '\n'.join([line.rstrip('\r\t ') for line in src.splitlines()]) + ('\n' if src.endswith('\n') else '')


register_gene(Gene('strip_trailing_ws', _strip_trailing_ws))



