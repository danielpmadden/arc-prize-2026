from __future__ import annotations

from typing import Callable, Optional

from src.arc_solver.types import Grid
from src.arc_solver.grid_utils import shape


def identity(g: Grid) -> Grid:
    return g


def rot90(g: Grid) -> Grid:
    h, w = shape(g)
    return tuple(tuple(g[h - 1 - r][c] for r in range(h)) for c in range(w))


def rot180(g: Grid) -> Grid:
    return rot90(rot90(g))


def rot270(g: Grid) -> Grid:
    return rot90(rot180(g))


def flip_h(g: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in g)


def flip_v(g: Grid) -> Grid:
    return tuple(reversed(g))


def transpose(g: Grid) -> Grid:
    h, w = shape(g)
    return tuple(tuple(g[r][c] for r in range(h)) for c in range(w))


GEOMS: list[tuple[str, Callable[[Grid], Grid]]] = [
    ("identity", identity),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
]


# ----------------------------
# Color-map helpers
# ----------------------------

def infer_color_map(srcs: list[Grid], outs: list[Grid]) -> Optional[dict[int, int]]:
    mapping: dict[int, int] = {}

    for src, out in zip(srcs, outs):
        if shape(src) != shape(out):
            return None

        for r in range(shape(src)[0]):
            for c in range(shape(src)[1]):
                a = src[r][c]
                b = out[r][c]
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b

    return mapping


def apply_color_map(g: Grid, mapping: dict[int, int]) -> Grid:
    return tuple(tuple(mapping.get(v, v) for v in row) for row in g)
