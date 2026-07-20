from __future__ import annotations

from collections import Counter
from typing import Optional

from src.arc_solver.types import Grid


def as_grid(rows) -> Grid:
    return tuple(tuple(int(v) for v in row) for row in rows)


def as_rows(g: Grid) -> list[list[int]]:
    return [list(row) for row in g]


def shape(g: Grid) -> tuple[int, int]:
    if not g:
        return (0, 0)
    return (len(g), len(g[0]))


def is_valid_grid(g: Optional[Grid]) -> bool:
    if g is None or not g:
        return False
    h, w = shape(g)
    if h < 1 or w < 1 or h > 30 or w > 30:
        return False
    if any(len(row) != w for row in g):
        return False
    return all(isinstance(v, int) and 0 <= v <= 9 for row in g for v in row)


def constant_grid(h: int, w: int, value: int) -> Grid:
    return tuple(tuple(value for _ in range(w)) for _ in range(h))


def colors(g: Grid) -> Counter[int]:
    return Counter(v for row in g for v in row)


def most_common_color(g: Grid) -> int:
    return colors(g).most_common(1)[0][0]


def nonzero_colors(g: Grid) -> list[int]:
    return [c for c in colors(g) if c != 0]


def positions_of(g: Grid, color: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, row in enumerate(g)
        for c, value in enumerate(row)
        if value == color
    }


def nonzero_positions(g: Grid) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, row in enumerate(g)
        for c, value in enumerate(row)
        if value != 0
    }


def bbox_for_positions(pos: set[tuple[int, int]]) -> Optional[tuple[int, int, int, int]]:
    if not pos:
        return None
    rs = [p[0] for p in pos]
    cs = [p[1] for p in pos]
    return min(rs), min(cs), max(rs), max(cs)


def crop_rect(g: Grid, box: tuple[int, int, int, int]) -> Grid:
    r0, c0, r1, c1 = box
    return tuple(tuple(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1))


def crop_non_bg(g: Grid, bg: int) -> Grid:
    pos = {
        (r, c)
        for r, row in enumerate(g)
        for c, value in enumerate(row)
        if value != bg
    }
    box = bbox_for_positions(pos)
    if box is None:
        return ((bg,),)
    return crop_rect(g, box)


def normalize_mask(pos: set[tuple[int, int]]) -> Optional[Grid]:
    box = bbox_for_positions(pos)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return tuple(
        tuple(1 if (r, c) in pos else 0 for c in range(c0, c1 + 1))
        for r in range(r0, r1 + 1)
    )


def dedupe_grids(grids: list[Grid]) -> list[Grid]:
    seen = set()
    out = []
    for g in grids:
        if not is_valid_grid(g):
            continue
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


# ----------------------------
