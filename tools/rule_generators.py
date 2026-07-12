from __future__ import annotations

from typing import Callable

from src.arc_solver.types import Grid, Rule
from src.arc_solver.grid_utils import (
    shape,
    nonzero_colors,
    nonzero_positions,
    bbox_for_positions,
    is_valid_grid,
)

Train = list[tuple[Grid, Grid]]


def _rule(name: str, fn: Callable[[Grid], Grid | None]) -> Rule:
    return Rule(name=name, priority=10_000, predict=fn)


def _validate(train: Train, fn: Callable[[Grid], Grid | None]) -> bool:
    for inp, out in train:
        try:
            pred = fn(inp)
        except Exception:
            return False
        if not is_valid_grid(pred) or pred != out:
            return False
    return True


def _mutable(g: Grid) -> list[list[int]]:
    return [list(row) for row in g]


def _freeze(rows: list[list[int]]) -> Grid:
    return tuple(tuple(row) for row in rows)


# Family 1: align_bbox_to_edge

def _align_bbox(g: Grid, edge: str) -> Grid | None:
    h, w = shape(g)
    pos = nonzero_positions(g)
    box = bbox_for_positions(pos)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if edge == "top":
        dr, dc = -r0, 0
    elif edge == "bottom":
        dr, dc = h - 1 - r1, 0
    elif edge == "left":
        dr, dc = 0, -c0
    elif edge == "right":
        dr, dc = 0, w - 1 - c1
    else:
        return None
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in pos:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        out[nr][nc] = g[r][c]
    return _freeze(out)


def fit_align_bbox_to_edge(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    for edge in ("top", "bottom", "left", "right"):
        fn = lambda g, edge=edge: _align_bbox(g, edge)
        if _validate(train, fn):
            rules.append(_rule(f"gen_align_bbox_{edge}", fn))
    return rules


# Families 2 and 3: translate/copy color preserve rest

def _move_color(g: Grid, color: int, dr: int, dc: int, copy: bool) -> Grid | None:
    h, w = shape(g)
    src = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
    if not src:
        return None
    rows = _mutable(g)
    dests: list[tuple[int, int]] = []
    for r, c in src:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if g[nr][nc] not in (0, color):
            return None
        dests.append((nr, nc))
    if not copy:
        for r, c in src:
            rows[r][c] = 0
    for nr, nc in dests:
        rows[nr][nc] = color
    return _freeze(rows)


def _fit_move_color(train: Train, copy: bool) -> list[Rule]:
    if not train:
        return []
    h, w = shape(train[0][0])
    common_colors = set(nonzero_colors(train[0][0]))
    for inp, _ in train[1:]:
        common_colors &= set(nonzero_colors(inp))
    rules: list[Rule] = []
    for color in sorted(common_colors):
        for dr in range(-(h - 1), h):
            for dc in range(-(w - 1), w):
                if dr == 0 and dc == 0:
                    continue
                fn = lambda g, color=color, dr=dr, dc=dc, copy=copy: _move_color(g, color, dr, dc, copy)
                if _validate(train, fn):
                    suffix = "copy" if copy else "translate"
                    rules.append(_rule(f"gen_{suffix}_color_{color}_{dr}_{dc}_preserve_rest", fn))
    return rules


def fit_translate_color_preserve_rest(train: Train) -> list[Rule]:
    return _fit_move_color(train, copy=False)


def fit_copy_color_preserve_rest(train: Train) -> list[Rule]:
    return _fit_move_color(train, copy=True)


# Family 4: connect same-color pairs

def _connect_same_color_pairs(g: Grid) -> Grid | None:
    h, w = shape(g)
    rows = _mutable(g)
    for color in nonzero_colors(g):
        pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = pts
        if r0 == r1:
            for c in range(min(c0, c1), max(c0, c1) + 1):
                rows[r0][c] = color
        elif c0 == c1:
            for r in range(min(r0, r1), max(r0, r1) + 1):
                rows[r][c0] = color
    out = _freeze(rows)
    return out if shape(out) == (h, w) else None


def fit_connect_same_color_pairs(train: Train) -> list[Rule]:
    fn = _connect_same_color_pairs
    return [_rule("gen_connect_same_color_pairs", fn)] if _validate(train, fn) else []


# Family 5: dilate nonzero

def _neighbors(kind: int) -> list[tuple[int, int]]:
    base = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if kind == 8:
        base += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return base


def _dilate(g: Grid, kind: int, added_color: int | None) -> Grid | None:
    h, w = shape(g)
    rows = _mutable(g)
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == 0:
                continue
            for dr, dc in _neighbors(kind):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 0:
                    rows[nr][nc] = v if added_color is None else added_color
    return _freeze(rows)


def _learn_added_colors(train: Train) -> list[int]:
    added: set[int] = set()
    for inp, out in train:
        if shape(inp) != shape(out):
            return []
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                if v == 0 and out[r][c] != 0:
                    added.add(out[r][c])
                elif v != 0 and out[r][c] != v:
                    return []
    return sorted(added) if len(added) == 1 else []


def fit_dilate_nonzero(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    for kind in (4, 8):
        fn = lambda g, kind=kind: _dilate(g, kind, None)
        if _validate(train, fn):
            rules.append(_rule(f"gen_dilate_{kind}_same_color", fn))
    for color in _learn_added_colors(train):
        for kind in (4, 8):
            fn = lambda g, kind=kind, color=color: _dilate(g, kind, color)
            if _validate(train, fn):
                rules.append(_rule(f"gen_dilate_{kind}_added_color_{color}", fn))
    return rules


# Family 6: pad input to output

def _offset(mode: str, ih: int, iw: int, oh: int, ow: int) -> tuple[int, int] | None:
    if ih > oh or iw > ow:
        return None
    return {
        "top_left": (0, 0),
        "top_right": (0, ow - iw),
        "bottom_left": (oh - ih, 0),
        "bottom_right": (oh - ih, ow - iw),
        "center": ((oh - ih) // 2, (ow - iw) // 2),
    }.get(mode)


def _pad_to_shape(g: Grid, mode: str, oh: int, ow: int) -> Grid | None:
    ih, iw = shape(g)
    off = _offset(mode, ih, iw, oh, ow)
    if off is None:
        return None
    r0, c0 = off
    rows = [[0 for _ in range(ow)] for _ in range(oh)]
    for r in range(ih):
        for c in range(iw):
            rows[r0 + r][c0 + c] = g[r][c]
    return _freeze(rows)


def fit_pad_input_to_output(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    for mode in ("top_left", "top_right", "bottom_left", "bottom_right", "center"):
        shapes = [shape(out) for _, out in train]
        if not shapes:
            continue
        def fn(g: Grid, mode=mode, shapes=shapes) -> Grid | None:
            # For test grids, infer the same size delta as first training pair.
            ih0, iw0 = shape(train[0][0])
            oh0, ow0 = shapes[0]
            h, w = shape(g)
            return _pad_to_shape(g, mode, h + (oh0 - ih0), w + (ow0 - iw0))
        if _validate(train, fn):
            rules.append(_rule(f"gen_pad_{mode}", fn))
    return rules


# Family 7: repeat input tile

def _repeat_tile_to(g: Grid, mh: int, mw: int) -> Grid | None:
    h, w = shape(g)
    return tuple(tuple(g[r % h][c % w] for c in range(w * mw)) for r in range(h * mh))


def fit_repeat_input_tile(train: Train) -> list[Rule]:
    if not train:
        return []
    ih, iw = shape(train[0][0])
    oh, ow = shape(train[0][1])
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return []
    mh, mw = oh // ih, ow // iw
    fn = lambda g, mh=mh, mw=mw: _repeat_tile_to(g, mh, mw)
    return [_rule("gen_repeat_input_tile", fn)] if _validate(train, fn) else []


FAMILY_FITTERS = {
    "align_bbox_to_edge": fit_align_bbox_to_edge,
    "translate_color_preserve_rest": fit_translate_color_preserve_rest,
    "copy_color_preserve_rest": fit_copy_color_preserve_rest,
    "connect_same_color_pairs": fit_connect_same_color_pairs,
    "dilate_nonzero": fit_dilate_nonzero,
    "pad_input_to_output": fit_pad_input_to_output,
    "repeat_input_tile": fit_repeat_input_tile,
}

FAMILY_ALIASES = {
    "align": "align_bbox_to_edge",
    "translate": "translate_color_preserve_rest",
    "copy": "copy_color_preserve_rest",
    "connect": "connect_same_color_pairs",
    "dilate": "dilate_nonzero",
    "pad": "pad_input_to_output",
    "repeat": "repeat_input_tile",
    "tile": "repeat_input_tile",
}
