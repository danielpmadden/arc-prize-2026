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


# Family 8: remove small components

def _components4(g: Grid) -> list[list[tuple[int, int]]]:
    h, w = shape(g)
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            color = g[r][c]
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and g[nr][nc] == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _remove_components(g: Grid, mode: str) -> Grid | None:
    comps = _components4(g)
    if not comps:
        return None
    sizes = [len(c) for c in comps]
    largest = max(sizes)
    common = max(sorted(set(sizes)), key=lambda n: (sizes.count(n), -n))
    rows = _mutable(g)
    for comp in comps:
        n = len(comp)
        remove = (
            (mode == "eq1" and n == 1)
            or (mode == "le2" and n <= 2)
            or (mode == "le3" and n <= 3)
            or (mode == "largest" and n != largest)
            or (mode == "common" and n != common)
        )
        if remove:
            for r, c in comp:
                rows[r][c] = 0
    return _freeze(rows)


def fit_remove_small_components(train: Train) -> list[Rule]:
    specs = [
        ("gen_remove_components_size_eq_1", "eq1"),
        ("gen_remove_components_size_le_2", "le2"),
        ("gen_remove_components_size_le_3", "le3"),
        ("gen_keep_largest_component", "largest"),
        ("gen_keep_most_common_component_size", "common"),
    ]
    rules = []
    for name, mode in specs:
        fn = lambda g, mode=mode: _remove_components(g, mode)
        if _validate(train, fn):
            rules.append(_rule(name, fn))
    return rules


# Family 9: recolor components by size

def fit_recolor_components_by_size(train: Train) -> list[Rule]:
    mapping: dict[int, int] = {}
    changed = False
    for inp, out in train:
        if shape(inp) != shape(out):
            return []
        for comp in _components4(inp):
            vals = {out[r][c] for r, c in comp}
            if len(vals) != 1:
                return []
            out_color = next(iter(vals))
            in_vals = {inp[r][c] for r, c in comp}
            if len(in_vals) != 1:
                return []
            if out_color == next(iter(in_vals)):
                continue
            size = len(comp)
            if size in mapping and mapping[size] != out_color:
                return []
            mapping[size] = out_color
            changed = True
    if not changed:
        return []
    def fn(g: Grid, mapping=mapping) -> Grid | None:
        rows = _mutable(g)
        for comp in _components4(g):
            if len(comp) in mapping:
                for r, c in comp:
                    rows[r][c] = mapping[len(comp)]
        return _freeze(rows)
    return [_rule("gen_recolor_components_by_size", fn)] if _validate(train, fn) else []


# Family 10: crop to nonzero bbox

def _crop_positions(g: Grid, pos: list[tuple[int, int]]) -> Grid | None:
    box = bbox_for_positions(pos)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return tuple(tuple(g[r][c] for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))


def fit_crop_to_nonzero_bbox(train: Train) -> list[Rule]:
    rules = []
    fn = lambda g: _crop_positions(g, nonzero_positions(g))
    if _validate(train, fn):
        rules.append(_rule("gen_crop_all_nonzero_bbox", fn))
    common = set(nonzero_colors(train[0][0])) if train else set()
    for inp, _ in train[1:]:
        common &= set(nonzero_colors(inp))
    for color in sorted(common):
        fn = lambda g, color=color: _crop_positions(g, [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color])
        if _validate(train, fn):
            rules.append(_rule(f"gen_crop_color_{color}_bbox", fn))
    return rules


def _sep_rows(g: Grid) -> list[int]:
    return [i for i, row in enumerate(g) if row and row[0] != 0 and all(v == row[0] for v in row)]


def _sep_cols(g: Grid) -> list[int]:
    h, w = shape(g)
    return [c for c in range(w) if h and g[0][c] != 0 and all(g[r][c] == g[0][c] for r in range(h))]


def _strip(g: Grid, rows: bool, cols: bool) -> Grid | None:
    sr = set(_sep_rows(g)) if rows else set()
    sc = set(_sep_cols(g)) if cols else set()
    out = [[v for c, v in enumerate(row) if c not in sc] for r, row in enumerate(g) if r not in sr]
    return _freeze(out) if out and out[0] else None


def fit_strip_separator_lines(train: Train) -> list[Rule]:
    specs = [("gen_strip_full_separator_rows", True, False), ("gen_strip_full_separator_cols", False, True), ("gen_strip_full_separator_rows_cols", True, True)]
    rules = []
    for name, rows, cols in specs:
        fn = lambda g, rows=rows, cols=cols: _strip(g, rows, cols)
        if _validate(train, fn):
            rules.append(_rule(name, fn))
    return rules


def _spans(n: int, seps: list[int]) -> list[tuple[int, int]]:
    cut = [-1] + seps + [n]
    return [(cut[i] + 1, cut[i + 1]) for i in range(len(cut) - 1) if cut[i] + 1 < cut[i + 1]]


def _select_panel(g: Grid, axis: str, which: str) -> Grid | None:
    h, w = shape(g)
    spans = _spans(w, _sep_cols(g)) if axis == "col" else _spans(h, _sep_rows(g))
    if not spans:
        return None
    idx = len(spans) - 1 if which == "last" else int(which)
    if idx >= len(spans):
        return None
    a, b = spans[idx]
    if axis == "col":
        return tuple(tuple(row[c] for c in range(a, b)) for row in g)
    return tuple(tuple(g[r][c] for c in range(w)) for r in range(a, b))


def fit_select_panel_by_index(train: Train) -> list[Rule]:
    rules = []
    for axis in ("col", "row"):
        for which in ("0", "1", "last"):
            fn = lambda g, axis=axis, which=which: _select_panel(g, axis, which)
            if _validate(train, fn):
                rules.append(_rule(f"gen_select_panel_{axis}_{which}", fn))
    return rules


def _two_panels(g: Grid) -> tuple[Grid, Grid] | None:
    h, w = shape(g)
    for axis, seps in (("col", _sep_cols(g)), ("row", _sep_rows(g))):
        spans = _spans(w if axis == "col" else h, seps)
        if len(spans) == 2:
            a0, b0 = spans[0]; a1, b1 = spans[1]
            p0 = tuple(tuple(row[c] for c in range(a0, b0)) for row in g) if axis == "col" else tuple(tuple(g[r][c] for c in range(w)) for r in range(a0, b0))
            p1 = tuple(tuple(row[c] for c in range(a1, b1)) for row in g) if axis == "col" else tuple(tuple(g[r][c] for c in range(w)) for r in range(a1, b1))
            if shape(p0) == shape(p1):
                return p0, p1
    return None


def _overlay(g: Grid, mode: str, color: int | None = None) -> Grid | None:
    panels = _two_panels(g)
    if panels is None:
        return None
    a, b = panels; h, w = shape(a)
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            x, y = a[r][c], b[r][c]
            if mode == "or": v = x or y
            elif mode == "and": v = x if x and y else 0
            elif mode == "xor": v = x if x and not y else (y if y and not x else 0)
            else: v = color if x and y else 0
            row.append(v)
        rows.append(row)
    return _freeze(rows)


def fit_overlay_two_panels(train: Train) -> list[Rule]:
    rules = []
    for mode in ("or", "and", "xor"):
        fn = lambda g, mode=mode: _overlay(g, mode)
        if _validate(train, fn):
            rules.append(_rule(f"gen_overlay_panels_{mode}", fn))
    colors = set(range(1, 10))
    for color in colors:
        fn = lambda g, color=color: _overlay(g, "color", color)
        if _validate(train, fn):
            rules.append(_rule(f"gen_overlay_panels_intersection_color_{color}", fn))
    return rules


def _replace(g: Grid, mode: str, a: int | None, b: int) -> Grid | None:
    if mode == "one": return tuple(tuple(b if v == a else v for v in row) for row in g)
    if mode == "nonzero": return tuple(tuple(b if v != 0 else 0 for v in row) for row in g)
    return tuple(tuple(b if v == 0 else v for v in row) for row in g)


def fit_replace_color(train: Train) -> list[Rule]:
    candidates: set[tuple[str, int | None, int]] = set()
    for inp, out in train:
        if shape(inp) != shape(out):
            return []
        diffs = {(inp[r][c], out[r][c]) for r, row in enumerate(inp) for c, _ in enumerate(row) if inp[r][c] != out[r][c]}
        for a, b in diffs:
            candidates.add(("one", a, b))
        out_nonzero = {out[r][c] for r, row in enumerate(inp) for c, v in enumerate(row) if v != 0}
        if len(out_nonzero) == 1:
            candidates.add(("nonzero", None, next(iter(out_nonzero))))
        out_zero_replacements = {out[r][c] for r, row in enumerate(inp) for c, v in enumerate(row) if v == 0}
        if out_zero_replacements and 0 not in out_zero_replacements and len(out_zero_replacements) == 1:
            candidates.add(("zero", None, next(iter(out_zero_replacements))))
    rules = []
    for mode, a, b in sorted(candidates, key=lambda x: (x[0], -1 if x[1] is None else x[1], x[2])):
        if a == b:
            continue
        fn = lambda g, mode=mode, a=a, b=b: _replace(g, mode, a, b)
        if _validate(train, fn):
            if mode == "one":
                name = f"gen_replace_color_{a}_with_{b}"
            elif mode == "nonzero":
                name = f"gen_replace_all_nonzero_with_{b}"
            else:
                name = f"gen_replace_zero_with_{b}"
            rules.append(_rule(name, fn))
    return rules

def _mirror_copy(g: Grid, mode: str) -> Grid | None:
    h, w = shape(g); rows = _mutable(g)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v == 0: continue
            nr, nc = r, c
            if mode == "lr" and c < w // 2: nc = w - 1 - c
            elif mode == "rl" and c >= (w + 1) // 2: nc = w - 1 - c
            elif mode == "tb" and r < h // 2: nr = h - 1 - r
            elif mode == "bt" and r >= (h + 1) // 2: nr = h - 1 - r
            else: continue
            if rows[nr][nc] == 0: rows[nr][nc] = v
    return _freeze(rows)


def fit_mirror_copy_nonzero(train: Train) -> list[Rule]:
    specs = [("gen_copy_mirror_left_to_right", "lr"), ("gen_copy_mirror_right_to_left", "rl"), ("gen_copy_mirror_top_to_bottom", "tb"), ("gen_copy_mirror_bottom_to_top", "bt")]
    return [_rule(name, fn) for name, mode in specs for fn in [lambda g, mode=mode: _mirror_copy(g, mode)] if _validate(train, fn)]


def _extend_lines(g: Grid, mode: str) -> Grid | None:
    h, w = shape(g); rows = _mutable(g)
    for color in nonzero_colors(g):
        pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == color]
        rs, cs = {r for r, _ in pts}, {c for _, c in pts}
        dirs = []
        if len(rs) == 1 and mode in ("h", "both"): dirs += [(0, -1), (0, 1)]
        if len(cs) == 1 and mode in ("v", "both"): dirs += [(-1, 0), (1, 0)]
        for r, c in pts:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 0:
                    rows[nr][nc] = color; nr += dr; nc += dc
    return _freeze(rows)


def fit_extend_lines_until_blocked(train: Train) -> list[Rule]:
    specs = [("gen_extend_lines_horizontal", "h"), ("gen_extend_lines_vertical", "v"), ("gen_extend_lines_both", "both")]
    return [_rule(name, fn) for name, mode in specs for fn in [lambda g, mode=mode: _extend_lines(g, mode)] if _validate(train, fn)]


FAMILY_FITTERS = {
    "align_bbox_to_edge": fit_align_bbox_to_edge,
    "translate_color_preserve_rest": fit_translate_color_preserve_rest,
    "copy_color_preserve_rest": fit_copy_color_preserve_rest,
    "connect_same_color_pairs": fit_connect_same_color_pairs,
    "dilate_nonzero": fit_dilate_nonzero,
    "pad_input_to_output": fit_pad_input_to_output,
    "repeat_input_tile": fit_repeat_input_tile,
    "remove_small_components": fit_remove_small_components,
    "recolor_components_by_size": fit_recolor_components_by_size,
    "crop_to_nonzero_bbox": fit_crop_to_nonzero_bbox,
    "strip_separator_lines": fit_strip_separator_lines,
    "select_panel_by_index": fit_select_panel_by_index,
    "overlay_two_panels": fit_overlay_two_panels,
    "replace_color": fit_replace_color,
    "mirror_copy_nonzero": fit_mirror_copy_nonzero,
    "extend_lines_until_blocked": fit_extend_lines_until_blocked,
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
    "remove_components": "remove_small_components",
    "components_size": "recolor_components_by_size",
    "crop_bbox": "crop_to_nonzero_bbox",
    "strip": "strip_separator_lines",
    "panel": "select_panel_by_index",
    "overlay": "overlay_two_panels",
    "replace": "replace_color",
    "mirror": "mirror_copy_nonzero",
    "extend_lines": "extend_lines_until_blocked",
}
