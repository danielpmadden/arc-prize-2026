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


# Shared helpers for experimental meta-families

def _learn_color_map_for_pairs(pred_out_pairs: list[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for pred, out in pred_out_pairs:
        if not is_valid_grid(pred) or shape(pred) != shape(out):
            return None
        for r, row in enumerate(pred):
            for c, src in enumerate(row):
                dst = out[r][c]
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    return mapping


def _apply_color_map(g: Grid, mapping: dict[int, int]) -> Grid | None:
    return tuple(tuple(mapping.get(v, v) for v in row) for row in g)


def _d4(g: Grid, op: str) -> Grid | None:
    h, w = shape(g)
    if h == 0 or w == 0:
        return None
    if op == "identity":
        return g
    if op == "rot90":
        return tuple(tuple(g[h - 1 - r][c] for r in range(h)) for c in range(w))
    if op == "rot180":
        return tuple(tuple(g[h - 1 - r][w - 1 - c] for c in range(w)) for r in range(h))
    if op == "rot270":
        return tuple(tuple(g[r][w - 1 - c] for r in range(h)) for c in range(w))
    if op == "flip_h":
        return tuple(tuple(row[w - 1 - c] for c in range(w)) for row in g)
    if op == "flip_v":
        return tuple(tuple(g[h - 1 - r][c] for c in range(w)) for r in range(h))
    if op == "transpose":
        return tuple(tuple(g[r][c] for r in range(h)) for c in range(w))
    if op == "anti_transpose":
        return tuple(tuple(g[h - 1 - r][w - 1 - c] for r in range(h)) for c in range(w))
    return None


_D4_INVERSE = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


_D4_BASE_FAMILIES = (
    "connect_same_color_pairs",
    "dilate_nonzero",
    "remove_small_components",
    "recolor_components_by_size",
    "crop_to_nonzero_bbox",
    "strip_separator_lines",
    "select_panel_by_index",
    "overlay_two_panels",
    "mirror_copy_nonzero",
    "extend_lines_until_blocked",
)


def fit_d4_conjugated_existing(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    seen: set[str] = set()
    for op in _D4_INVERSE:
        try:
            transformed = []
            for inp, out in train:
                tinp = _d4(inp, op)
                tout = _d4(out, op)
                if tinp is None or tout is None:
                    transformed = []
                    break
                transformed.append((tinp, tout))
            if not transformed:
                continue
            for family in _D4_BASE_FAMILIES:
                fitter = FAMILY_FITTERS.get(family)
                if fitter is None or fitter is fit_d4_conjugated_existing:
                    continue
                try:
                    base_rules = fitter(transformed)
                except Exception:
                    continue
                for base_rule in base_rules:
                    name = f"gen_d4_{op}__{family}__{base_rule.name}"
                    if name in seen:
                        continue
                    inv_op = _D4_INVERSE[op]
                    def fn(g: Grid, op=op, inv_op=inv_op, base_rule=base_rule) -> Grid | None:
                        tg = _d4(g, op)
                        if tg is None:
                            return None
                        pred = base_rule.predict(tg)
                        if not is_valid_grid(pred):
                            return None
                        return _d4(pred, inv_op)  # type: ignore[arg-type]
                    if _validate(train, fn):
                        rules.append(_rule(name, fn))
                        seen.add(name)
        except Exception:
            continue
    return rules


_STRUCTURE_COLOR_BASE_FAMILIES = (
    "crop_to_nonzero_bbox",
    "pad_input_to_output",
    "repeat_input_tile",
    "overlay_two_panels",
    "select_panel_by_index",
    "strip_separator_lines",
    "mirror_copy_nonzero",
    "extend_lines_until_blocked",
)


def fit_structure_then_color_map(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    seen_outputs: set[tuple[Grid, ...]] = set()
    for family in _STRUCTURE_COLOR_BASE_FAMILIES:
        fitter = FAMILY_FITTERS.get(family)
        if fitter is None:
            continue
        try:
            base_rules = fitter(train)
        except Exception:
            continue
        for base_rule in base_rules:
            try:
                preds = [(base_rule.predict(inp), out) for inp, out in train]
                if all(pred == out for pred, out in preds):
                    # Exact candidates are already represented by their base family.
                    continue
                mapping = _learn_color_map_for_pairs(preds)  # type: ignore[arg-type]
                if mapping is None:
                    continue
                def fn(g: Grid, base_rule=base_rule, mapping=mapping) -> Grid | None:
                    pred = base_rule.predict(g)
                    if not is_valid_grid(pred):
                        return None
                    return _apply_color_map(pred, mapping)  # type: ignore[arg-type]
                if not _validate(train, fn):
                    continue
                train_sig = tuple(fn(inp) for inp, _ in train)
                if train_sig in seen_outputs:
                    continue
                seen_outputs.add(train_sig)
                rules.append(_rule(f"gen_structure_then_color_map__{family}__{base_rule.name}", fn))
            except Exception:
                continue
    return rules


def _background_color(g: Grid, mode: str) -> int | None:
    h, w = shape(g)
    vals = [v for row in g for v in row]
    if not vals:
        return None
    if mode == "zero":
        return 0
    if mode == "most_common":
        return max(sorted(set(vals)), key=lambda v: (vals.count(v), -v))
    if mode == "corner":
        corners = [g[0][0], g[0][w - 1], g[h - 1][0], g[h - 1][w - 1]]
        return max(sorted(set(corners)), key=lambda v: (corners.count(v), -v))
    if mode == "border_most_common":
        border = []
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    border.append(g[r][c])
        return max(sorted(set(border)), key=lambda v: (border.count(v), -v)) if border else None
    return None


def _crop_non_bg_bbox(g: Grid, mode: str) -> Grid | None:
    bg = _background_color(g, mode)
    if bg is None:
        return None
    pos = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not pos:
        return None
    return _crop_positions(g, pos)


def fit_crop_to_background_bbox(train: Train) -> list[Rule]:
    specs = [("zero", "zero"), ("most_common", "most_common"), ("corner", "corner"), ("border_most_common", "border_most_common")]
    rules: list[Rule] = []
    for slug, mode in specs:
        fn = lambda g, mode=mode: _crop_non_bg_bbox(g, mode)
        if _validate(train, fn):
            rules.append(_rule(f"gen_crop_non_bg_bbox_{slug}", fn))
        try:
            pairs = [(fn(inp), out) for inp, out in train]
            mapping = _learn_color_map_for_pairs(pairs)  # type: ignore[arg-type]
            if mapping is None:
                continue
            cmap_fn = lambda g, mode=mode, mapping=mapping: (None if _crop_non_bg_bbox(g, mode) is None else _apply_color_map(_crop_non_bg_bbox(g, mode), mapping))
            if _validate(train, cmap_fn):
                rules.append(_rule(f"gen_crop_non_bg_bbox_{slug}_color_map", cmap_fn))
        except Exception:
            continue
    return rules

# Experimental family: recolor components by rank

def _comp_info(g: Grid, comp: list[tuple[int, int]]) -> dict:
    rs = [r for r, _ in comp]; cs = [c for _, c in comp]
    return {"cells": comp, "r0": min(rs), "c0": min(cs), "r1": max(rs), "c1": max(cs), "area": len(comp), "h": max(rs)-min(rs)+1, "w": max(cs)-min(cs)+1}


def _sort_infos(infos: list[dict], mode: str) -> list[dict]:
    rev = mode.endswith("descending") or mode in ("right_to_left", "bottom_to_top")
    if mode.startswith("area_"): key = lambda x: (x["area"], x["r0"], x["c0"])
    elif mode.startswith("height_"): key = lambda x: (x["h"], x["r0"], x["c0"])
    elif mode.startswith("width_"): key = lambda x: (x["w"], x["r0"], x["c0"])
    elif mode in ("left_to_right", "right_to_left"): key = lambda x: (x["c0"], x["r0"], x["area"])
    else: key = lambda x: (x["r0"], x["c0"], x["area"])
    return sorted(infos, key=key, reverse=rev)


def _fit_rank_recolor(train: Train, modes: tuple[str, ...], bars: bool = False) -> list[Rule]:
    rules: list[Rule] = []
    for mode in modes:
        mapping: dict[int, int] = {}; changed = False; ok = True
        for inp, out in train:
            if shape(inp) != shape(out): ok = False; break
            comps = [_comp_info(inp, c) for c in _components4(inp)]
            if bars:
                want_v = mode.startswith("vertical_"); want_h = mode.startswith("horizontal_")
                comps = [x for x in comps if (want_v and x["h"] > 1 and x["w"] == 1) or (want_h and x["h"] == 1 and x["w"] > 1)]
                smode = mode.replace("vertical_bars_", "").replace("horizontal_bars_", "")
            else:
                smode = mode
            bar_cells = {p for x in comps for p in x["cells"]}
            if bars:
                for r, row in enumerate(inp):
                    for c, v in enumerate(row):
                        if (r, c) not in bar_cells and out[r][c] != v: ok = False; break
                    if not ok: break
                if not ok: break
            else:
                for r, row in enumerate(inp):
                    for c, v in enumerate(row):
                        if v == 0 and out[r][c] != 0: ok = False; break
                    if not ok: break
                if not ok: break
            for idx, info in enumerate(_sort_infos(comps, smode)):
                vals = {out[r][c] for r, c in info["cells"]}
                if len(vals) != 1 or 0 in vals: ok = False; break
                col = next(iter(vals))
                if idx in mapping and mapping[idx] != col: ok = False; break
                mapping[idx] = col
                if any(inp[r][c] != col for r, c in info["cells"]): changed = True
            if not ok: break
        if not ok or not changed: continue
        def fn(g: Grid, mode=mode, mapping=mapping, bars=bars) -> Grid | None:
            comps = [_comp_info(g, c) for c in _components4(g)]
            if bars:
                want_v = mode.startswith("vertical_"); want_h = mode.startswith("horizontal_")
                comps = [x for x in comps if (want_v and x["h"] > 1 and x["w"] == 1) or (want_h and x["h"] == 1 and x["w"] > 1)]
                smode = mode.replace("vertical_bars_", "").replace("horizontal_bars_", "")
            else: smode = mode
            rows = _mutable(g)
            for idx, info in enumerate(_sort_infos(comps, smode)):
                if idx in mapping:
                    for r, c in info["cells"]: rows[r][c] = mapping[idx]
            return _freeze(rows)
        if _validate(train, fn):
            prefix = "gen_recolor_bars_by_order" if bars else "gen_recolor_components_by_rank"
            rules.append(_rule(f"{prefix}_{mode}", fn))
    return rules


def fit_recolor_components_by_rank(train: Train) -> list[Rule]:
    return _fit_rank_recolor(train, ("left_to_right","right_to_left","top_to_bottom","bottom_to_top","area_ascending","area_descending","height_ascending","height_descending","width_ascending","width_descending"), False)


def fit_recolor_bars_by_order(train: Train) -> list[Rule]:
    return _fit_rank_recolor(train, ("vertical_bars_left_to_right","vertical_bars_top_to_bottom","vertical_bars_height_ascending","vertical_bars_height_descending","horizontal_bars_top_to_bottom","horizontal_bars_left_to_right","horizontal_bars_width_ascending","horizontal_bars_width_descending"), True)


# Experimental family: split-panel boolean operations

def _two_panels_axis(g: Grid, axis: str) -> tuple[Grid, Grid] | None:
    h, w = shape(g); seps = _sep_cols(g) if axis == "col" else _sep_rows(g)
    if len(seps) != 1: return None
    spans = _spans(w if axis == "col" else h, seps)
    if len(spans) != 2: return None
    a0,b0 = spans[0]; a1,b1 = spans[1]
    p0 = tuple(tuple(row[c] for c in range(a0,b0)) for row in g) if axis == "col" else tuple(tuple(g[r][c] for c in range(w)) for r in range(a0,b0))
    p1 = tuple(tuple(row[c] for c in range(a1,b1)) for row in g) if axis == "col" else tuple(tuple(g[r][c] for c in range(w)) for r in range(a1,b1))
    return (p0, p1) if shape(p0) == shape(p1) else None


def _panel_bool(g: Grid, axis: str, op: str, color: int | None) -> Grid | None:
    panels = _two_panels_axis(g, axis)
    if panels is None: return None
    a,b = panels; h,w = shape(a); rows=[]
    for r in range(h):
        row=[]
        for c in range(w):
            x,y=a[r][c],b[r][c]
            if op == "and_prefer_first": v = x if x and y else 0
            elif op == "and_prefer_second": v = y if x and y else 0
            elif op == "xor": v = x if x and not y else (y if y and not x else 0)
            elif op == "left_minus_right": v = x if x and not y else 0
            elif op == "right_minus_left": v = y if y and not x else 0
            elif op == "overlap_only_color": v = (color or 0) if x and y else 0
            else: v = (color or 0) if x and y else (x if x and not y else (y if y and not x else 0))
            row.append(v)
        rows.append(row)
    return _freeze(rows)


def fit_split_panel_boolean_operation(train: Train) -> list[Rule]:
    rules=[]; ops=("and_prefer_first","and_prefer_second","xor","left_minus_right","right_minus_left")
    for axis in ("col","row"):
        for op in ops:
            fn=lambda g,axis=axis,op=op:_panel_bool(g,axis,op,None)
            if _validate(train,fn): rules.append(_rule(f"gen_panel_bool_{axis}_{op}",fn))
        for op,prefix in (("overlap_only_color","overlap_color"),("union_learned_overlap_color","union_overlap_color")):
            for color in range(1,10):
                fn=lambda g,axis=axis,op=op,color=color:_panel_bool(g,axis,op,color)
                if _validate(train,fn): rules.append(_rule(f"gen_panel_bool_{axis}_{prefix}_{color}",fn))
    return rules


# Experimental family: bounded rays
_DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}

def _seed_points(g: Grid, seed_mode: str) -> list[tuple[int,int]]:
    infos=[_comp_info(g,c) for c in _components4(g)]; pts=[]
    for x in infos:
        cells=x["cells"]
        if seed_mode=="singletons" and len(cells)==1: pts += cells
        elif seed_mode=="horizontal_endpoints" and x["h"]==1 and x["w"]>1 and len(cells)==x["w"]:
            r=x["r0"]; pts += [(r,x["c0"]),(r,x["c1"])]
        elif seed_mode=="vertical_endpoints" and x["w"]==1 and x["h"]>1 and len(cells)==x["h"]:
            c=x["c0"]; pts += [(x["r0"],c),(x["r1"],c)]
    return pts


def _extend_rays(g: Grid, seed_mode: str, direction: str, stop: str) -> Grid | None:
    h,w=shape(g); dr,dc=_DIRS[direction]; rows=_mutable(g)
    for r,c in _seed_points(g, seed_mode):
        col=g[r][c]; nr,nc=r+dr,c+dc
        while 0 <= nr < h and 0 <= nc < w:
            v=g[nr][nc]
            if stop=="until_blocked" and v!=0: break
            if stop=="until_same_color" and v==col: break
            if stop=="until_different_nonzero" and v!=0 and v!=col: break
            if v==0: rows[nr][nc]=col
            nr+=dr; nc+=dc
    return _freeze(rows)


def fit_extend_rays_to_marker_or_boundary(train: Train) -> list[Rule]:
    rules=[]
    for seed in ("singletons","horizontal_endpoints","vertical_endpoints"):
        for direction in _DIRS:
            for stop in ("to_boundary","until_blocked","until_same_color","until_different_nonzero"):
                fn=lambda g,seed=seed,direction=direction,stop=stop:_extend_rays(g,seed,direction,stop)
                if _validate(train,fn): rules.append(_rule(f"gen_extend_ray_{seed}_{direction}_{stop}",fn))
    return rules


# Experimental family: periodic tile with phase crop

def _tile_source(g: Grid, source: str) -> Grid | None:
    if source=="full_input": return g if shape(g)[0] and shape(g)[1] else None
    if source=="nonzero_bbox_crop": return _crop_positions(g, nonzero_positions(g))
    bgmode={"background_bbox_crop_zero":"zero","background_bbox_crop_most_common":"most_common","background_bbox_crop_corner":"corner"}.get(source)
    return _crop_non_bg_bbox(g, bgmode) if bgmode else None


def _periodic(tile: Grid, oh: int, ow: int, pr: int, pc: int) -> Grid | None:
    th,tw=shape(tile)
    if th<=0 or tw<=0: return None
    return tuple(tuple(tile[(r+pr)%th][(c+pc)%tw] for c in range(ow)) for r in range(oh))


def fit_periodic_tile_with_phase_crop(train: Train) -> list[Rule]:
    rules: list[Rule] = []
    sources = ("full_input", "nonzero_bbox_crop", "background_bbox_crop_zero", "background_bbox_crop_most_common", "background_bbox_crop_corner")
    for source in sources:
        tiles: list[Grid] = []
        ok = True
        for inp, _ in train:
            tile = _tile_source(inp, source)
            if not is_valid_grid(tile):
                ok = False; break
            tiles.append(tile)  # type: ignore[arg-type]
        if not ok or not tiles:
            continue
        th0, tw0 = shape(tiles[0])
        first_out = train[0][1]
        ohf, owf = shape(first_out)
        phase_candidates = []
        for pr in range(th0):
            for pc in range(tw0):
                if ohf and owf and tiles[0][pr % th0][pc % tw0] != first_out[0][0]:
                    continue
                if ohf > 1 and tiles[0][(1 + pr) % th0][pc % tw0] != first_out[1][0]:
                    continue
                if owf > 1 and tiles[0][pr % th0][(1 + pc) % tw0] != first_out[0][1]:
                    continue
                phase_candidates.append((pr, pc))
        for pr, pc in phase_candidates:
                matched = True
                for tile, (_, out) in zip(tiles, train):
                    th, tw = shape(tile)
                    if pr >= th or pc >= tw:
                        matched = False; break
                    oh, ow = shape(out)
                    pred = _periodic(tile, oh, ow, pr, pc)
                    if pred != out:
                        matched = False; break
                if not matched:
                    continue
                def fn(g: Grid, source=source, pr=pr, pc=pc) -> Grid | None:
                    tile = _tile_source(g, source)
                    if not is_valid_grid(tile):
                        return None
                    # Preserve the observed output-size delta for test grids.
                    ih0, iw0 = shape(train[0][0]); oh0, ow0 = shape(train[0][1]); h, w = shape(g)
                    return _periodic(tile, oh0 + (h - ih0), ow0 + (w - iw0), pr, pc)  # type: ignore[arg-type]
                if _validate(train, fn):
                    slug = {"full_input": "full_input", "nonzero_bbox_crop": "nonzero_bbox", "background_bbox_crop_zero": "bg_zero", "background_bbox_crop_most_common": "bg_most_common", "background_bbox_crop_corner": "bg_corner"}[source]
                    rules.append(_rule(f"gen_periodic_tile_{slug}_phase_{pr}_{pc}", fn))
    return rules

# Experimental family: periodic add mask completion

def _same_shape_add_only(train: Train) -> bool:
    return all(shape(i)==shape(o) and all((i[r][c]==o[r][c] or (i[r][c]==0 and o[r][c]!=0)) for r in range(shape(i)[0]) for c in range(shape(i)[1])) for i,o in train)

def _complete_lattice(g: Grid, sr: int, sc: int) -> Grid | None:
    h,w=shape(g); rows=_mutable(g)
    for color in nonzero_colors(g):
        pts=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==color]
        if len(pts)<2: continue
        rs={r%sr for r,_ in pts} if sr else set(); cs={c%sc for _,c in pts} if sc else set()
        for r in range(h):
            for c in range(w):
                if rows[r][c]==0 and (not sr or r%sr in rs) and (not sc or c%sc in cs): rows[r][c]=color
    return _freeze(rows)

def _full_grid_added(g: Grid, pattern: dict[tuple[int,int], int], ph: int, pw: int) -> Grid | None:
    h,w=shape(g); rows=_mutable(g)
    for r in range(h):
        for c in range(w):
            col=pattern.get((r%ph,c%pw))
            if col and g[r][c]==0: rows[r][c]=col
    return _freeze(rows)

def _copy_component_offsets(g: Grid, dr: int, dc: int) -> Grid | None:
    h,w=shape(g); rows=_mutable(g)
    for comp in _components4(g):
        for r,c in comp:
            nr,nc=r+dr,c+dc
            if not (0<=nr<h and 0<=nc<w): return None
            if g[nr][nc]==0: rows[nr][nc]=g[r][c]
    return _freeze(rows)

def fit_periodic_add_mask_completion(train: Train) -> list[Rule]:
    if not train or not _same_shape_add_only(train): return []
    rules=[]
    for sr in range(1,11):
        for sc in range(1,11):
            fn=lambda g,sr=sr,sc=sc:_complete_lattice(g,sr,sc)
            if _validate(train,fn): rules.append(_rule(f"gen_periodic_add_complete_lattice_{sr}_{sc}",fn))
    for ph in range(1,11):
        for pw in range(1,11):
            pat={}; ok=True
            for inp,out in train:
                h,w=shape(inp)
                for r in range(h):
                    for c in range(w):
                        if inp[r][c]==0 and out[r][c]!=0:
                            k=(r%ph,c%pw)
                            if k in pat and pat[k]!=out[r][c]: ok=False; break
                            pat[k]=out[r][c]
                    if not ok: break
                if not ok: break
            if not pat or not ok: continue
            fn=lambda g,pat=pat,ph=ph,pw=pw:_full_grid_added(g,pat,ph,pw)
            if _validate(train,fn): rules.append(_rule(f"gen_periodic_add_full_grid_period_{ph}_{pw}",fn))
    h,w=shape(train[0][0])
    for dr in range(-min(10,h-1),min(10,h-1)+1):
        for dc in range(-min(10,w-1),min(10,w-1)+1):
            if dr==0 and dc==0: continue
            fn=lambda g,dr=dr,dc=dc:_copy_component_offsets(g,dr,dc)
            if _validate(train,fn): rules.append(_rule(f"gen_periodic_add_copy_component_offsets_{dr}_{dc}",fn))
    return rules

# Experimental family: separator component bbox extract

def _strip_seps(g: Grid) -> Grid | None: return _strip(g, True, True)

def _crop_largest_component(g: Grid) -> Grid | None:
    comps=_components4(g)
    return _crop_positions(g, max(comps,key=len)) if comps else None

def fit_separator_component_bbox_extract(train: Train) -> list[Rule]:
    rules=[]
    specs=[("gen_sep_crop_nonzero", lambda s:_crop_positions(s, nonzero_positions(s))), ("gen_sep_crop_largest_component", _crop_largest_component), ("gen_sep_crop_components_bbox", lambda s:_crop_positions(s, [p for comp in _components4(s) for p in comp]))]
    colors=set(range(1,10))
    for name,op in specs:
        fn=lambda g,op=op: (None if _strip_seps(g) is None else op(_strip_seps(g)))
        if _validate(train,fn): rules.append(_rule(name,fn))
    for color in colors:
        fn=lambda g,color=color: (None if _strip_seps(g) is None else _crop_positions(_strip_seps(g), [(r,c) for r,row in enumerate(_strip_seps(g)) for c,v in enumerate(row) if v==color]))
        if _validate(train,fn): rules.append(_rule(f"gen_sep_crop_color_{color}",fn))
    return rules

# Experimental family: recolor preserved masks by role

def _role(info: dict, h:int, w:int, mode:str):
    if mode=="touches_border": return info["r0"]==0 or info["c0"]==0 or info["r1"]==h-1 or info["c1"]==w-1
    if mode=="inside_not_border": return not _role(info,h,w,"touches_border")
    if mode=="is_line": return info["h"]==1 or info["w"]==1
    if mode=="is_square_bbox": return info["h"]==info["w"]
    if mode=="bbox_height": return info["h"]
    if mode=="bbox_width": return info["w"]
    if mode=="bbox_area": return info["h"]*info["w"]
    if mode=="component_area": return info["area"]
    cr,cc=(h-1)/2,(w-1)/2; cen=((info["r0"]+info["r1"])/2,(info["c0"]+info["c1"])/2); dist=(cen[0]-cr)**2+(cen[1]-cc)**2
    if mode in ("nearest_to_center","farthest_from_center"): return round(dist,6)
    if mode=="relative_quadrant": return (cen[0]>=cr, cen[1]>=cc)
    return None

def fit_recolor_preserved_masks_by_role(train: Train) -> list[Rule]:
    modes=("touches_border","inside_not_border","is_line","is_square_bbox","bbox_height","bbox_width","bbox_area","component_area","nearest_to_center","farthest_from_center","relative_quadrant")
    rules=[]
    for mode in modes:
        mapping={}; changed=False; ok=True
        for inp,out in train:
            if shape(inp)!=shape(out): ok=False; break
            h,w=shape(inp)
            for r in range(h):
                for c in range(w):
                    if (inp[r][c]==0)!=(out[r][c]==0): ok=False; break
                if not ok: break
            if not ok: break
            infos=[_comp_info(inp,c) for c in _components4(inp)]
            if mode=="nearest_to_center": vals=[_role(x,h,w,mode) for x in infos]; mark=min(vals) if vals else None
            elif mode=="farthest_from_center": vals=[_role(x,h,w,mode) for x in infos]; mark=max(vals) if vals else None
            else: mark=None
            for info in infos:
                rv=(mark==_role(info,h,w,mode)) if mode in ("nearest_to_center","farthest_from_center") else _role(info,h,w,mode)
                cols={out[r][c] for r,c in info["cells"]}
                if len(cols)!=1 or 0 in cols: ok=False; break
                col=next(iter(cols))
                if rv in mapping and mapping[rv]!=col: ok=False; break
                mapping[rv]=col; changed |= any(inp[r][c]!=col for r,c in info["cells"])
            if not ok: break
        if not ok or not changed: continue
        def fn(g:Grid,mode=mode,mapping=mapping):
            h,w=shape(g); rows=_mutable(g); infos=[_comp_info(g,c) for c in _components4(g)]
            mark=None
            if mode in ("nearest_to_center","farthest_from_center") and infos:
                vals=[_role(x,h,w,mode) for x in infos]; mark=min(vals) if mode.startswith("nearest") else max(vals)
            for info in infos:
                rv=(mark==_role(info,h,w,mode)) if mode in ("nearest_to_center","farthest_from_center") else _role(info,h,w,mode)
                if rv in mapping:
                    for r,c in info["cells"]: rows[r][c]=mapping[rv]
            return _freeze(rows)
        if _validate(train,fn): rules.append(_rule(f"gen_recolor_components_by_role_{mode}",fn))
    return rules

# Experimental family: component move or copy by offset

def _selected_components(g: Grid, selector: str) -> list[list[tuple[int,int]]]:
    comps=_components4(g); infos=[_comp_info(g,c) for c in comps]
    if selector.startswith("color_"):
        col=int(selector.split("_",1)[1]); return [x["cells"] for x in infos if g[x["cells"][0][0]][x["cells"][0][1]]==col]
    if selector.startswith("size_"):
        n=int(selector.split("_",1)[1]); return [x["cells"] for x in infos if x["area"]==n]
    if selector=="largest": return [max(comps,key=len)] if comps else []
    if selector=="smallest": return [min(comps,key=len)] if comps else []
    if selector=="singleton": return [c for c in comps if len(c)==1]
    if selector.startswith("rank_left_to_right_"):
        k=int(selector.rsplit("_",1)[1]); s=_sort_infos(infos,"left_to_right"); return [s[k]["cells"]] if k<len(s) else []
    if selector.startswith("rank_top_to_bottom_"):
        k=int(selector.rsplit("_",1)[1]); s=_sort_infos(infos,"top_to_bottom"); return [s[k]["cells"]] if k<len(s) else []
    return []

def _move_copy_components(g: Grid, selector: str, dr:int, dc:int, copy:bool) -> Grid | None:
    h,w=shape(g); comps=_selected_components(g,selector)
    if not comps: return None
    rows=_mutable(g); src={p for comp in comps for p in comp}; writes=[]
    for comp in comps:
        for r,c in comp:
            nr,nc=r+dr,c+dc
            if not (0<=nr<h and 0<=nc<w): return None
            if g[nr][nc]!=0 and (nr,nc) not in src and g[nr][nc]!=g[r][c]: return None
            writes.append((nr,nc,g[r][c]))
    if not copy:
        for r,c in src: rows[r][c]=0
    for r,c,v in writes: rows[r][c]=v
    return _freeze(rows)

def fit_component_move_or_copy_by_offset(train: Train) -> list[Rule]:
    if not train or any(shape(i)!=shape(o) for i,o in train): return []
    base_selectors={"largest","smallest","singleton"}; offsets:set[tuple[int,int]]=set()
    for inp,out in train:
        infos=[_comp_info(inp,c) for c in _components4(inp)]
        out_infos=[_comp_info(out,c) for c in _components4(out)]
        for a in infos:
            ca=inp[a["cells"][0][0]][a["cells"][0][1]]
            for b in out_infos:
                cb=out[b["cells"][0][0]][b["cells"][0][1]]
                if ca==cb and a["area"]==b["area"]:
                    dr,dc=b["r0"]-a["r0"],b["c0"]-a["c0"]
                    if (dr or dc) and abs(dr)<=10 and abs(dc)<=10: offsets.add((dr,dc))
    rules=[]
    for mode,copy in (("move_one_component_by_offset",False),("copy_one_component_by_offset",True),("move_all_components_by_offset",False),("copy_all_components_by_offset",True)):
        for sel in sorted(base_selectors):
            for dr,dc in sorted(offsets):
                fn=lambda g,sel=sel,dr=dr,dc=dc,copy=copy:_move_copy_components(g,sel,dr,dc,copy)
                if _validate(train,fn): rules.append(_rule(f"gen_{mode}_{sel}_offset_{dr}_{dc}",fn))
    return rules


# Experimental family: bounded hole and region fill variants

def _zero_regions(g: Grid) -> list[list[tuple[int,int]]]:
    h,w=shape(g); seen=set(); regs=[]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=0 or (r,c) in seen: continue
            st=[(r,c)]; seen.add((r,c)); comp=[]
            while st:
                rr,cc=st.pop(); comp.append((rr,cc))
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc=rr+dr,cc+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]==0 and (nr,nc) not in seen: seen.add((nr,nc)); st.append((nr,nc))
            regs.append(comp)
    return regs

def _enclosed_region_color(g: Grid, reg:list[tuple[int,int]]) -> int | None:
    h,w=shape(g)
    if any(r in (0,h-1) or c in (0,w-1) for r,c in reg): return None
    neigh=set()
    s=set(reg)
    for r,c in reg:
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            p=(r+dr,c+dc)
            if p not in s and g[p[0]][p[1]]!=0: neigh.add(g[p[0]][p[1]])
    return next(iter(neigh)) if len(neigh)==1 else None

def _fill_holes(g: Grid, mode:str, color:int|None=None, n:int|None=None) -> Grid | None:
    rows=_mutable(g)
    if mode=="bbox":
        for comp in _components4(g):
            b=bbox_for_positions(comp)
            if b:
                r0,c0,r1,c1=b; col=g[comp[0][0]][comp[0][1]]
                for r in range(r0,r1+1):
                    for c in range(c0,c1+1):
                        if g[r][c]==0: rows[r][c]=col
    else:
        for reg in _zero_regions(g):
            col = _enclosed_region_color(g,reg) if mode=="enclosing" else color
            if mode=="size" and (n is None or len(reg)>n): continue
            if mode=="size": col=_enclosed_region_color(g,reg)
            if col:
                for r,c in reg: rows[r][c]=col
    return _freeze(rows)

def fit_bounded_hole_and_region_fill_variants(train: Train) -> list[Rule]:
    if not train or not _same_shape_add_only(train): return []
    rules=[]
    for name,fn in [("gen_fill_holes_enclosing_color",lambda g:_fill_holes(g,"enclosing")), ("gen_fill_inside_component_bbox",lambda g:_fill_holes(g,"bbox"))]:
        if _validate(train,fn): rules.append(_rule(name,fn))
    for color in range(1,10):
        fn=lambda g,color=color:_fill_holes(g,"learned",color=color)
        if _validate(train,fn): rules.append(_rule(f"gen_fill_holes_learned_color_{color}",fn))
    for n in range(1,21):
        fn=lambda g,n=n:_fill_holes(g,"size",n=n)
        if _validate(train,fn): rules.append(_rule(f"gen_fill_holes_size_le_{n}",fn))
    return rules


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
    "d4_conjugated_existing": fit_d4_conjugated_existing,
    "structure_then_color_map": fit_structure_then_color_map,
    "crop_to_background_bbox": fit_crop_to_background_bbox,
    "periodic_tile_with_phase_crop": fit_periodic_tile_with_phase_crop,
    "extend_rays_to_marker_or_boundary": fit_extend_rays_to_marker_or_boundary,
    "split_panel_boolean_operation": fit_split_panel_boolean_operation,
    "recolor_bars_by_order": fit_recolor_bars_by_order,
    "recolor_components_by_rank": fit_recolor_components_by_rank,
    "periodic_add_mask_completion": fit_periodic_add_mask_completion,
    "separator_component_bbox_extract": fit_separator_component_bbox_extract,
    "recolor_preserved_masks_by_role": fit_recolor_preserved_masks_by_role,
    "component_move_or_copy_by_offset": fit_component_move_or_copy_by_offset,
    "bounded_hole_and_region_fill_variants": fit_bounded_hole_and_region_fill_variants,
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
    "d4": "d4_conjugated_existing",
    "structure_color": "structure_then_color_map",
    "crop_bg": "crop_to_background_bbox",
    "rank_components": "recolor_components_by_rank",
    "recolor_components_by_rank": "recolor_components_by_rank",
    "bars": "recolor_bars_by_order",
    "recolor_bars_by_order": "recolor_bars_by_order",
    "panel_bool": "split_panel_boolean_operation",
    "split_panel_boolean_operation": "split_panel_boolean_operation",
    "rays": "extend_rays_to_marker_or_boundary",
    "extend_rays": "extend_rays_to_marker_or_boundary",
    "extend_rays_to_marker_or_boundary": "extend_rays_to_marker_or_boundary",
    "periodic": "periodic_tile_with_phase_crop",
    "periodic_tile_with_phase_crop": "periodic_tile_with_phase_crop",
    "periodic_add": "periodic_add_mask_completion",
    "sep_bbox": "separator_component_bbox_extract",
    "role_recolor": "recolor_preserved_masks_by_role",
    "component_move": "component_move_or_copy_by_offset",
    "component_copy": "component_move_or_copy_by_offset",
    "hole_fill": "bounded_hole_and_region_fill_variants",
}
