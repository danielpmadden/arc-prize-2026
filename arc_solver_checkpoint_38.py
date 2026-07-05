#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

Grid = tuple[tuple[int, ...], ...]
PredictFn = Callable[[Grid], Optional[Grid]]


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    predict: PredictFn


# ----------------------------
# Grid utilities
# ----------------------------

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
# Geometric transforms
# ----------------------------

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


# ----------------------------
# Candidate rule fitters
# ----------------------------

def fit_geom_and_color(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []
    outs = [out for _, out in train]

    for geom_name, geom in GEOMS:
        srcs = [geom(inp) for inp, _ in train]
        mapping = infer_color_map(srcs, outs)
        if mapping is None:
            continue

        def predict(g: Grid, geom=geom, mapping=mapping) -> Grid:
            return apply_color_map(geom(g), mapping)

        priority = 10 if geom_name == "identity" else 20
        rules.append(Rule(f"{geom_name}+color_map", priority, predict))

    return rules


def fit_crop_geom_color(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []
    outs = [out for _, out in train]

    bg_selectors: list[tuple[str, Callable[[Grid], int]]] = [
        ("zero_bg", lambda g: 0),
        ("mode_bg", most_common_color),
    ]

    for bg_name, bg_fn in bg_selectors:
        for geom_name, geom in GEOMS:
            srcs: list[Grid] = []
            ok = True

            for inp, out in train:
                bg = bg_fn(inp)
                src = geom(crop_non_bg(inp, bg))
                if shape(src) != shape(out):
                    ok = False
                    break
                srcs.append(src)

            if not ok:
                continue

            mapping = infer_color_map(srcs, outs)
            if mapping is None:
                continue

            def predict(g: Grid, bg_fn=bg_fn, geom=geom, mapping=mapping) -> Grid:
                bg = bg_fn(g)
                return apply_color_map(geom(crop_non_bg(g, bg)), mapping)

            rules.append(Rule(f"crop_{bg_name}_{geom_name}+color_map", 30, predict))

    return rules


def fit_upscale(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    factors: Optional[tuple[int, int]] = None
    coarse_outs: list[Grid] = []
    srcs: list[Grid] = []

    for inp, out in train:
        hi, wi = shape(inp)
        ho, wo = shape(out)

        if hi == 0 or wi == 0 or ho % hi != 0 or wo % wi != 0:
            return []

        fr, fc = ho // hi, wo // wi
        if fr < 1 or fc < 1:
            return []

        if factors is None:
            factors = (fr, fc)
        elif factors != (fr, fc):
            return []

        coarse_rows = []
        for r in range(hi):
            coarse_row = []
            for c in range(wi):
                block_values = {
                    out[rr][cc]
                    for rr in range(r * fr, (r + 1) * fr)
                    for cc in range(c * fc, (c + 1) * fc)
                }
                if len(block_values) != 1:
                    return []
                coarse_row.append(next(iter(block_values)))
            coarse_rows.append(tuple(coarse_row))

        srcs.append(inp)
        coarse_outs.append(tuple(coarse_rows))

    mapping = infer_color_map(srcs, coarse_outs)
    if mapping is None or factors is None:
        return []

    fr, fc = factors

    def predict(g: Grid, mapping=mapping, fr=fr, fc=fc) -> Grid:
        mapped = apply_color_map(g, mapping)
        rows = []
        for row in mapped:
            expanded = []
            for v in row:
                expanded.extend([v] * fc)
            for _ in range(fr):
                rows.append(tuple(expanded))
        return tuple(rows)

    return [Rule(f"nearest_upscale_{fr}x{fc}", 25, predict)]


def fit_repeated_tile(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []

    for geom_name, geom in GEOMS:
        factors: Optional[tuple[int, int]] = None
        ok = True

        for inp, out in train:
            src = geom(inp)
            hs, ws = shape(src)
            ho, wo = shape(out)

            if hs == 0 or ws == 0 or ho % hs != 0 or wo % ws != 0:
                ok = False
                break

            fr, fc = ho // hs, wo // ws
            if factors is None:
                factors = (fr, fc)
            elif factors != (fr, fc):
                ok = False
                break

            expected = tuple(
                tuple(src[r % hs][c % ws] for c in range(wo))
                for r in range(ho)
            )

            if expected != out:
                ok = False
                break

        if not ok or factors is None:
            continue

        fr, fc = factors

        def predict(g: Grid, geom=geom, fr=fr, fc=fc) -> Grid:
            src = geom(g)
            hs, ws = shape(src)
            return tuple(
                tuple(src[r % hs][c % ws] for c in range(ws * fc))
                for r in range(hs * fr)
            )

        rules.append(Rule(f"repeat_tile_{geom_name}_{fr}x{fc}", 35, predict))

    return rules


def kron_self_mask(g: Grid, bg: int = 0) -> Grid:
    h, w = shape(g)
    bg_block = constant_grid(h, w, bg)

    rows = []
    for br in range(h):
        for ir in range(h):
            row = []
            for bc in range(w):
                block = g if g[br][bc] != bg else bg_block
                row.extend(block[ir])
            rows.append(tuple(row))

    return tuple(rows)


def fit_kron_self_mask(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []

    bg_selectors: list[tuple[str, Callable[[Grid], int]]] = [
        ("zero", lambda g: 0),
        ("mode", most_common_color),
    ]

    for bg_name, bg_fn in bg_selectors:
        ok = True
        for inp, out in train:
            bg = bg_fn(inp)
            if kron_self_mask(inp, bg) != out:
                ok = False
                break

        if not ok:
            continue

        def predict(g: Grid, bg_fn=bg_fn) -> Grid:
            return kron_self_mask(g, bg_fn(g))

        rules.append(Rule(f"kron_self_mask_{bg_name}", 15, predict))

    return rules


def weave_2x2_to_6x6(g: Grid) -> Optional[Grid]:
    if shape(g) != (2, 2):
        return None

    rows = []
    for group in range(3):
        reverse = group == 1
        for r in range(2):
            base = list(g[r])
            if reverse:
                base = list(reversed(base))
            rows.append(tuple((base * 3)[:6]))

    return tuple(rows)


def fit_2x2_weave(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    if all(shape(inp) == (2, 2) and weave_2x2_to_6x6(inp) == out for inp, out in train):
        return [Rule("2x2_weave_to_6x6", 12, weave_2x2_to_6x6)]
    return []


def fit_recolor_by_key_shape(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    """
    Learns tasks where:
    - input contains a large object and a small key object;
    - output removes the key object;
    - output recolors the large object;
    - the key object's normalized shape determines the output color.
    """
    key_shape_to_color: dict[Grid, int] = {}

    for inp, out in train:
        in_cols = nonzero_colors(inp)
        out_cols = nonzero_colors(out)

        if len(in_cols) != 2 or len(out_cols) != 1:
            return []

        target_color = out_cols[0]
        out_pos = nonzero_positions(out)

        source_color = None
        for c in in_cols:
            if positions_of(inp, c) == out_pos:
                source_color = c
                break

        if source_color is None:
            # Fallback: source is usually the larger object.
            source_color = max(in_cols, key=lambda c: len(positions_of(inp, c)))

        key_color = next(c for c in in_cols if c != source_color)
        source_pos = positions_of(inp, source_color)

        expected = constant_grid(*shape(inp), 0)
        expected_rows = [list(row) for row in expected]
        for r, c in source_pos:
            expected_rows[r][c] = target_color
        expected = as_grid(expected_rows)

        if expected != out:
            return []

        key_shape = normalize_mask(positions_of(inp, key_color))
        if key_shape is None:
            return []

        if key_shape in key_shape_to_color and key_shape_to_color[key_shape] != target_color:
            return []

        key_shape_to_color[key_shape] = target_color

    if not key_shape_to_color:
        return []

    def predict(g: Grid, learned=key_shape_to_color) -> Optional[Grid]:
        cols = nonzero_colors(g)
        if len(cols) != 2:
            return None

        matched_key_color = None
        target_color = None

        for c in cols:
            key_shape = normalize_mask(positions_of(g, c))
            if key_shape in learned:
                matched_key_color = c
                target_color = learned[key_shape]
                break

        if matched_key_color is None or target_color is None:
            return None

        source_color = next(c for c in cols if c != matched_key_color)
        source_pos = positions_of(g, source_color)

        h, w = shape(g)
        rows = [[0 for _ in range(w)] for _ in range(h)]
        for r, c in source_pos:
            rows[r][c] = target_color

        return as_grid(rows)

    return [Rule("recolor_large_object_by_key_shape", 18, predict)]


def fit_constant_output(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    outs = [out for _, out in train]
    out_shapes = {shape(out) for out in outs}

    if len(out_shapes) != 1:
        return []

    h, w = next(iter(out_shapes))
    color_counts = Counter(v for out in outs for row in out for v in row)
    value = color_counts.most_common(1)[0][0]

    def predict(_: Grid, h=h, w=w, value=value) -> Grid:
        return constant_grid(h, w, value)

    return [Rule("constant_output_fallback", 90, predict)]

def fit_split_intersection_bar(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    learned_out_color: Optional[int] = None

    for inp, out in train:
        h, w = shape(inp)

        if w % 2 == 0:
            return []

        mid = w // 2
        left_w = mid
        right_w = w - mid - 1

        if left_w != right_w:
            return []

        sep_color = inp[0][mid]

        if any(inp[r][mid] != sep_color for r in range(h)):
            return []

        left = tuple(tuple(inp[r][c] for c in range(left_w)) for r in range(h))
        right = tuple(tuple(inp[r][mid + 1 + c] for c in range(right_w)) for r in range(h))

        out_colors = nonzero_colors(out)
        if len(out_colors) != 1:
            return []

        out_color = out_colors[0]

        if learned_out_color is None:
            learned_out_color = out_color
        elif learned_out_color != out_color:
            return []

        expected = tuple(
            tuple(out_color if left[r][c] != 0 and right[r][c] != 0 else 0 for c in range(left_w))
            for r in range(h)
        )

        if expected != out:
            return []

    if learned_out_color is None:
        return []

    def predict(g: Grid, out_color=learned_out_color) -> Optional[Grid]:
        h, w = shape(g)

        if w % 2 == 0:
            return None

        mid = w // 2
        left_w = mid
        right_w = w - mid - 1

        if left_w != right_w:
            return None

        sep_color = g[0][mid]

        if any(g[r][mid] != sep_color for r in range(h)):
            return None

        left = tuple(tuple(g[r][c] for c in range(left_w)) for r in range(h))
        right = tuple(tuple(g[r][mid + 1 + c] for c in range(right_w)) for r in range(h))

        return tuple(
            tuple(out_color if left[r][c] != 0 and right[r][c] != 0 else 0 for c in range(left_w))
            for r in range(h)
        )

    return [Rule("split_intersection_bar", 16, predict)]

def fit_fill_enclosed_regions(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    learned_fill_color: Optional[int] = None

    def fill_enclosed(g: Grid, fill_color: int) -> Grid:
        h, w = shape(g)

        # Find the wall color: the most common nonzero color.
        nonzero = nonzero_colors(g)
        if len(nonzero) != 1:
            return g

        wall_color = nonzero[0]

        reachable = set()
        stack = []

        # Border zeros are outside.
        for r in range(h):
            for c in (0, w - 1):
                if g[r][c] == 0:
                    stack.append((r, c))

        for c in range(w):
            for r in (0, h - 1):
                if g[r][c] == 0:
                    stack.append((r, c))

        while stack:
            r, c = stack.pop()

            if (r, c) in reachable:
                continue

            if not (0 <= r < h and 0 <= c < w):
                continue

            if g[r][c] != 0:
                continue

            reachable.add((r, c))

            stack.append((r - 1, c))
            stack.append((r + 1, c))
            stack.append((r, c - 1))
            stack.append((r, c + 1))

        rows = [list(row) for row in g]

        for r in range(h):
            for c in range(w):
                if g[r][c] == 0 and (r, c) not in reachable:
                    rows[r][c] = fill_color

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        diff_colors = set()

        for r in range(shape(inp)[0]):
            for c in range(shape(inp)[1]):
                a = inp[r][c]
                b = out[r][c]

                if a == b:
                    continue

                if a != 0:
                    return []

                diff_colors.add(b)

        if len(diff_colors) != 1:
            return []

        fill_color = next(iter(diff_colors))

        if learned_fill_color is None:
            learned_fill_color = fill_color
        elif learned_fill_color != fill_color:
            return []

        if fill_enclosed(inp, fill_color) != out:
            return []

    if learned_fill_color is None:
        return []

    def predict(g: Grid, fill_color=learned_fill_color) -> Grid:
        return fill_enclosed(g, fill_color)

    return [Rule("fill_enclosed_regions", 14, predict)]

def fit_fill_rectangles_by_size(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    size_to_fill: dict[tuple[int, int], int] = {}

    def find_rectangles(g: Grid) -> list[tuple[int, int, int, int]]:
        h, w = shape(g)
        nonzero = nonzero_colors(g)

        if len(nonzero) != 1:
            return []

        wall = nonzero[0]
        rects = []

        for r0 in range(h):
            for c0 in range(w):
                if g[r0][c0] != wall:
                    continue

                for r1 in range(r0 + 2, h):
                    for c1 in range(c0 + 2, w):
                        # Corners must be wall.
                        if (
                            g[r0][c1] != wall
                            or g[r1][c0] != wall
                            or g[r1][c1] != wall
                        ):
                            continue

                        # Top/bottom edges.
                        ok = True
                        for c in range(c0, c1 + 1):
                            if g[r0][c] != wall or g[r1][c] != wall:
                                ok = False
                                break

                        if not ok:
                            continue

                        # Left/right edges.
                        for r in range(r0, r1 + 1):
                            if g[r][c0] != wall or g[r][c1] != wall:
                                ok = False
                                break

                        if ok:
                            rects.append((r0, c0, r1, c1))

        # Keep only rectangles not contained inside bigger rectangles.
        final = []
        for rect in rects:
            r0, c0, r1, c1 = rect
            contained = False

            for other in rects:
                if rect == other:
                    continue

                a0, b0, a1, b1 = other

                if a0 <= r0 and b0 <= c0 and r1 <= a1 and c1 <= b1:
                    if (a1 - a0) * (b1 - b0) > (r1 - r0) * (c1 - c0):
                        contained = True
                        break

            if not contained:
                final.append(rect)

        return final

    def apply_rule(g: Grid, mapping: dict[tuple[int, int], int]) -> Grid:
        rows = [list(row) for row in g]

        for r0, c0, r1, c1 in find_rectangles(g):
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            key = (rect_h, rect_w)

            if key not in mapping:
                continue

            fill = mapping[key]

            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if rows[r][c] == 0:
                        rows[r][c] = fill

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        rects = find_rectangles(inp)
        if not rects:
            return []

        local_mapping: dict[tuple[int, int], int] = {}

        for r0, c0, r1, c1 in rects:
            fill_colors = set()

            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if inp[r][c] == 0 and out[r][c] != 0:
                        fill_colors.add(out[r][c])

            if len(fill_colors) != 1:
                return []

            fill = next(iter(fill_colors))
            key = (r1 - r0 + 1, c1 - c0 + 1)
            local_mapping[key] = fill

        for key, fill in local_mapping.items():
            if key in size_to_fill and size_to_fill[key] != fill:
                return []
            size_to_fill[key] = fill

        if apply_rule(inp, size_to_fill) != out:
            return []

    if not size_to_fill:
        return []

    def predict(g: Grid, mapping=size_to_fill) -> Grid:
        return apply_rule(g, mapping)

    return [Rule("fill_rectangles_by_size", 13, predict)]

def fit_extend_vertical_period(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    learned_in_color: Optional[int] = None
    learned_out_color: Optional[int] = None
    learned_out_height: Optional[int] = None

    def find_row_period(g: Grid) -> int:
        h, w = shape(g)

        for period in range(1, h + 1):
            ok = True

            for r in range(h):
                if g[r] != g[r % period]:
                    ok = False
                    break

            if ok:
                return period

        return h

    def recolor_and_extend(
        g: Grid,
        out_height: int,
        in_color: int,
        out_color: int,
    ) -> Grid:
        period = find_row_period(g)
        rows = []

        for r in range(out_height):
            row = tuple(
                out_color if value == in_color else value
                for value in g[r % period]
            )
            rows.append(row)

        return tuple(rows)

    for inp, out in train:
        hi, wi = shape(inp)
        ho, wo = shape(out)

        if wi != wo:
            return []

        in_nonzero = nonzero_colors(inp)
        out_nonzero = nonzero_colors(out)

        if len(in_nonzero) != 1 or len(out_nonzero) != 1:
            return []

        in_color = in_nonzero[0]
        out_color = out_nonzero[0]

        if learned_in_color is None:
            learned_in_color = in_color
        elif learned_in_color != in_color:
            return []

        if learned_out_color is None:
            learned_out_color = out_color
        elif learned_out_color != out_color:
            return []

        if learned_out_height is None:
            learned_out_height = ho
        elif learned_out_height != ho:
            return []

        expected = recolor_and_extend(inp, ho, in_color, out_color)

        if expected != out:
            return []

    if (
        learned_in_color is None
        or learned_out_color is None
        or learned_out_height is None
    ):
        return []

    def predict(
        g: Grid,
        out_height=learned_out_height,
        in_color=learned_in_color,
        out_color=learned_out_color,
    ) -> Grid:
        return recolor_and_extend(g, out_height, in_color, out_color)

    return [Rule("extend_vertical_period", 17, predict)]

def fit_staircase_components(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def components(g: Grid) -> list[tuple[int, int, Grid]]:
        h, w = shape(g)
        seen = set()
        comps = []

        for r in range(h):
            for c in range(w):
                if g[r][c] == 0 or (r, c) in seen:
                    continue

                color = g[r][c]
                stack = [(r, c)]
                cells = set()

                while stack:
                    rr, cc = stack.pop()

                    if (rr, cc) in seen:
                        continue

                    if not (0 <= rr < h and 0 <= cc < w):
                        continue

                    if g[rr][cc] != color:
                        continue

                    seen.add((rr, cc))
                    cells.add((rr, cc))

                    stack.append((rr - 1, cc))
                    stack.append((rr + 1, cc))
                    stack.append((rr, cc - 1))
                    stack.append((rr, cc + 1))

                box = bbox_for_positions(cells)
                if box is None:
                    continue

                r0, c0, r1, c1 = box
                comp = crop_rect(g, box)
                comps.append((r0, c0, comp))

        comps.sort(key=lambda item: item[1])
        return comps

    def apply_rule(g: Grid) -> Grid:
        h, w = shape(g)
        rows = [[0 for _ in range(w)] for _ in range(h)]

        r_cursor = 0
        c_cursor = 0

        for _, _, comp in components(g):
            ch, cw = shape(comp)

            for r in range(ch):
                for c in range(cw):
                    if comp[r][c] != 0:
                        rr = r_cursor + r
                        cc = c_cursor + c

                        if 0 <= rr < h and 0 <= cc < w:
                            rows[rr][cc] = comp[r][c]

            r_cursor += ch - 1
            c_cursor += cw - 1

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        if apply_rule(inp) != out:
            return []

    def predict(g: Grid) -> Grid:
        return apply_rule(g)

    return [Rule("staircase_components", 15, predict)]

def fit_expand_crosses_5x5(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def find_crosses(g: Grid) -> list[tuple[int, int, int, int]]:
        h, w = shape(g)
        crosses = []

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                center = g[r][c]

                if center == 0:
                    continue

                up = g[r - 1][c]
                down = g[r + 1][c]
                left = g[r][c - 1]
                right = g[r][c + 1]

                if up == 0:
                    continue

                if up == down == left == right and up != center:
                    # Make sure diagonals of the original 3x3 cross are empty.
                    if (
                        g[r - 1][c - 1] == 0
                        and g[r - 1][c + 1] == 0
                        and g[r + 1][c - 1] == 0
                        and g[r + 1][c + 1] == 0
                    ):
                        crosses.append((r, c, center, up))

        return crosses

    def apply_rule(g: Grid) -> Grid:
        h, w = shape(g)
        rows = [list(row) for row in g]

        for r, c, center_color, arm_color in find_crosses(g):
            # Orthogonal expansion with arm color.
            for dr, dc in [
                (-2, 0),
                (-1, 0),
                (1, 0),
                (2, 0),
                (0, -2),
                (0, -1),
                (0, 1),
                (0, 2),
            ]:
                rr = r + dr
                cc = c + dc

                if 0 <= rr < h and 0 <= cc < w:
                    rows[rr][cc] = arm_color

            # Diagonal expansion with center color.
            for dr, dc in [
                (-2, -2),
                (-1, -1),
                (-1, 1),
                (-2, 2),
                (1, -1),
                (2, -2),
                (1, 1),
                (2, 2),
            ]:
                rr = r + dr
                cc = c + dc

                if 0 <= rr < h and 0 <= cc < w:
                    rows[rr][cc] = center_color

            rows[r][c] = center_color

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        if not find_crosses(inp):
            return []

        if apply_rule(inp) != out:
            return []

    def predict(g: Grid) -> Grid:
        return apply_rule(g)

    return [Rule("expand_crosses_5x5", 14, predict)]

def fit_complete_symmetry(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def mirror_lr(g: Grid) -> Grid:
        h, w = shape(g)
        rows = [list(row) for row in g]

        for r in range(h):
            for c in range(w):
                v = g[r][c]
                if v == 0:
                    continue

                mc = w - 1 - c
                if rows[r][mc] == 0:
                    rows[r][mc] = v

        return as_grid(rows)

    def mirror_tb(g: Grid) -> Grid:
        h, w = shape(g)
        rows = [list(row) for row in g]

        for r in range(h):
            for c in range(w):
                v = g[r][c]
                if v == 0:
                    continue

                mr = h - 1 - r
                if rows[mr][c] == 0:
                    rows[mr][c] = v

        return as_grid(rows)

    candidates: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("complete_symmetry_left_right", mirror_lr),
        ("complete_symmetry_top_bottom", mirror_tb),
    ]

    rules: list[Rule] = []

    for name, fn in candidates:
        ok = True

        for inp, out in train:
            if shape(inp) != shape(out):
                ok = False
                break

            if fn(inp) != out:
                ok = False
                break

        if ok:
            def predict(g: Grid, fn=fn) -> Grid:
                return fn(g)

            rules.append(Rule(name, 14, predict))

    return rules

def fit_block_compress(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def compress(g: Grid, br: int, bc: int, mode: str) -> Optional[Grid]:
        h, w = shape(g)

        if h % br != 0 or w % bc != 0:
            return None

        rows = []

        for r0 in range(0, h, br):
            row = []

            for c0 in range(0, w, bc):
                vals = [
                    g[r][c]
                    for r in range(r0, r0 + br)
                    for c in range(c0, c0 + bc)
                ]

                if mode == "top_left":
                    row.append(g[r0][c0])

                elif mode == "bottom_right":
                    row.append(g[r0 + br - 1][c0 + bc - 1])

                elif mode == "mode":
                    row.append(Counter(vals).most_common(1)[0][0])

                elif mode == "nonzero":
                    nz = [v for v in vals if v != 0]
                    if nz:
                        row.append(Counter(nz).most_common(1)[0][0])
                    else:
                        row.append(0)

                else:
                    return None

            rows.append(tuple(row))

        return tuple(rows)

    candidates = []

    for br in range(2, 6):
        for bc in range(2, 6):
            for mode in ["top_left", "bottom_right", "mode", "nonzero"]:
                candidates.append((br, bc, mode))

    rules: list[Rule] = []

    for br, bc, mode in candidates:
        ok = True

        for inp, out in train:
            hi, wi = shape(inp)
            ho, wo = shape(out)

            if hi // br != ho or wi // bc != wo:
                ok = False
                break

            pred = compress(inp, br, bc, mode)

            if pred != out:
                ok = False
                break

        if ok:
            def predict(g: Grid, br=br, bc=bc, mode=mode) -> Optional[Grid]:
                return compress(g, br, bc, mode)

            rules.append(Rule(f"block_compress_{br}x{bc}_{mode}", 18, predict))

    return rules

def fit_translate_nonzero(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    learned_shift: Optional[tuple[int, int]] = None

    def translate(g: Grid, dr: int, dc: int) -> Optional[Grid]:
        h, w = shape(g)
        rows = [[0 for _ in range(w)] for _ in range(h)]

        for r in range(h):
            for c in range(w):
                v = g[r][c]

                if v == 0:
                    continue

                rr = r + dr
                cc = c + dc

                if not (0 <= rr < h and 0 <= cc < w):
                    return None

                if rows[rr][cc] != 0:
                    return None

                rows[rr][cc] = v

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        h, w = shape(inp)

        found_shift: Optional[tuple[int, int]] = None

        for dr in range(-h + 1, h):
            for dc in range(-w + 1, w):
                if dr == 0 and dc == 0:
                    continue

                if translate(inp, dr, dc) == out:
                    found_shift = (dr, dc)
                    break

            if found_shift is not None:
                break

        if found_shift is None:
            return []

        if learned_shift is None:
            learned_shift = found_shift
        elif learned_shift != found_shift:
            return []

    if learned_shift is None:
        return []

    dr, dc = learned_shift

    def predict(g: Grid, dr=dr, dc=dc) -> Optional[Grid]:
        return translate(g, dr, dc)

    return [Rule(f"translate_nonzero_{dr}_{dc}", 18, predict)]

def fit_rules(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []
    rules.extend(fit_2x2_weave(train))
    rules.extend(fit_translate_nonzero(train))
    rules.extend(fit_block_compress(train))
    rules.extend(fit_complete_symmetry(train))
    rules.extend(fit_kron_self_mask(train))
    rules.extend(fit_recolor_by_key_shape(train))
    rules.extend(fit_expand_crosses_5x5(train))
    rules.extend(fit_split_intersection_bar(train))
    rules.extend(fit_fill_rectangles_by_size(train))
    rules.extend(fit_fill_enclosed_regions(train))
    rules.extend(fit_staircase_components(train))
    rules.extend(fit_extend_vertical_period(train))
    rules.extend(fit_geom_and_color(train))
    rules.extend(fit_upscale(train))
    rules.extend(fit_crop_geom_color(train))
    rules.extend(fit_repeated_tile(train))
    rules.extend(fit_constant_output(train))
    return sorted(rules, key=lambda r: r.priority)


# ----------------------------
# Prediction, scoring, and submission
# ----------------------------

def fallback_predictions(inp: Grid) -> list[Grid]:
    h, w = shape(inp)
    bg = most_common_color(inp)
    return [
        inp,
        crop_non_bg(inp, 0),
        crop_non_bg(inp, bg),
        constant_grid(h, w, 0),
        constant_grid(h, w, bg),
        ((0,),),
    ]


def predict_task(task: dict, verbose: bool = False) -> tuple[list[dict], list[str]]:
    train = [
        (as_grid(pair["input"]), as_grid(pair["output"]))
        for pair in task["train"]
    ]
    tests = [as_grid(item["input"]) for item in task["test"]]

    rules = fit_rules(train)
    rule_names = [r.name for r in rules]

    if verbose:
        print("  rules:", rule_names)

    task_attempts = []

    for test_grid in tests:
        predictions: list[Grid] = []

        for rule in rules:
            try:
                pred = rule.predict(test_grid)
            except Exception:
                pred = None

            if is_valid_grid(pred):
                predictions.append(pred)  # type: ignore[arg-type]

        predictions.extend(fallback_predictions(test_grid))
        predictions = dedupe_grids(predictions)

        while len(predictions) < 2:
            predictions.append(((0,),))

        task_attempts.append({
            "attempt_1": as_rows(predictions[0]),
            "attempt_2": as_rows(predictions[1]),
        })

    return task_attempts, rule_names


def attempt_matches(attempt_record: dict, expected_output) -> bool:
    expected = as_grid(expected_output)

    attempt_1 = as_grid(attempt_record.get("attempt_1", []))
    attempt_2 = as_grid(attempt_record.get("attempt_2", []))

    return attempt_1 == expected or attempt_2 == expected


def score_task_attempts(task_attempts: list[dict], expected_outputs: list) -> tuple[int, int]:
    hits = 0
    total = 0

    for i, expected in enumerate(expected_outputs):
        total += 1

        if i >= len(task_attempts):
            continue

        if attempt_matches(task_attempts[i], expected):
            hits += 1

    return hits, total


def auto_find_solutions(challenges_path: str) -> Optional[Path]:
    p = Path(challenges_path)

    candidates = [
        p.with_name(p.name.replace("_challenges", "_solutions")),
        p.with_name(p.name.replace("challenges", "solutions")),
    ]

    for candidate in candidates:
        if candidate != p and candidate.exists():
            return candidate

    return None


def build_submission(
    challenges: dict,
    solutions: Optional[dict] = None,
    verbose: bool = False,
    progress_every: int = 25,
    stop_after: Optional[int] = None,
) -> tuple[dict, tuple[int, int, float]]:
    submission = {}

    running_hits = 0
    running_total = 0

    task_items = list(challenges.items())

    if stop_after is not None:
        task_items = task_items[:stop_after]

    total_tasks = len(task_items)

    for idx, (task_id, task) in enumerate(task_items, start=1):
        if verbose:
            print(f"\nSolving {task_id} [{idx}/{total_tasks}]")

        task_attempts, rule_names = predict_task(task, verbose=verbose)
        submission[task_id] = task_attempts

        task_status = "unscored"

        if solutions is not None and task_id in solutions:
            task_hits, task_total = score_task_attempts(task_attempts, solutions[task_id])
            running_hits += task_hits
            running_total += task_total

            if task_hits > 0:
                likely_rule = rule_names[0] if rule_names else "fallback_identity_or_crop"
                print(f"*** HIT {task_id}: {task_hits}/{task_total} correct; likely={likely_rule} ***")

                for i, expected in enumerate(solutions[task_id]):
                    expected_grid = as_grid(expected)

                    attempt_1 = as_grid(task_attempts[i]["attempt_1"])
                    attempt_2 = as_grid(task_attempts[i]["attempt_2"])

                    if attempt_1 == expected_grid:
                        print(f"    test[{i}] matched attempt_1")
                    elif attempt_2 == expected_grid:
                        print(f"    test[{i}] matched attempt_2")

            task_acc = task_hits / task_total if task_total else 0.0
            running_acc = running_hits / running_total if running_total else 0.0

            task_status = (
                f"task {task_hits}/{task_total} = {task_acc:.1%}; "
                f"running {running_hits}/{running_total} = {running_acc:.2%}"
            )

        should_report = (
            verbose
            or idx == 1
            or idx == total_tasks
            or (progress_every > 0 and idx % progress_every == 0)
        )

        if should_report:
            print(
                f"[{idx:4d}/{total_tasks}] {task_id}: {task_status}; "
                f"rules={len(rule_names)}"
            )

    final_acc = running_hits / running_total if running_total else 0.0
    return submission, (running_hits, running_total, final_acc)

def inspect_task(task_id: str, challenges: dict, solutions: Optional[dict] = None) -> None:
    if task_id not in challenges:
        print(f"Task not found: {task_id}")
        return

    task = challenges[task_id]

    print()
    print(f"=== INSPECT {task_id} ===")

    for i, pair in enumerate(task["train"]):
        inp = as_grid(pair["input"])
        out = as_grid(pair["output"])

        print()
        print(f"TRAIN[{i}] input shape={shape(inp)} colors={dict(colors(inp))}")
        print(as_rows(inp))

        print(f"TRAIN[{i}] output shape={shape(out)} colors={dict(colors(out))}")
        print(as_rows(out))

    for i, item in enumerate(task["test"]):
        inp = as_grid(item["input"])

        print()
        print(f"TEST[{i}] input shape={shape(inp)} colors={dict(colors(inp))}")
        print(as_rows(inp))

        if solutions is not None and task_id in solutions and i < len(solutions[task_id]):
            sol = as_grid(solutions[task_id][i])
            print(f"TEST[{i}] solution shape={shape(sol)} colors={dict(colors(sol))}")
            print(as_rows(sol))

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline ARC rule-based solver")
    parser.add_argument("--challenges", required=True, help="Path to ARC challenges JSON")
    parser.add_argument("--out", default="submission.json", help="Output submission JSON")
    parser.add_argument(
        "--solutions",
        help="Optional solutions JSON. If omitted, the solver tries to auto-detect it.",
    )
    parser.add_argument(
        "--no-auto-solutions",
        action="store_true",
        help="Disable automatic solutions-file detection.",
    )

    parser.add_argument("--inspect", help="Print one task's train/test/solution grids and exit.")

    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N tasks. Use 1 for every task.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Debug mode: only solve the first N tasks.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.challenges, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    solution_path: Optional[Path] = None

    if args.solutions:
        solution_path = Path(args.solutions)
    elif not args.no_auto_solutions:
        solution_path = auto_find_solutions(args.challenges)

    solutions = None

    if args.inspect:
        inspect_task(args.inspect, challenges, solutions)
        return

    if solution_path is not None and solution_path.exists():
        with open(solution_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)
        print(f"Loaded solutions: {solution_path}")
    else:
        print("No solutions file loaded. Predictions will be generated but not scored.")

    submission, (hits, total, acc) = build_submission(
        challenges,
        solutions=solutions,
        verbose=args.verbose,
        progress_every=args.progress_every,
        stop_after=args.stop_after,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    print()
    print(f"Wrote {args.out} with {len(submission)} tasks.")

    if total:
        print(f"Final exact-match score: {hits}/{total} = {acc:.3%}")
    else:
        print("Final exact-match score: unavailable because no solutions were loaded.")


if __name__ == "__main__":
    main()