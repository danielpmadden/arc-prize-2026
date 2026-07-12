from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from src.arc_solver.types import Grid, Rule
from src.arc_solver.grid_utils import (
    as_grid, bbox_for_positions, colors, constant_grid, crop_rect, is_valid_grid, most_common_color,
    nonzero_colors, nonzero_positions, positions_of, shape,
)
from src.arc_solver.transforms import infer_color_map, apply_color_map, rot90, rot270, transpose


# Proven custom high-value ARC rules; keep their behavior narrow and order-controlled in predict.py.

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


def fit_connect_same_color_pairs(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def predict(g: Grid) -> Grid:
        rows = [list(row) for row in g]
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
        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []
        if predict(inp) != out:
            return []

    return [Rule("connect_same_color_pairs", 17, predict)]


def fit_d4_connect_same_color_pairs(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def anti_transpose_grid(g: Grid) -> Grid:
        h, w = shape(g)
        return tuple(tuple(g[h - 1 - r][w - 1 - c] for r in range(h)) for c in range(w))

    def connect_same_color_pairs(g: Grid) -> Grid:
        rows = [list(row) for row in g]
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
        return as_grid(rows)

    ops = [
        ("rot90", rot90, rot270),
        ("rot270", rot270, rot90),
        ("transpose", transpose, transpose),
        ("anti_transpose", anti_transpose_grid, anti_transpose_grid),
    ]

    rules: list[Rule] = []
    for op_name, forward, inverse in ops:
        ok = True
        for inp, out in train:
            if shape(inp) != shape(out):
                ok = False
                break
            if connect_same_color_pairs(forward(inp)) != forward(out):
                ok = False
                break
        if not ok:
            continue

        def predict(g: Grid, forward=forward, inverse=inverse) -> Grid:
            return inverse(connect_same_color_pairs(forward(g)))

        rules.append(Rule(f"d4_{op_name}_connect_same_color_pairs", 16, predict))

    return rules


def fit_dilate_8_added_color_1(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def predict(g: Grid) -> Grid:
        h, w = shape(g)
        rows = [list(row) for row in g]
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                if v == 0:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == 0:
                            rows[nr][nc] = 1
        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []
        if predict(inp) != out:
            return []

    return [Rule("dilate_8_added_color_1", 20, predict)]

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



def fit_recolor_components_by_size(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    size_to_color: dict[int, int] = {}
    changed = False

    def components4(g: Grid) -> list[list[tuple[int, int]]]:
        h, w = shape(g)
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []

        for r in range(h):
            for c in range(w):
                if g[r][c] == 0 or (r, c) in seen:
                    continue

                color = g[r][c]
                stack = [(r, c)]
                comp: list[tuple[int, int]] = []

                while stack:
                    rr, cc = stack.pop()

                    if (rr, cc) in seen:
                        continue

                    if not (0 <= rr < h and 0 <= cc < w):
                        continue

                    if g[rr][cc] != color:
                        continue

                    seen.add((rr, cc))
                    comp.append((rr, cc))
                    stack.append((rr - 1, cc))
                    stack.append((rr + 1, cc))
                    stack.append((rr, cc - 1))
                    stack.append((rr, cc + 1))

                comps.append(comp)

        return comps

    def apply_rule(g: Grid, mapping: dict[int, int]) -> Grid:
        rows = [list(row) for row in g]

        for comp in components4(g):
            size = len(comp)
            if size not in mapping:
                continue

            out_color = mapping[size]
            for r, c in comp:
                rows[r][c] = out_color

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        for comp in components4(inp):
            out_colors = {out[r][c] for r, c in comp}
            if len(out_colors) != 1:
                return []

            out_color = next(iter(out_colors))
            if out_color == 0:
                return []

            in_colors = {inp[r][c] for r, c in comp}
            if len(in_colors) != 1:
                return []

            if out_color == next(iter(in_colors)):
                continue

            size = len(comp)
            if size in size_to_color and size_to_color[size] != out_color:
                return []

            size_to_color[size] = out_color
            changed = True

        if apply_rule(inp, size_to_color) != out:
            return []

    if not changed:
        return []

    def predict(g: Grid, mapping=size_to_color) -> Grid:
        return apply_rule(g, mapping)

    return [Rule("recolor_components_by_size", 16, predict)]



def _components4_same_color(g: Grid) -> list[list[tuple[int, int]]]:
    h, w = shape(g)
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 or (r, c) in seen:
                continue

            color = g[r][c]
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in seen:
                    continue
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if g[rr][cc] != color:
                    continue

                seen.add((rr, cc))
                comp.append((rr, cc))
                stack.append((rr - 1, cc))
                stack.append((rr + 1, cc))
                stack.append((rr, cc - 1))
                stack.append((rr, cc + 1))

            comps.append(comp)

    return comps


def fit_recolor_components_by_rank(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def comp_info(comp: list[tuple[int, int]]) -> dict:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        return {
            "cells": comp,
            "r0": min(rs), "c0": min(cs),
            "r1": max(rs), "c1": max(cs),
            "area": len(comp),
            "h": max(rs) - min(rs) + 1,
            "w": max(cs) - min(cs) + 1,
        }

    def sort_infos(infos: list[dict], mode: str) -> list[dict]:
        reverse = mode.endswith("descending") or mode in ("right_to_left", "bottom_to_top")
        if mode.startswith("area_"):
            key = lambda x: (x["area"], x["r0"], x["c0"])
        elif mode.startswith("height_"):
            key = lambda x: (x["h"], x["r0"], x["c0"])
        elif mode.startswith("width_"):
            key = lambda x: (x["w"], x["r0"], x["c0"])
        elif mode in ("left_to_right", "right_to_left"):
            key = lambda x: (x["c0"], x["r0"], x["area"])
        else:
            key = lambda x: (x["r0"], x["c0"], x["area"])
        return sorted(infos, key=key, reverse=reverse)

    def apply_rule(g: Grid, mode: str, mapping: dict[int, int]) -> Grid:
        infos = [comp_info(comp) for comp in _components4_same_color(g)]
        rows = [list(row) for row in g]
        for idx, info in enumerate(sort_infos(infos, mode)):
            if idx not in mapping:
                continue
            for r, c in info["cells"]:
                rows[r][c] = mapping[idx]
        return as_grid(rows)

    modes = (
        "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top",
        "area_ascending", "area_descending", "height_ascending", "height_descending",
        "width_ascending", "width_descending",
    )
    rules: list[Rule] = []

    for mode in modes:
        mapping: dict[int, int] = {}
        changed = False
        ok = True

        for inp, out in train:
            if shape(inp) != shape(out):
                ok = False
                break
            for r, row in enumerate(inp):
                for c, v in enumerate(row):
                    if v == 0 and out[r][c] != 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

            infos = [comp_info(comp) for comp in _components4_same_color(inp)]
            for idx, info in enumerate(sort_infos(infos, mode)):
                vals = {out[r][c] for r, c in info["cells"]}
                if len(vals) != 1 or 0 in vals:
                    ok = False
                    break
                out_color = next(iter(vals))
                if idx in mapping and mapping[idx] != out_color:
                    ok = False
                    break
                mapping[idx] = out_color
                if any(inp[r][c] != out_color for r, c in info["cells"]):
                    changed = True
            if not ok or apply_rule(inp, mode, mapping) != out:
                ok = False
                break

        if not ok or not changed:
            continue

        learned = dict(mapping)
        def predict(g: Grid, mode=mode, mapping=learned) -> Grid:
            return apply_rule(g, mode, mapping)

        rules.append(Rule(f"recolor_components_by_rank_{mode}", 16, predict))

    return rules


def fit_periodic_tile_with_phase_crop(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def tile_source(g: Grid, source_mode: str) -> Optional[Grid]:
        if source_mode == "full_input":
            return g if shape(g)[0] and shape(g)[1] else None
        if source_mode == "nonzero_bbox":
            pos = nonzero_positions(g)
        else:
            if source_mode == "bg_zero":
                bg = 0
            elif source_mode == "bg_most_common":
                bg = most_common_color(g)
            elif source_mode == "bg_corner":
                bg = g[0][0]
            else:
                return None
            pos = {(r, c) for r, row in enumerate(g) for c, value in enumerate(row) if value != bg}
        box = bbox_for_positions(pos)
        return crop_rect(g, box) if box is not None else None

    def periodic(tile: Grid, oh: int, ow: int, phase_r: int, phase_c: int) -> Optional[Grid]:
        th, tw = shape(tile)
        if th <= 0 or tw <= 0 or oh <= 0 or ow <= 0:
            return None
        return tuple(tuple(tile[(r + phase_r) % th][(c + phase_c) % tw] for c in range(ow)) for r in range(oh))

    def infer_output_shape(g: Grid) -> Optional[tuple[int, int]]:
        out_shapes = [shape(out) for _, out in train]
        if len(set(out_shapes)) == 1:
            return out_shapes[0]

        deltas = [(shape(out)[0] - shape(inp)[0], shape(out)[1] - shape(inp)[1]) for inp, out in train]
        if len(set(deltas)) == 1:
            h, w = shape(g)
            dh, dw = deltas[0]
            return h + dh, w + dw

        ratios: list[tuple[int, int]] = []
        for inp, out in train:
            ih, iw = shape(inp)
            oh, ow = shape(out)
            if ih == 0 or iw == 0 or oh % ih != 0 or ow % iw != 0:
                return None
            ratios.append((oh // ih, ow // iw))
        if len(set(ratios)) == 1:
            h, w = shape(g)
            rh, rw = ratios[0]
            return h * rh, w * rw
        return None

    if infer_output_shape(train[0][0]) is None:
        return []

    source_modes = ("full_input", "nonzero_bbox", "bg_zero", "bg_most_common", "bg_corner")
    rules: list[Rule] = []
    for source_mode in source_modes:
        first_tile = tile_source(train[0][0], source_mode)
        if first_tile is None:
            continue
        th0, tw0 = shape(first_tile)
        for phase_r in range(th0):
            for phase_c in range(tw0):
                ok = True
                for inp, out in train:
                    tile = tile_source(inp, source_mode)
                    if not is_valid_grid(tile):
                        ok = False
                        break
                    th, tw = shape(tile)  # type: ignore[arg-type]
                    if phase_r >= th or phase_c >= tw:
                        ok = False
                        break
                    oh, ow = shape(out)
                    if periodic(tile, oh, ow, phase_r, phase_c) != out:  # type: ignore[arg-type]
                        ok = False
                        break
                if not ok:
                    continue

                def predict(g: Grid, source_mode=source_mode, phase_r=phase_r, phase_c=phase_c) -> Optional[Grid]:
                    tile = tile_source(g, source_mode)
                    if not is_valid_grid(tile):
                        return None
                    out_shape = infer_output_shape(g)
                    if out_shape is None:
                        return None
                    oh, ow = out_shape
                    return periodic(tile, oh, ow, phase_r, phase_c)  # type: ignore[arg-type]

                rules.append(Rule(f"periodic_tile_{source_mode}_phase_{phase_r}_{phase_c}", 14, predict))

    return rules


def fit_extend_ray_singletons_down(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def extend(g: Grid, stop_mode: str) -> Grid:
        h, w = shape(g)
        rows = [list(row) for row in g]
        color_counts = colors(g)
        seeds = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != 0 and color_counts[v] == 1]

        for r, c in seeds:
            color = g[r][c]
            nr = r + 1
            while nr < h:
                v = g[nr][c]
                if stop_mode == "until_blocked" and v != 0:
                    break
                if stop_mode == "until_same_color" and v == color:
                    break
                if stop_mode == "until_different_nonzero" and v != 0 and v != color:
                    break
                if v == 0:
                    rows[nr][c] = color
                nr += 1
        return as_grid(rows)

    rules: list[Rule] = []
    for stop_mode in ("to_boundary", "until_blocked", "until_different_nonzero", "until_same_color"):
        ok = True
        changed = False
        for inp, out in train:
            if shape(inp) != shape(out):
                ok = False
                break
            pred = extend(inp, stop_mode)
            if pred != out:
                ok = False
                break
            if pred != inp:
                changed = True
        if not ok or not changed:
            continue

        def predict(g: Grid, stop_mode=stop_mode) -> Grid:
            return extend(g, stop_mode)

        rules.append(Rule(f"extend_ray_singletons_down_{stop_mode}", 18, predict))

    return rules

def fit_overlay_two_panels_or(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    def full_separator_cols(g: Grid) -> list[int]:
        h, w = shape(g)
        return [c for c in range(w) if h and g[0][c] != 0 and all(g[r][c] == g[0][c] for r in range(h))]

    def full_separator_rows(g: Grid) -> list[int]:
        h, _ = shape(g)
        return [r for r in range(h) if g[r] and g[r][0] != 0 and all(v == g[r][0] for v in g[r])]

    def overlay_panels(g: Grid, axis: str) -> Optional[Grid]:
        h, w = shape(g)

        if axis == "col":
            seps = full_separator_cols(g)
            if len(seps) != 1:
                return None
            sep = seps[0]
            if sep * 2 + 1 != w:
                return None
            left = tuple(tuple(g[r][c] for c in range(sep)) for r in range(h))
            right = tuple(tuple(g[r][c] for c in range(sep + 1, w)) for r in range(h))
            p_h, p_w = shape(left)
        else:
            seps = full_separator_rows(g)
            if len(seps) != 1:
                return None
            sep = seps[0]
            if sep * 2 + 1 != h:
                return None
            left = tuple(tuple(g[r][c] for c in range(w)) for r in range(sep))
            right = tuple(tuple(g[r][c] for c in range(w)) for r in range(sep + 1, h))
            p_h, p_w = shape(left)

        if shape(left) != shape(right):
            return None

        return tuple(
            tuple(left[r][c] if left[r][c] != 0 else right[r][c] for c in range(p_w))
            for r in range(p_h)
        )

    rules: list[Rule] = []
    for axis in ("col", "row"):
        if all(overlay_panels(inp, axis) == out for inp, out in train):
            name = f"overlay_two_panels_or_{axis}"
            rules.append(Rule(name, 15, lambda g, axis=axis: overlay_panels(g, axis)))

    return rules

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

def fit_translate_nonzero_color_map(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    learned_shift: Optional[tuple[int, int]] = None
    learned_color_map: dict[int, int] = {}

    def translate_with_map(
        g: Grid,
        dr: int,
        dc: int,
        color_map: dict[int, int],
    ) -> Optional[Grid]:
        h, w = shape(g)
        rows = [[0 for _ in range(w)] for _ in range(h)]

        for r in range(h):
            for c in range(w):
                v = g[r][c]

                if v == 0:
                    continue

                if v not in color_map:
                    return None

                rr = r + dr
                cc = c + dc

                if not (0 <= rr < h and 0 <= cc < w):
                    return None

                if rows[rr][cc] != 0:
                    return None

                rows[rr][cc] = color_map[v]

        return as_grid(rows)

    for inp, out in train:
        if shape(inp) != shape(out):
            return []

        h, w = shape(inp)
        found: Optional[tuple[int, int, dict[int, int]]] = None

        for dr in range(-h + 1, h):
            for dc in range(-w + 1, w):
                if dr == 0 and dc == 0:
                    continue

                local_map: dict[int, int] = {}
                ok = True

                for r in range(h):
                    for c in range(w):
                        v = inp[r][c]

                        if v == 0:
                            continue

                        rr = r + dr
                        cc = c + dc

                        if not (0 <= rr < h and 0 <= cc < w):
                            ok = False
                            break

                        out_v = out[rr][cc]

                        if out_v == 0:
                            ok = False
                            break

                        if v in local_map and local_map[v] != out_v:
                            ok = False
                            break

                        local_map[v] = out_v

                    if not ok:
                        break

                if not ok:
                    continue

                pred = translate_with_map(inp, dr, dc, local_map)

                if pred == out:
                    found = (dr, dc, local_map)
                    break

            if found is not None:
                break

        if found is None:
            return []

        dr, dc, local_map = found

        if learned_shift is None:
            learned_shift = (dr, dc)
        elif learned_shift != (dr, dc):
            return []

        for src, dst in local_map.items():
            if src in learned_color_map and learned_color_map[src] != dst:
                return []
            learned_color_map[src] = dst

    if learned_shift is None or not learned_color_map:
        return []

    dr, dc = learned_shift
    color_map = dict(learned_color_map)

    def predict(g: Grid, dr=dr, dc=dc, color_map=color_map) -> Optional[Grid]:
        return translate_with_map(g, dr, dc, color_map)

    return [Rule(f"translate_nonzero_color_map_{dr}_{dc}", 19, predict)]

