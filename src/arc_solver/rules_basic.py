from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from .types import Grid, Rule
from .grid_utils import (
    as_grid, bbox_for_positions, colors, constant_grid, crop_non_bg, crop_rect,
    most_common_color, nonzero_colors, nonzero_positions, normalize_mask, positions_of, shape,
)
from .transforms import GEOMS, apply_color_map, infer_color_map


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
