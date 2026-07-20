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


def fit_rules(train: list[tuple[Grid, Grid]]) -> list[Rule]:
    rules: list[Rule] = []
    rules.extend(fit_2x2_weave(train))
    rules.extend(fit_kron_self_mask(train))
    rules.extend(fit_recolor_by_key_shape(train))
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