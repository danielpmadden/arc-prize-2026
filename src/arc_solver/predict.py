from __future__ import annotations

from src.arc_solver.types import Grid, Rule
from src.arc_solver.grid_utils import (
    as_grid, as_rows, constant_grid, crop_non_bg, dedupe_grids, is_valid_grid, most_common_color, shape,
)
from src.arc_solver.rules_basic import (
    fit_2x2_weave, fit_constant_output, fit_crop_geom_color, fit_geom_and_color,
    fit_kron_self_mask, fit_recolor_by_key_shape, fit_repeated_tile, fit_upscale,
)
from src.arc_solver.rules_special import (
    fit_block_compress, fit_complete_symmetry, fit_expand_crosses_5x5,
    fit_extend_vertical_period, fit_fill_enclosed_regions, fit_fill_rectangles_by_size,
    fit_split_intersection_bar, fit_staircase_components, fit_translate_nonzero,
    fit_translate_nonzero_color_map,
)


# WARNING: rule order affects output attempts and the confirmed training score.
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
    rules.extend(fit_translate_nonzero_color_map(train))
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

