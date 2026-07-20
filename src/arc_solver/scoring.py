from __future__ import annotations

from typing import Optional

from .grid_utils import as_grid
from .predict import predict_task


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
