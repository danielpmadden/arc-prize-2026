from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .grid_utils import as_grid, as_rows, colors, shape
from .scoring import build_submission


def resolve_input_path(path_value: str) -> Path:
    """Resolve CLI input paths, falling back to data/ for bare filenames."""
    path = Path(path_value)

    candidates = [path]
    if path.parent == Path("."):
        candidates.append(Path("data") / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def auto_find_solutions(challenges_path: str | Path) -> Optional[Path]:
    p = Path(challenges_path)

    candidates = [
        p.with_name(p.name.replace("_challenges", "_solutions")),
        p.with_name(p.name.replace("challenges", "solutions")),
    ]

    if p.parent == Path("."):
        candidates.extend(
            [
                Path("data") / p.name.replace("_challenges", "_solutions"),
                Path("data") / p.name.replace("challenges", "solutions"),
            ]
        )

    for candidate in candidates:
        if candidate != p and candidate.exists():
            return candidate

    return None



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
    parser.add_argument(
        "--challenges",
        default="data/arc-agi_evaluation_challenges.json",
        help="Path to ARC challenges JSON (default: bundled evaluation challenges)",
    )
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

    challenges_path = resolve_input_path(args.challenges)

    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    solution_path: Optional[Path] = None

    if args.solutions:
        solution_path = resolve_input_path(args.solutions)
    elif not args.no_auto_solutions:
        solution_path = auto_find_solutions(challenges_path)

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
