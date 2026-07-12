from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arc_solver.types import Grid, Rule
from src.arc_solver.grid_utils import as_grid, as_rows, is_valid_grid, dedupe_grids
from src.arc_solver.predict import predict_task
from src.arc_solver.scoring import score_task_attempts, attempt_matches
from tools.rule_generators import FAMILY_ALIASES, FAMILY_FITTERS


@dataclass
class CandidateStats:
    family: str
    name: str
    hits: int = 0
    new: int = 0
    overlap: int = 0
    tasks: set[str] = field(default_factory=set)
    errors: int = 0


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_families(names: list[str] | None) -> list[str]:
    if not names:
        return list(FAMILY_FITTERS)
    selected: list[str] = []
    for name in names:
        key = FAMILY_ALIASES.get(name, name)
        if key not in FAMILY_FITTERS:
            valid = ", ".join(sorted(set(FAMILY_FITTERS) | set(FAMILY_ALIASES)))
            raise SystemExit(f"Unknown family '{name}'. Valid families/aliases: {valid}")
        if key not in selected:
            selected.append(key)
    return selected


def train_pairs(task: dict) -> list[tuple[Grid, Grid]]:
    return [(as_grid(p["input"]), as_grid(p["output"])) for p in task["train"]]


def candidate_attempts(rule: Rule, task: dict) -> tuple[list[dict], int]:
    attempts: list[dict] = []
    errors = 0
    for item in task["test"]:
        preds: list[Grid] = []
        try:
            pred = rule.predict(as_grid(item["input"]))
        except Exception:
            pred = None
            errors += 1
        if is_valid_grid(pred):
            preds.append(pred)  # type: ignore[arg-type]
        else:
            errors += 1
        preds = dedupe_grids(preds)
        while len(preds) < 2:
            preds.append(((0,),))
        attempts.append({"attempt_1": as_rows(preds[0]), "attempt_2": as_rows(preds[1])})
    return attempts, errors


def hit_indices(attempts: list[dict], expected_outputs: list) -> set[int]:
    return {i for i, _ in enumerate(expected_outputs) if i < len(attempts) and attempt_matches(attempts[i], expected_outputs[i])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline experimental ARC rule-discovery lab.")
    parser.add_argument("--all", action="store_true", help="Run all families (default).")
    parser.add_argument("--family", action="append", help="Family name or alias to run; repeatable.")
    parser.add_argument("--task", help="Run one task id only.")
    parser.add_argument("--limit", type=int, help="Limit number of tasks processed.")
    parser.add_argument("--only-new-hits", action="store_true", help="Hide candidates with zero new hits.")
    parser.add_argument("--challenges", default="data/arc-agi_training_challenges.json")
    parser.add_argument("--solutions", default="data/arc-agi_training_solutions.json")
    args = parser.parse_args()

    families = select_families(None if args.all or not args.family else args.family)
    challenges = load_json(ROOT / args.challenges)
    solutions = load_json(ROOT / args.solutions)

    items = list(challenges.items())
    if args.task:
        items = [(tid, task) for tid, task in items if tid == args.task]
        if not items:
            raise SystemExit(f"Task not found: {args.task}")
    if args.limit is not None:
        items = items[: args.limit]

    stats: dict[tuple[str, str], CandidateStats] = {}
    production_hits = production_total = 0
    all_errors = 0

    for task_id, task in items:
        if task_id not in solutions:
            continue
        expected = solutions[task_id]
        prod_attempts, _ = predict_task(task)
        ph, pt = score_task_attempts(prod_attempts, expected)
        production_hits += ph
        production_total += pt
        prod_hit_idxs = hit_indices(prod_attempts, expected)

        train = train_pairs(task)
        for family in families:
            try:
                rules = FAMILY_FITTERS[family](train)
            except Exception:
                all_errors += 1
                continue
            for rule in rules:
                key = (family, rule.name)
                rec = stats.setdefault(key, CandidateStats(family=family, name=rule.name))
                attempts, errors = candidate_attempts(rule, task)
                rec.errors += errors
                all_errors += errors
                ch, _ = score_task_attempts(attempts, expected)
                cand_hit_idxs = hit_indices(attempts, expected)
                new_idxs = cand_hit_idxs - prod_hit_idxs
                overlap_idxs = cand_hit_idxs & prod_hit_idxs
                rec.hits += ch
                rec.new += len(new_idxs)
                rec.overlap += len(overlap_idxs)
                if cand_hit_idxs:
                    rec.tasks.add(task_id)

    rows = sorted(stats.values(), key=lambda r: (-r.new, -r.hits, r.family, r.name))
    if args.only_new_hits:
        rows = [r for r in rows if r.new > 0]

    print(f"{'Family':30} {'Rule Name':55} {'Hits':>5} {'New':>4} {'Overlap':>7}  Tasks")
    for r in rows:
        task_list = ", ".join(sorted(r.tasks)) if r.tasks else "-"
        err = f" errors={r.errors}" if r.errors else ""
        print(f"{r.family:30} {r.name:55} {r.hits:5d} {r.new:4d} {r.overlap:7d}  {task_list}{err}")

    candidate_total = sum(r.hits for r in stats.values())
    candidate_new = sum(r.new for r in stats.values())
    print()
    print(f"Production baseline hits: {production_hits}/{production_total}")
    print(f"Candidate total hits: {candidate_total}")
    print(f"Candidate new hits: {candidate_new}")
    print(f"Errors: {all_errors}")
    print("Best candidates to promote:")
    promoted = [r for r in sorted(stats.values(), key=lambda r: (-r.new, -r.hits, r.name)) if r.new > 0]
    if promoted:
        for r in promoted[:20]:
            print(f"- {r.family} / {r.name}: new={r.new}, hits={r.hits}, tasks={', '.join(sorted(r.tasks))}")
    else:
        print("- None found in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
