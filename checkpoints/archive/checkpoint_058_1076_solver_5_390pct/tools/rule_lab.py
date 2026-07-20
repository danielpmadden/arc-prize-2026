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
    loo: dict[str, str] = field(default_factory=dict)
    support: dict[str, tuple[int, list[str]]] = field(default_factory=dict)
    residuals: dict[str, str] = field(default_factory=dict)


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


def _grid_sig(g: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in g)


def _loo_status(family: str, rule_name: str, train: list[tuple[Grid, Grid]]) -> str:
    if len(train) < 2:
        return "partial"
    passed = 0
    total = 0
    for i, (held_inp, held_out) in enumerate(train):
        subset = train[:i] + train[i + 1:]
        try:
            rules = FAMILY_FITTERS[family](subset)
        except Exception:
            total += 1
            continue
        matches = [r for r in rules if r.name == rule_name]
        total += 1
        ok = False
        for rule in matches:
            try:
                ok = rule.predict(held_inp) == held_out
            except Exception:
                ok = False
            if ok:
                break
        if ok:
            passed += 1
    if passed == total and total:
        return "yes"
    if passed == 0:
        return "no"
    return "partial"


def _color_map_repairable(pred: Grid, exp: Grid) -> bool:
    if not is_valid_grid(pred) or not is_valid_grid(exp) or len(pred) != len(exp) or len(pred[0]) != len(exp[0]):
        return False
    mapping: dict[int, int] = {}
    for r, row in enumerate(pred):
        for c, v in enumerate(row):
            dst = exp[r][c]
            if v in mapping and mapping[v] != dst:
                return False
            mapping[v] = dst
    return True


def _residual_summary(pred: Grid | None, exp_rows) -> str:
    exp = as_grid(exp_rows)
    if not is_valid_grid(pred):
        return "shape_match=no mismatch_count=? mismatch_bbox=- added_cells_count=? deleted_cells_count=? recolored_cells_count=? foreground_mask_agreement=no color_map_repairable=no"
    p = pred  # type: ignore[assignment]
    shape_match = len(p) == len(exp) and (not p or not exp or len(p[0]) == len(exp[0]))
    if not shape_match:
        return "shape_match=no mismatch_count=? mismatch_bbox=- added_cells_count=? deleted_cells_count=? recolored_cells_count=? foreground_mask_agreement=no color_map_repairable=no"
    mism=[]; added=deleted=recolored=fg_agree=fg_total=0
    for r,row in enumerate(p):
        for c,v in enumerate(row):
            e=exp[r][c]
            if (v != 0) == (e != 0): fg_agree += 1
            fg_total += 1
            if v != e:
                mism.append((r,c))
                if v == 0 and e != 0: added += 1
                elif v != 0 and e == 0: deleted += 1
                else: recolored += 1
    bbox = "-" if not mism else f"({min(r for r,_ in mism)},{min(c for _,c in mism)})-({max(r for r,_ in mism)},{max(c for _,c in mism)})"
    fg = f"{fg_agree}/{fg_total}"
    repair = "yes" if _color_map_repairable(p, exp) else "no"
    return f"shape_match=yes mismatch_count={len(mism)} mismatch_bbox={bbox} added_cells_count={added} deleted_cells_count={deleted} recolored_cells_count={recolored} foreground_mask_agreement={fg} color_map_repairable={repair}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline experimental ARC rule-discovery lab.")
    parser.add_argument("--all", action="store_true", help="Run all families (default).")
    parser.add_argument("--family", action="append", help="Family name or alias to run; repeatable.")
    parser.add_argument("--task", help="Run one task id only.")
    parser.add_argument("--limit", type=int, help="Limit number of tasks processed.")
    parser.add_argument("--only-new-hits", action="store_true", help="Hide candidates with zero new hits.")
    parser.add_argument("--challenges", default="data/arc-agi_training_challenges.json")
    parser.add_argument("--solutions", default="data/arc-agi_training_solutions.json")
    parser.add_argument("--diagnostics", action="store_true", help="Report leave-one-out, support grouping, and compact residuals for evidenced candidates.")
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
        task_support: dict[tuple[int, tuple[tuple[int, ...], ...]], list[str]] = {}
        task_predictions: dict[tuple[str, str], tuple[list[Grid | None], set[int], set[int]]] = {}
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
                preds_for_diag: list[Grid | None] = []
                for item in task["test"]:
                    try:
                        p = rule.predict(as_grid(item["input"]))
                    except Exception:
                        p = None
                    preds_for_diag.append(p if is_valid_grid(p) else None)  # type: ignore[arg-type]
                if args.diagnostics:
                    for ti, pred in enumerate(preds_for_diag):
                        if pred is not None:
                            task_support[(ti, _grid_sig(pred))] = task_support.get((ti, _grid_sig(pred)), []) + [rule.name]
                    task_predictions[(family, rule.name)] = (preds_for_diag, cand_hit_idxs, new_idxs)
                if cand_hit_idxs:
                    rec.tasks.add(task_id)

        if args.diagnostics:
            for (family, name), (preds, cand_hit_idxs, new_idxs) in task_predictions.items():
                if not (cand_hit_idxs or new_idxs):
                    continue
                rec = stats[(family, name)]
                if new_idxs:
                    rec.loo[task_id] = _loo_status(family, name, train)
                for ti, pred in enumerate(preds):
                    if pred is None:
                        continue
                    supporters = task_support.get((ti, _grid_sig(pred)), [])
                    if len(supporters) > 1 and (ti in cand_hit_idxs or ti in new_idxs):
                        rec.support[f"{task_id}#{ti}"] = (len(supporters), sorted(set(supporters)))
                    if ti < len(expected) and ti not in cand_hit_idxs:
                        rec.residuals[f"{task_id}#{ti}"] = _residual_summary(pred, expected[ti])

    rows = sorted(stats.values(), key=lambda r: (-r.new, -r.hits, r.family, r.name))
    if args.only_new_hits:
        rows = [r for r in rows if r.new > 0]

    print("Family | Rule Name | Hits | New | Overlap | Tasks")
    print("--- | --- | ---: | ---: | ---: | ---")
    for r in rows:
        task_list = ", ".join(sorted(r.tasks)) if r.tasks else "-"
        err = f" errors={r.errors}" if r.errors else ""
        print(f"{r.family} | {r.name} | {r.hits} | {r.new} | {r.overlap} | {task_list}{err}")
        if args.diagnostics:
            if r.loo:
                print(f"  diagnostics: LOO pass: {'; '.join(f'{k}={v}' for k, v in sorted(r.loo.items()))}")
            for key, (count, names) in sorted(r.support.items()):
                print(f"  diagnostics: {key} support_count={count} supporting_rule_names={', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
            for key, summary in sorted(r.residuals.items()):
                print(f"  diagnostics: {key} residual {summary}")

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
