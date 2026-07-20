# ARC Prize 2026 Solver

Deterministic Python solver for ARC-AGI / ARC Prize 2026 experiments. The project prioritizes exact train-pair fitting, reproducible offline scoring, strict grid validation, and conservative promotion of small hand-written rules.

## Current production status

- **Exact output-level score:** 66/1076 = **6.134%**
- **Task-normalized score:** **63.000**
- **Fully solved tasks:** 62
- **Partially solved tasks:** 2

## Architecture

Production code fits deterministic rule functions from all train pairs, exact-validates each fitted rule, and returns up to two ranked attempts per test output. Specialized production rules live in `src/arc_solver/rules_special.py`; orchestration and ordering live in `src/arc_solver/predict.py`. Experimental rules remain isolated under `tools/` until a bounded candidate proves new exact hits without reducing the production score.

## Repository layout

```text
arc_solver.py                         # Root production scoring/submission CLI
src/arc_solver/predict.py             # Production rule fitting order and prediction assembly
src/arc_solver/rules_special.py       # Narrow promoted handwritten rules
src/arc_solver/grid_utils.py          # Shared grid conversion, validation, crops, helpers
src/arc_solver/transforms.py          # D4 and geometric transforms
src/arc_solver/scoring.py             # Attempt scoring helpers
tools/rule_lab.py                     # Offline candidate evaluation, diagnostics, signatures
tools/rule_generators.py              # Lab-only candidate families
CHECKPOINTS.md                        # Score and promotion history
docs/ASSISTANT_HANDOFF.md             # Durable context for future assistants/developers
checkpoints/                          # Confirmed solver snapshots
data/                                 # Local ARC challenge/solution JSON files
```

## Run production

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
```

Expected current result: `Final exact-match score: 66/1076 = 6.134%`.

## Run the rule lab

```bash
python tools/rule_lab.py --all --only-new-hits
python tools/rule_lab.py --all --only-new-hits --diagnostics
```

Expected current post-promotion result: production baseline `66/1076`, candidate new hits `0`, errors `0`.

## Signatures and residual inspection

```bash
python tools/rule_lab.py --signatures
python tools/rule_lab.py --inspect-residual TASK_ID
python tools/rule_lab.py --inspect-residual TASK_ID --show-grids
```

## Promotion workflow

1. Add bounded candidates only in `tools/rule_generators.py` / `tools/rule_lab.py`.
2. Run family-specific lab diagnostics and the full `--all --only-new-hits --diagnostics` lab.
3. Promote only candidates with exact train validation, exact new test hits, no errors, and LOO=yes where available.
4. Implement one narrow deterministic production fitter, with no task-ID exceptions and no per-test search.
5. Re-run all 1,076 training outputs and ensure lab new hits disappear after promotion.
6. Update [`CHECKPOINTS.md`](CHECKPOINTS.md), [`docs/ASSISTANT_HANDOFF.md`](docs/ASSISTANT_HANDOFF.md), and create a checkpoint snapshot.

## History and handoff

- Milestone history: [`CHECKPOINTS.md`](CHECKPOINTS.md)
- Assistant/developer handoff: [`docs/ASSISTANT_HANDOFF.md`](docs/ASSISTANT_HANDOFF.md)
