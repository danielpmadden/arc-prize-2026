# Assistant Handoff

## Project identity

This is a deterministic Python ARC solver. Production rules are fitted from all train pairs, exact-validated against every train output, and then applied to test inputs with at most two attempts per test output. Production is intentionally simple and deterministic; experiments are isolated under `tools/` until promoted manually.

## Current verified state

- Exact output-level score: **66/1076 = 6.134%**
- Task-normalized score: **63.000**
- Fully solved tasks: **62**
- Partially solved tasks: **2**
- Latest validated commit hash: **efbe6f70363e916241150a9404c3bf53192589dc**
- Latest checkpoint: `CHECKPOINTS.md` entry for `66/1076 = 6.134%` and `checkpoints/checkpoint_066_1076_solver_6_134pct/`

## Repository map

- `arc_solver.py`: root CLI for production scoring and local submission generation.
- `src/arc_solver/predict.py`: production rule fitting order, fallback predictions, de-duplication, and two-attempt assembly.
- `src/arc_solver/rules_special.py`: promoted narrow handwritten rules.
- `src/arc_solver/grid_utils.py`: grid conversion, validation, shape, crop, color, and helper routines.
- `src/arc_solver/transforms.py`: geometric transforms such as D4 rotations/reflections.
- `src/arc_solver/scoring.py`: attempt matching and score helpers.
- `tools/rule_lab.py`: offline rule lab, production baseline comparison, diagnostics, LOO, signatures, and residual inspection.
- `tools/rule_generators.py`: lab-only candidate families.
- `checkpoints/`: confirmed snapshots of solver and lab/documentation state.
- `data/`: local ARC challenge and solution JSON files.

## Production architecture

Fitter functions learn task-constant parameters from train pairs, exact-validate those parameters on all train pairs, and return deterministic `Rule` objects. `predict_task` fits rules, runs each candidate safely, rejects malformed grids through validation, de-duplicates predictions, appends fallbacks, and emits two ranked attempts. A failing candidate must not crash the run.

## Rule-lab architecture

The lab supports named families and aliases, compares candidates against the current production baseline, reports new hits versus overlaps, enforces strict ARC grid validation, and provides LOO diagnostics when available. It also tracks task-normalized scoring, candidate provenance/support aggregation, residual signature clusters, `--inspect-residual`, and `--inspect-residual TASK_ID --show-grids`. Counterfactual/negative-control work is partially deferred: the CLI reports deferred negative controls rather than mutating candidate fitting.

Candidate de-duplication/support exists because multiple parameterizations can emit the same grid. Treat duplicate variants as weak evidence, not independent proof.

## Current production rules

Important active capabilities include:

- identity/color maps
- D4 geometry and color maps
- zero/mode/background crops
- repeat/tile/upscale/compress
- translations
- enclosure filling
- rectangle filling
- panel splitting and OR overlay
- symmetry completion
- same-color connection
- D4-conjugated connection
- component recolor by size
- component recolor by rank
- periodic tile with phase
- singleton downward ray
- separator crop largest component
- periodic lattice completion
- periodic 2x2 added-mask completion
- shortest Manhattan marker connection
- midpoint cross placement

## Milestone history

- **24/1076 = 2.230%**: initial basic solver.
- **39/1076 = 3.625%**: targeted handwritten rules.
- **41/1076 = 3.810%**: connect/dilate lab promotions.
- **48/1076 = 4.461%**: component-size recolor and panel OR.
- **49/1076 = 4.554%**: D4-conjugated connection.
- **58/1076 = 5.390%**: periodic tile, rank recolor, singleton ray.
- **62/1076 = 5.762%**: separator crop and periodic-add rules.
- **66/1076 = 6.134%**: shortest Manhattan marker connection and midpoint-cross placement.

## Rejected or deferred experiments

Keep these out of production until a bounded, exact, non-regressing mechanism is proven:

- `fit_project_profile`
- `fit_complete_rot180_symmetry`
- `fit_filter_color`
- broad crop-selected-component
- gridline-cell compression
- unsafe crop-then-upscale
- broad geometry-then-crop-color
- unrestricted rule composition
- large DSL/controller architecture
- neural/GPU integration in production
- random color permutation search
- last-output fallback
- zero-grid fallback
- automatic promotion

## Current unresolved residual mechanisms

### `3a301edc`

- nested rectangular structure
- existing components preserved
- added component is a rectangle/ring-like completion
- exact margin/shell recurrence not yet established
- do not implement a broad rectangle rule yet

### `4a1cacc2`

- recolor-only region expansion from a singleton marker
- same palette
- no additions/deletions under background-zero classification
- likely directional or boundary-fill behavior
- needs grid-level inspection

### `8dab14c2`

- same-palette boundary relocation
- sparse recolor changes
- component boundary shifts
- mechanism not yet established

### `7acdf6d3`

- singleton pattern recoloring
- likely local-neighborhood or pattern-completion rule
- needs 3x3-neighborhood diagnostics

### `bda2d7a6`

- nested shells/rings
- broad whole-grid changes
- likely shell shift or nested-region transformation
- defer until a precise recurrence is derived

## Next recommended experiments

1. Inspect full grids for `3a301edc` and derive nested ring recurrence.
2. Add boundary-displacement diagnostics for `4a1cacc2` and `8dab14c2`.
3. Add local 3x3 neighborhood signatures for `7acdf6d3`.
4. Derive nested shell-index transformation for `bda2d7a6`.
5. Rerun residual signatures after every production promotion.

## Exact commands

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
python tools/rule_lab.py --all --only-new-hits
python tools/rule_lab.py --all --only-new-hits --diagnostics
python tools/rule_lab.py --signatures
python tools/rule_lab.py --inspect-residual TASK_ID
python tools/rule_lab.py --inspect-residual TASK_ID --show-grids
```

## Non-negotiable development rules

- never drop train pairs
- never silently ignore malformed grids
- never promote a rule solely because it fits train examples
- never add task-ID special cases
- never let experimentation mutate production automatically
- never interpret duplicate generator variants as independent evidence
- always exact-validate all train pairs
- always run the full 1,076-output production score after promotion
