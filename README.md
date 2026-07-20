# ARC Prize Renegade

ARC Prize Renegade is a deterministic, rule-based Python solver for ARC-AGI / ARC Prize 2026 research. It fits a fixed collection of handwritten rule fitters to every training pair in a task, exact-validates a fitter before using it, and emits at most two ranked predictions per test input. It is an offline research repository, not a general-purpose ARC system.

## Repository status

**Implemented now.** The production solver is the `arc_solver` package. It provides deterministic grid validation, geometric/basic transformations, and a finite ordered set of promoted rule fitters. The command-line interface scores local challenge data and can write a local submission. The separate rule lab evaluates experimental candidate families; it does not alter production automatically.

**Most recently verified result.** The repository's recorded full training-corpus result is **72/1076 exact outputs (6.691%)**, task-normalized score **68.000**, with **67** fully solved and **2** partially solved tasks. This is a historical verification recorded at commit `c60f9d0`, not a claim that every checkout or unrecorded experiment has that result. See [verified history](#verified-history).

**Not implemented.** This repository does not implement Concepts, planning, learning, automatic rule promotion, a general DSL/controller, neural/GPU inference, or unrestricted rule composition. Mentions of those ideas in historical or experimental documents are not implemented capabilities.

## Install

The production solver uses only the Python standard library. Python 3.12 was used for the recorded verification.

```bash
python -m pip install --editable .
```

Editable installation exposes the production module and the `renegade` console command. Local ARC JSON data is versioned under `data/`; no download step is required for the commands below.

## Run

Run the installed production entry point:

```bash
python -m renegade --challenges data/arc-agi_training_challenges.json --progress-every 100
```

The checked-in root launcher remains equivalent after installation:

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
```

For available options, use `python -m renegade --help`. A full corpus score is intentionally an offline computation and may take roughly two minutes on the recorded environment.

### Experimental rule lab

`tools/rule_lab.py` is **experimental tooling**, not part of the production prediction path. It evaluates candidate rule families against the production baseline; its output is evidence for a maintainer, not an automatic promotion decision.

```bash
python tools/rule_lab.py --all --only-new-hits --diagnostics
python tools/rule_lab.py --signatures
```

## Test and check

The test suite currently contains a small import smoke test for the installed module entry point; it does not yet exercise solver behavior. Run the complete repository checks from the root:

```bash
python -m pip install --editable .
python -m unittest discover -s tests -v
python -m renegade
python -m compileall -q src tests
git diff --check
```

`python -m renegade` without arguments uses the bundled evaluation challenges and writes `submission.json` locally. That generated file is intentionally ignored by Git.

## Package and repository layout

```text
pyproject.toml                    # Build metadata and installed entry point
arc_solver.py                     # Backward-compatible root launcher
src/arc_solver/                   # Implemented production solver
  cli.py                          # Scoring, inspection, and submission CLI
  predict.py                      # Ordered fitter orchestration and attempts
  rules_basic.py                  # General deterministic fitters
  rules_special.py                # Narrow promoted fitters
  grid_utils.py, transforms.py    # Grid validation and transformations
  scoring.py, types.py            # Scoring/submission and shared types
src/renegade/                     # `python -m renegade` launcher only
tools/                            # Experimental, manually evaluated rule candidates
data/                             # Versioned local ARC challenge and solution JSON
tests/                            # Unit-test package (currently one entry-point smoke test)
CHECKPOINTS.md                    # Verified historical score/milestone record
checkpoints/                      # Immutable historical solver snapshots
docs/                             # Contributor guidance and historical experiment record
```

## Documentation guide

Read documents by their role; none of the documents below silently changes production behavior.

| Document or location | Role | Use it for |
| --- | --- | --- |
| This README | **Current implementation and contributor entry point** | Installation, commands, scope, and navigation. |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | **Verified historical record** | Confirmed score milestones, commits, and snapshots. It is not a roadmap or a specification. |
| [`checkpoints/`](checkpoints/) | **Historical record** | Frozen snapshots supporting the milestone record; do not treat their READMEs as current behavior. |
| [`docs/ASSISTANT_HANDOFF.md`](docs/ASSISTANT_HANDOFF.md) | **Contributor guidance** | Current code map, maintenance constraints, and explicitly labeled deferred work. |
| [`docs/MIME_EXPERIMENT.md`](docs/MIME_EXPERIMENT.md) | **Historical experiment record** | Evidence and decisions from one completed experiment; not an implementation specification. |
| `src/arc_solver/` | **Implementation** | The authoritative source for current solver behavior. |
| `tools/` | **Experimental implementation** | Candidate evaluation only; it is not a production capability claim. |

There are no `MILESTONES.md`, `CHANGELOG.md`, `CONSTITUTION.md`, `CAPABILITY_CONTRACT.md`, `LIFECYCLE.md`, `LINEAGE.md`, or `EMERGENCE.md` files in this checkout. Do not infer their contents or status from similarly named documents in other branches or repositories.

## Verified history

`CHECKPOINTS.md` is the sole repository-level chronology of confirmed production-score milestones. Each row records only the evidence that was captured at that milestone. The latest snapshot is `checkpoints/checkpoint_072_1076_solver_6_691pct/`; older snapshots live under `checkpoints/archive/`, whose index explains their scope.

The historical score is reproducible only relative to the recorded dataset and code revision. When changing solver behavior, rerun the full production score and record a new milestone only after the result is confirmed; do not overwrite or reinterpret prior rows.

## Contributor navigation

1. Start here, then run the checks above to establish your local baseline.
2. Read `src/arc_solver/predict.py` to see the production fitting order, then the relevant rule module. That order is part of the implemented behavior.
3. Use `CHECKPOINTS.md` for facts about confirmed past scores; use checkpoint directories only to inspect that past state.
4. Read `docs/ASSISTANT_HANDOFF.md` before proposing solver work. It separates current facts from deferred or rejected directions.
5. Keep experiments in `tools/` until a maintainer manually promotes a narrow, deterministic, exact-validated rule. Do not add task-ID exceptions or automatic promotion.
