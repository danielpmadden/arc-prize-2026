# ARC Prize 2026 Solver

## Project Overview

This repository contains a compact, deterministic rule-based solver for ARC Prize 2026 / ARC-AGI experimentation. It emphasizes reproducible training-set scoring, careful rule promotion, and checkpointed solver snapshots at confirmed milestones.

## Current Status

Current confirmed production score: **58/1076 = 5.390%**.

## Repository Layout

```text
arc_solver.py       # Root CLI entry point
src/arc_solver/     # Production solver package
checkpoints/        # Solver snapshots for confirmed milestones
tools/rule_lab.py   # Experimental rule-candidate evaluator
tools/              # Supporting development tools
data/               # ARC challenge and solution JSON files
outputs/            # Generated local run artifacts
CHECKPOINTS.md      # Detailed milestone/checkpoint history
```

## High-Level Solver Architecture

The production solver loads ARC tasks, tests a prioritized set of hand-written deterministic rules against each task's training examples, and emits predictions from the first matching rule. The implementation is split across the root CLI wrapper and the `src/arc_solver/` package, with specialized rules separated from shared grid utilities, transforms, scoring, and CLI code.

## Rule Lab Overview

The rule lab evaluates experimental rule families before promotion. It compares candidate hits against the current production baseline so rules can be promoted only when they improve or preserve confirmed behavior.

## How to Run the Production Score

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
```

Expected current result:

```text
Final exact-match score: 58/1076 = 5.390%
```

## How to Run the Rule Lab

```bash
python tools/rule_lab.py --all --only-new-hits
```

Expected current summary:

```text
Production baseline hits: 58/1076
Candidate new hits: 0
Errors: 0
```

## Development Workflow

- Keep solver behavior deterministic and easy to score.
- Use the rule lab for experiments before changing production logic.
- Promote only rules that improve or preserve the confirmed score.
- Re-run the production score and relevant lab checks after solver changes.
- Create a checkpoint after each confirmed score improvement.

## Milestone and Checkpoint History

Detailed milestone history and checkpoint notes live in [`CHECKPOINTS.md`](CHECKPOINTS.md). The latest checkpoint captures the **58/1076 = 5.390%** solver state.
