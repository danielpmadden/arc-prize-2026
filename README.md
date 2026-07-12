# ARC Prize 2026 Solver

## Overview

This repository contains a compact, rule-based solver for ARC Prize 2026 / ARC-AGI experimentation. The focus is on deterministic transformations, fast training-set scoring, and careful promotion of rules that improve exact-match performance.

## Current Status

Training score: **49/1076 = 4.554%**.

## Repository Layout

```text
arc_solver.py        # Root CLI entry point
src/arc_solver/     # Production solver package
checkpoints/        # Solver snapshots for known milestones
tools/rule_lab.py   # Experimental rule-candidate evaluator
tools/              # Supporting development tools
data/               # ARC challenge and solution JSON files
outputs/            # Generated local run artifacts
```

## Solver Architecture

The solver loads ARC tasks, fits a prioritized list of hand-written rules against each task's training examples, and emits predictions from the first rule that matches. Production logic is organized under `src/arc_solver/`, with specialized rules separated from shared grid utilities, transforms, scoring, and CLI code.

## Rule Lab

The rule lab is used to evaluate experimental rule families before promotion. It reports candidates that solve additional tasks without changing the production solver until a rule is deliberately moved into the solver package.

## Running

Run the main training score:

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
```

Run the rule lab for remaining new-hit candidates:

```bash
python tools/rule_lab.py --all --only-new-hits
```

## Development Workflow

- Keep solver changes small and deterministic.
- Promote only rules that improve or preserve the confirmed score.
- Re-run the full training score after solver changes.
- Use the rule lab for experiments before changing production logic.
- Create a checkpoint after each confirmed score improvement.

## Checkpoints and Milestones

Detailed milestone history and checkpoint notes live in [`CHECKPOINTS.md`](CHECKPOINTS.md). The latest checkpoint captures the 49-hit solver state.
