# ARC Prize 2026 Rule-Based Solver

A small Python rule-based solver for ARC-AGI 2026 experimentation. The active solver lives in `src/arc_solver.py`, with a root-level compatibility launcher preserved as `arc_solver.py` for existing commands.

Current confirmed training score: **39/1076 = 3.625%**.

## Requirements

- Python 3.10+
- No required third-party Python packages for the solver CLI

## File Structure

```text
.
├── arc_solver.py                  # Compatibility launcher
├── src/
│   └── arc_solver.py              # Active solver implementation
├── data/                          # ARC challenge, solution, and sample JSON files
├── checkpoints/                   # Historical solver checkpoints
├── outputs/                       # Generated submissions and run artifacts
└── README.md
```

## Common Commands

Run the full training score:

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100
```

Run a first 10 task smoke test:

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 1 --stop-after 10
```

Inspect a task:

```bash
python arc_solver.py --challenges data/arc-agi_training_challenges.json --inspect <task_id>
```

Generate a test submission:

```bash
python arc_solver.py --challenges data/arc-agi_test_challenges.json --out outputs/test_submission.json --no-auto-solutions
```

Restore a checkpoint:

```bash
cp checkpoints/arc_solver_checkpoint_39.py src/arc_solver.py
```

The launcher also supports old-style bare filenames when possible by checking both the repository root and `data/`:

```bash
python arc_solver.py --challenges arc-agi_training_challenges.json --progress-every 100
```

## Current Checkpoint History

- `24` — `checkpoints/arc_solver_checkpoint_24.py`
- `32` — `checkpoints/arc_solver_checkpoint_32.py`
- `34` — `checkpoints/arc_solver_checkpoint_34.py`
- `36` — `checkpoints/arc_solver_checkpoint_36.py`
- `38` — `checkpoints/arc_solver_checkpoint_38.py`
- `39` — `checkpoints/arc_solver_checkpoint_39.py`

## Development Philosophy

- Change one function at a time.
- Add or remove one `fit_rules` line at a time.
- Run a full-score test after every solver change.
- Keep only rules that improve score.
- Checkpoint after every improvement.

## Known Useful Rule Families

Rules currently worth preserving include transformations for intersection bars, enclosed region filling, rectangle-size fills, vertical periodic extension, staircase components, 5x5 cross expansion, symmetry completion, block compression, nonzero translation, and color-mapped nonzero translation.

In particular, keep the existing working rules such as:

- `split_intersection_bar`
- `fill_enclosed_regions`
- `fill_rectangles_by_size`
- `extend_vertical_period`
- `staircase_components`
- `expand_crosses_5x5`
- `complete_symmetry`
- `block_compress`
- `translate_nonzero`
- `translate_nonzero_color_map`

## Rejected Experiments

Do not re-add these unless a future change clearly proves an improvement:

- `fit_project_profile`
- `fit_complete_rot180_symmetry`
- `fit_filter_color`
- `fit_crop_selected_component`
- `fit_gridline_cell_compress`
- `fit_crop_then_upscale`
