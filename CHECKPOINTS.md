# Checkpoints and Milestones

This file records detailed solver milestone history that is intentionally kept out of the repo-level README.

## Milestones

| Score | Date | Description | Notable rules added | Checkpoint |
| --- | --- | --- | --- | --- |
| 24/1076 = 2.230% | 2026-07-04 | Initial basic solver baseline. | Early basic deterministic transformations. | `checkpoints/arc_solver_checkpoint_24.py` |
| 39/1076 = 3.625% | 2026-07-05 | Targeted hand-written rules and 2025 repo adaptations. | Intersection bars, enclosed-region filling, rectangle-size fills, vertical periodic extension, staircase components, cross expansion, symmetry completion, block compression, nonzero translation, and color-mapped nonzero translation. | `checkpoints/arc_solver_checkpoint_39.py` |
| 41/1076 = 3.810% | 2026-07-11 | Promoted lab-discovered connect/dilate rules. | Connect and dilate rule families discovered through the rule lab. | No standalone checkpoint found; see commit `f3831d9`. |
| 48/1076 = 4.461% | 2026-07-11 | Promoted `recolor_components_by_size` and `overlay_two_panels_or`. | Component recoloring by size and two-panel OR overlay. | No standalone checkpoint found; see commit `872830a`. |
| 49/1076 = 4.554% | 2026-07-11 | Promoted D4-conjugated `connect_same_color_pairs`. | D4-conjugated same-color pair connection. | `checkpoints/checkpoint_049_1076_solver_4_554pct/` |

## Checkpoint Notes

Older checkpoints are single-file snapshots from the pre-package solver layout. The 49-hit checkpoint uses a directory snapshot because the current production solver is split between the root `arc_solver.py` entry point and the `src/arc_solver/` package.

Checkpoint contents should avoid large data files and generated outputs unless the checkpoint convention changes in the future.
