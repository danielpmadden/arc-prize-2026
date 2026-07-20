# Checkpoints and Milestones

This file records confirmed production score milestones and intentionally keeps chronological detail out of the repo-level README.

## Milestones

| Score | Task-normalized | Date | Description | Notable rules added | New task IDs | Lab after promotion | Commit | Checkpoint |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 24/1076 = 2.230% | - | 2026-07-04 | Initial basic solver baseline. | Early basic deterministic transformations. | - | - | - | `checkpoints/arc_solver_checkpoint_24.py` |
| 39/1076 = 3.625% | - | 2026-07-05 | Targeted handwritten rules and 2025 repo adaptations. | Intersection bars, enclosed-region filling, rectangle-size fills, vertical periodic extension, staircase components, cross expansion, symmetry completion, block compression, nonzero translation, and color-mapped nonzero translation. | - | - | - | `checkpoints/arc_solver_checkpoint_39.py` |
| 41/1076 = 3.810% | - | 2026-07-11 | Connect/dilate lab promotions. | Connect and dilate rule families. | - | - | `f3831d9` | No standalone checkpoint found. |
| 48/1076 = 4.461% | - | 2026-07-11 | Component-size recolor and panel OR. | `recolor_components_by_size`, `overlay_two_panels_or`. | - | - | `872830a` | No standalone checkpoint found. |
| 49/1076 = 4.554% | - | 2026-07-11 | D4-conjugated connection. | D4-conjugated same-color pair connection. | - | - | - | `checkpoints/checkpoint_049_1076_solver_4_554pct/` |
| 58/1076 = 5.390% | - | 2026-07-11 | Periodic tile, rank recolor, singleton ray. | `periodic_tile_with_phase_crop`, `recolor_components_by_rank`, `extend_ray_singletons_down`. | - | 0 new-hit candidates, 0 errors. | - | `checkpoints/checkpoint_058_1076_solver_5_390pct/` |
| 62/1076 = 5.762% | 60.000 | 2026-07-12 | Separator crop and periodic-add rules. | `sep_crop_largest_component`, `periodic_add_complete_lattice_10_10`, `periodic_add_full_grid_period_2_2`. | - | 0 new-hit candidates, 0 errors. | `7395c8b62d304de4ccfb013d49c65c78bd1a0a96` | - |
| 66/1076 = 6.134% | 63.000 | 2026-07-12 | Promoted shortest Manhattan marker connection and midpoint-cross placement. | `connect_two_markers_manhattan`, `place_cross_at_marker_midpoint`. | `a2fd1cf0`, `d4a91cb9`, `e9614598` | `python tools/rule_lab.py --all --only-new-hits --diagnostics`: 0 remaining new hits, 0 errors. | efbe6f70363e916241150a9404c3bf53192589dc | `checkpoints/checkpoint_066_1076_solver_6_134pct/` |

## Checkpoint notes

Older checkpoints are single-file snapshots from the pre-package solver layout. Directory checkpoints capture the split production package plus rule-lab and documentation files. Checkpoints should avoid large data files and generated submissions unless the convention changes.
