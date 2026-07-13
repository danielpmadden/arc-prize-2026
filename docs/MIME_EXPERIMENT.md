# MIME Composition Experiment

## 1. Starting state

- Starting branch: `work`.
- Experimental branch: `codex/mime-composition-experiment`.
- Starting commit: `615c5a35bd9af64a9bbcd1bfdc29f0eaa23cfb57`.
- Working tree before edits: clean.
- Python: `Python 3.12.13`.

### Dataset hashes

- `data/arc-agi_training_challenges.json`: `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`.
- `data/arc-agi_training_solutions.json`: `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9`.
- `data/arc-agi_evaluation_challenges.json`: `e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8`.
- `data/arc-agi_evaluation_solutions.json`: `84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1`.
- `data/arc-agi_test_challenges.json`: `232264c58f825ee77327dcfc9f4e5cb2f83b8d997eb69032be1bf2205bbe1a83`.

### Reproduced baseline

- Command: `python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100`.
- Observed exact output score: `66/1076 = 6.134%`.
- Observed task-normalized score: `63.000`, from documented baseline.
- Fully solved tasks: `62`, from documented baseline.
- Partially solved tasks: `2`, from documented baseline.
- Runtime: `105.181` seconds.
- Exceptions: none in the completed baseline production run.

The documented baseline in `README.md` and `docs/ASSISTANT_HANDOFF.md` was `66/1076 = 6.134%`, task-normalized `63.000`, fully solved tasks `62`, partially solved tasks `2`.

## 2. Puzzle-piece inventory

Existing reusable primitives observed in production and lab code:

- Grid transforms: identity, D4 rotations/reflections/transpose, crop by background, upscale, block compression, repeated/periodic tiling, phase crop, overlay.
- Object extraction: nonzero positions, bounding boxes, colors, singleton markers, connected components in diagnostics, largest component after separator split.
- Selection: unique singleton marker, same-color marker pair, component size/rank, largest component, separator panels, nonzero bounding box.
- Relations: same row/column alignment, D4-conjugated alignment, midpoint between markers, shortest Manhattan path, symmetry, periodic lattice phase, panel intersection/OR.
- Actions: recolor, translate, copy/overlay, connect, dilate, fill enclosed regions, fill rectangles, complete symmetry, extend ray, place cross, crop selected component.

## 3. Residual cluster selected

Selected the single-marker recolor expansion cluster centered on residual task `4a1cacc2` because diagnostics showed:

- input/output shape preserved;
- palette preserved;
- exactly one singleton non-background marker in a mostly uniform background;
- output recolors a rectangle from that marker toward a grid corner;
- no added/deleted foreground under the two-color classification, only recoloring.

Related residuals inspected but not promoted:

- `3a301edc`: nested ring/shell completion; exact recurrence remained ambiguous.
- `8dab14c2`: sparse boundary relocation/recoloring; not a simple corner fill.

## 4. Search performed

This bounded experiment used manual/lab inspection rather than an unrestricted DSL. The effective composition tested was:

1. select unique singleton marker color;
2. compute all four marker-to-corner Manhattan distances;
3. reject if the nearest corner is ambiguous;
4. fill the closed axis-aligned rectangle spanned by marker and nearest corner with the marker color;
5. exact-validate every training pair before activation.

Limits:

- residual batch: 3 inspected tasks (`3a301edc`, `4a1cacc2`, `8dab14c2`), with production promotion only for `4a1cacc2`'s cluster;
- depth: 2 conceptual operations after selection (`nearest corner` relation + `rectangle fill` action);
- candidate constants: none; colors, marker location, shape, and corner are inferred per grid;
- ambiguity: equal nearest-corner distance rejects;
- runtime: full production scoring stayed near baseline (`104-107` seconds in repeated post-promotion runs).

## Discovery MIME-001: marker-to-nearest-corner rectangle fill

## Residual tasks examined

- Primary: `4a1cacc2`.
- Cluster description: two-color grids with one singleton marker color embedded in a uniform background, where demonstrations expand the marker into the rectangle connecting it to the nearest corner.

## Constraints observed

- Output shape equals input shape.
- Palette is preserved.
- Input contains exactly two colors.
- One color occurs exactly once and is the marker.
- Output differs only by recoloring background cells to marker color.
- The recolored cells form the closed rectangle from the marker to a grid corner.
- Across demonstrations, the target corner changes with marker position, so the corner cannot be a learned fixed corner.

## Existing pieces used

- `shape` for grid dimensions.
- `positions_of` for marker localization.
- `as_grid` for validated immutable grid construction.
- Existing production pattern of exact-validating all train pairs before returning a `Rule`.

## New primitive, if any

`fit_fill_marker_to_nearest_corner`: a narrow production primitive that selects a unique singleton marker and fills the marker-to-nearest-corner rectangle. Existing pieces could express marker selection and fill, but no existing production rule combined unique singleton selection with a nearest-corner relation and rectangle recoloring.

## Candidate program

`select unique singleton marker -> compute unique nearest grid corner -> fill closed marker/corner rectangle with marker color`

## Parameter inference

- Marker color: the only color with count `1` in the input.
- Marker location: the only cell of that marker color.
- Target corner: the unique grid corner with minimum Manhattan distance to the marker.
- Fill color: marker color.
- Fill bounds: min/max rows and columns of marker and selected corner.

No task ID, fixed coordinate, fixed color, fixed dimension, or expected test output is used.

## Exact-fit evidence

- All four `4a1cacc2` training demonstrations exact-fit the primitive before the rule activates.
- Targeted prediction for `4a1cacc2` produced `fill_marker_to_nearest_corner` and matched its held-out training-corpus test output.

## Corpus effect

- Previous score: `66/1076 = 6.134%`.
- Candidate score: `67/1076 = 6.227%`.
- New exact hits: `4a1cacc2` test output, `+1` output.
- Overlapping hits retained: all previously reported hits remained present in the two repeated full score runs.
- Lost hits: none observed.
- Task-normalized score: expected `64.000` after adding one newly fully solved one-output task.

## Failure cases

The rule rejects when:

- there are not exactly two colors;
- no color occurs exactly once;
- multiple singleton colors exist;
- marker-to-corner nearest distance is tied;
- train exact-validation fails.

## Generality assessment

Medium. The primitive is compact and relational, and it may transfer to other singleton-marker corner-expansion tasks, but the preconditions are intentionally narrow.

## Promotion status

Promoted on experiment branch.

## Unsuccessful hypotheses

- Nested shell/ring completion for `3a301edc`: rejected for this pass because expansion margins vary across demonstrations and the recurrence was not derived without ambiguity.
- Sparse boundary relocation for `8dab14c2`: rejected for this pass because changes are local boundary shifts, not corner rectangles or simple fills.

## 6. Score comparison

- Baseline exact score: `66/1076 = 6.134%`.
- New exact score: `67/1076 = 6.227%` in two consecutive full production runs.
- Baseline task-normalized score: `63.000`.
- New task-normalized score: expected `64.000` because `4a1cacc2` is a one-output task newly solved exactly.
- New hits: `4a1cacc2`.
- Lost hits: none observed in full-corpus scoring output.
- Runtime impact: baseline `105.181` seconds; post-promotion `107.030` and `104.156` seconds.
- Exceptions/invalid grids: none in final full production scoring runs.

## 7. Code changes

- `src/arc_solver/rules_special.py`: added `fit_fill_marker_to_nearest_corner`.
- `src/arc_solver/predict.py`: added the fitter to production imports and rule order after existing marker rules.
- `docs/MIME_EXPERIMENT.md`: recorded baseline, experiment, discovery, and scores.

## 8. Reproduction commands and real results

- `pwd`: `/workspace/arc-prize-2026`.
- `git status --short`: clean before edits.
- `git branch --show-current`: `work` before branch creation.
- `git rev-parse HEAD`: `615c5a35bd9af64a9bbcd1bfdc29f0eaa23cfb57`.
- `git log -8 --oneline`: recent history included `615c5a3 Merge pull request #16 ...` through `c0d82bd Add experimental rule lab families`.
- `git diff --stat`: no output before edits.
- `git diff`: no output before edits.
- `python --version`: `Python 3.12.13`.
- `sha256sum data/*.json`: hashes listed above.
- `python arc_solver.py --challenges data/arc-agi_training_challenges.json --progress-every 100`: baseline `66/1076 = 6.134%`, runtime `105.181` seconds.
- `python tools/rule_lab.py --inspect-residual 3a301edc --show-grids`: completed; showed nested shell completion constraints.
- `python tools/rule_lab.py --inspect-residual 4a1cacc2 --show-grids`: completed; showed singleton marker corner-rectangle evidence.
- `python tools/rule_lab.py --inspect-residual 8dab14c2 --show-grids`: completed; showed sparse boundary relocation evidence.
- `python -m py_compile src/arc_solver/rules_special.py src/arc_solver/predict.py`: passed.
- Targeted `predict_task` check for `4a1cacc2`: `['fill_marker_to_nearest_corner']`, test match `[True]`.
- Two-run full score command: both runs passed with `67/1076 = 6.227%`, runtimes `107.030` and `104.156` seconds.
- `python tools/rule_lab.py --all --only-new-hits --diagnostics`: attempted after promotion but manually stopped after several minutes without output; full production regression above completed and is the acceptance check for this promoted rule.

## 9. Recommendation

Recommend merging the specific score-positive commits if owner review accepts the narrow singleton-marker nearest-corner primitive. A second bounded experiment should target `3a301edc` nested shell recurrence or `8dab14c2` boundary relocation next.

# Continuation Round: cumulative MIME search from 67/1076

## Starting state for continuation

- Branch: `codex/mime-composition-experiment`.
- Starting commit: `db97c5e48228d9da4fb54ee0615310408caea46d`.
- Reproduced starting score twice: `67/1076 = 6.227%`.
- Reproduction runtimes: `103.678` seconds and `103.295` seconds.
- Preserved score-positive rule: `fit_fill_marker_to_nearest_corner`.

## Practical primitive catalog

A lab-only catalog was added in `tools/mime_lab.py`. It records direct candidate programs over these reusable pieces:

- singleton marker extraction;
- nearest, farthest, and fixed-corner relations;
- marker-to-corner rectangle fill;
- marker-to-edge line fill;
- non-background bounding-box extraction;
- bounding-box frame drawing with role-derived colors;
- production baseline comparison and residual clustering.

The catalog intentionally keeps each candidate printable as a short program name such as `singleton_rect_corner_nearest`, `singleton_line_edge_top`, or `bbox_frame_least_non_bg_pad_2`.

## Residual coverage and clusters

The continuation lab pass inspected and clustered `300` currently unsolved outputs, exceeding the requested minimum of 100. It formed `39` measurable residual clusters and searched `140` selected residual tasks sampled from the largest clusters.

Top clusters included:

- same-shape, palette-preserved, medium-change tasks: 66 tasks;
- shape-changing, one-color-removal, small-change tasks: 28 tasks;
- same-shape, palette-preserved, small-change tasks: 28 tasks;
- shape-changing, palette-preserved, small-change tasks: 18 tasks;
- same-shape, one-color-addition, small-change tasks: 18 tasks;
- same-shape, one-color-removal, medium-change tasks: 17 tasks;
- same-shape, one-color-addition, medium-change tasks: 17 tasks;
- same-shape, palette-preserved, large-change tasks: 9 tasks, including nested shell task `3a301edc`.

## Search statistics

- Unique candidate programs tested in `tools/mime_lab.py`: `2800`.
- Depth-1 count: `2800` direct primitive candidates.
- Depth-2/depth-3/depth-4 counts: `0` implemented in this continuation pass; the tested candidates were direct reusable primitive hypotheses.
- Exact-fitting candidates from this lab pass: `0`.
- New hit candidates from this lab pass: `0`.
- Existing rule-lab sampled families run over `--limit 300`: `structure_then_color_map`, `crop_to_background_bbox`, `recolor_preserved_masks_by_role`, `component_move_or_copy_by_offset`, and `bounded_hole_and_region_fill_variants`; none produced new hits in that sampled run.
- Two-attempt opportunity check before the second promotion: one hidden correct fallback for `5582e5ca`.
- Two-attempt opportunity check after the second promotion: `0` hidden correct predictions below attempt 2.

## Discovery MIME-002: constant most-frequent input color

## Tasks and cluster

- Primary task: `5582e5ca`.
- Cluster: small 3x3 grids where each output is a same-shape constant grid, and the constant output color is the unique most frequent color in the corresponding input.

## Forced constraints

- Output shape equals input shape for every demonstration.
- Output is constant for every demonstration.
- The output color varies across demonstrations, so it cannot be a fixed learned color.
- The output color equals the input's unique most frequent color in every demonstration.

## Eliminated mechanisms

- Not a global geometric transform.
- Not a crop or extraction.
- Not a fixed constant-output rule.
- Not arbitrary fallback ordering; the color role is inferable from demonstrations.

## Existing primitives reused

- Grid shape extraction.
- Color counting / mode-color role.
- Constant-grid construction.
- Exact train validation before rule activation.

## New primitive, if any

`fit_constant_most_frequent_color`: a compact production primitive that predicts a same-shape constant grid using the input's unique mode color. It rejects tied modes rather than choosing arbitrarily.

## Exact candidate program

`count input colors -> select unique most frequent color -> output same-shape constant grid of that color`

## Parameter inference

- Output shape: input shape.
- Output color: unique most frequent color in each input.
- No task ID, fixed color, fixed coordinate, or expected test output is used.

## Ambiguity rejection

- If the input mode color is tied, the rule returns no candidate.
- If any train output is non-constant, differently shaped, or not equal to the corresponding input mode color, the fitter rejects.

## Training-pair evidence

All three `5582e5ca` training demonstrations exact-fit the mode-color constant-grid rule.

## New corpus hits

- `5582e5ca` test output, `+1` exact output.

## Lost hits

None observed in the full production score run after promotion.

## Known failure cases

- Tied input mode colors.
- Non-constant outputs.
- Tasks where the output constant color is a non-mode role, such as center color, corner color, least frequent color, or a color absent from the input.

## Generality assessment

Medium-low. The rule is simple and reusable for mode-color abstraction tasks, but evidence is a one-task public-training gain.

## Production status

Promoted on experiment branch.

## Score progression

- `67/1076 = 6.227%`: starting continuation score with `fit_fill_marker_to_nearest_corner` preserved.
- `68/1076 = 6.320%`: promoted `fit_constant_most_frequent_color`, adding `5582e5ca`.

## Additional near misses and eliminated families

- Singleton marker corner/edge variants over 140 selected residual tasks: no exact fits beyond the already promoted nearest-corner rule.
- Bounding-box frame variants over selected residual tasks: no exact fits; nested shell tasks likely require a more precise shell-depth or recurrence primitive rather than simple padded frames.
- Crop/background-bbox and component-move sampled rule-lab families: produced only overlaps with existing production hits in the sampled run.
- Two-attempt ranking after MIME-002: no remaining correct production/fallback prediction was found below attempt 2.

## Final continuation verification

- Final score run 1: `68/1076 = 6.320%`, runtime `102.276` seconds.
- Final score run 2: `68/1076 = 6.320%`, runtime `102.332` seconds.
- Expected final task-normalized score: `65.000` after two newly solved one-output tasks since the documented 66-output baseline.
- Expected fully solved tasks: `64`; partially solved tasks remain `2`.
- New continuation hit: `5582e5ca`.
- Cumulative experiment hits over the original 66-output baseline: `4a1cacc2`, `5582e5ca`.
- Invalid outputs/exceptions in final score runs: none observed.

## Continuation merge recommendation

- Production score-positive: `fit_fill_marker_to_nearest_corner` from the previous commit and `fit_constant_most_frequent_color` from this continuation.
- Reusable lab infrastructure: `tools/mime_lab.py` is useful for residual clustering and direct primitive searches, but its first direct primitive pass found no new exact candidates.
- Documentation-only: `docs/MIME_EXPERIMENT.md` and `docs/MIME_LAB_REPORT.md` record the experiment evidence.
- Experimental not recommended for production: the unsuccessful singleton edge/corner variants and padded bounding-box frame hypotheses remain lab-only.

# Focused continuation from 68/1076

## Starting state

- Branch: `codex/mime-composition-experiment`.
- Starting commit for this focused continuation: `008aa358636606048806eebb72583957a70051ca`.
- Starting score reproduced before edits: `68/1076 = 6.320%`, runtime `105.497` seconds.

## Task investigations

### `3a301edc`

Forced facts: same-size output; existing nested rectangles are preserved; only background cells are recolored to the inner rectangle color; added cells form an outer shell centered on the nested object. Existing bbox-frame rules are close but insufficient.

Hypotheses tested: fixed bbox padding; padding by inner-object height; centered shell clipping at grid boundary; asymmetric shell placement. These fit many demonstrations but leave an underdetermined horizontal expansion case, so no production rule was promoted.

Concrete blocker: the training pairs do not uniquely distinguish the unclipped even-inner-height / odd-outer-width horizontal expansion convention without an arbitrary tie choice.

### `4a1cacc2`

Already solved by `fit_fill_marker_to_nearest_corner`; targeted checks preserved this hit.

### `8dab14c2`

Forced facts: same-size output; two-color palette preserved; sparse paired recolor changes move a boundary locally; total foreground area is sometimes preserved and sometimes changes by two cells. Existing fill, crop, translation, and simple recolor rules are insufficient.

Hypotheses considered: global boundary shift; fill nearest gap; one-cell horizontal projection; local symmetry repair. None explained all train pairs without overfitting, so no production rule was promoted.

### `7acdf6d3`

Forced facts: same-size output; dominant background preserved; old sparse source-color cells are erased; new source-color cells appear as horizontal row gaps inside exactly one 8-connected group of the other foreground color.

Successful mechanism: `move_sparse_color_to_row_gaps` removes the less frequent foreground color, finds 8-connected groups of the other foreground color, fills horizontal same-row gaps in exactly one group whose gap count equals the removed source count, and rejects ambiguity.

Score impact: added `7acdf6d3`, moving `70/1076` to `71/1076` after `bda2d7a6` was also promoted.

### `bda2d7a6`

Forced facts: same-size output; whole-grid recolor; concentric rectangular shell geometry preserved; colors cycle by shell-depth order.

Successful mechanism: `recolor_nested_shell_cycle` peels uniform rectangular perimeters, records first shell-color occurrences from outside inward, and maps each color to the previous color in that cyclic shell order. Ambiguous non-uniform shells reject.

Score impact: added both `bda2d7a6` test outputs, moving `68/1076` to `70/1076`.

## Score progression in focused continuation

- `68/1076 = 6.320%`: starting score.
- `70/1076 = 6.506%`: promoted `recolor_nested_shell_cycle`, new task `bda2d7a6` with two exact outputs.
- `71/1076 = 6.599%`: promoted `move_sparse_color_to_row_gaps`, new task `7acdf6d3` with one exact output.

## Cleanup

Removed the previous generic MIME residual harness and broad lab report from the branch (`tools/mime_lab.py`, `docs/MIME_LAB_REPORT.md`) because this continuation is intentionally focused on concrete unresolved tasks rather than broad candidate-count searches.

## Final focused verification

- Final score run 1: `71/1076 = 6.599%`, runtime `101.929` seconds.
- Final score run 2: `71/1076 = 6.599%`, runtime `102.401` seconds.
- New focused-continuation hits: `bda2d7a6` test outputs 0 and 1, plus `7acdf6d3` test output 0.
- Lost hits: none observed in full-corpus scoring.
- Production score-positive commits in this focused continuation: nested shell cycle and sparse row-gap transfer.

# Leverage-point pass: role-based color/region ordering

## Missing capability

The highest-leverage missing capability appears to be role-based color/region ordering: the solver repeatedly needs to infer colors or regions by structural role, not by raw color number. Recent score-positive rules independently needed unique mode color, sparse-vs-target foreground roles, and ordered shell colors.

## Evidence

Existing mechanisms already contain pieces of the same concept: component size/rank recoloring, marker-color transfer, nested shell cycling, singleton marker fills, and mode-color constants. Residual blockers such as `3a301edc` and `8dab14c2` are not blocked by lack of pixel painting operations; they are blocked by missing conventions for which structural role controls the target region and how ambiguous roles should be rejected.

## Implementation change

Added small internal role helpers in `rules_special.py` for unique mode color, two-foreground sparse/target roles, first-unique ordered colors, and cyclic predecessor mappings. Refactored the existing promoted mode-color, sparse row-gap, and shell-cycle rules to use these helpers. No new production rule was added in this pass.

## Measurement

- Targeted checks preserved `7acdf6d3`, `bda2d7a6`, `5582e5ca`, and `4a1cacc2`.
- Full production score after refactor: `71/1076 = 6.599%`, runtime `102.669` seconds.
- Production score change: `+0`; capability leverage is cleaner reuse of role inference across three existing score-positive mechanisms.

## Next capability

Investigate object/region correspondence under ambiguous anchors next: it appears necessary for boundary relocation (`8dab14c2`) and unresolved shell placement (`3a301edc`).
