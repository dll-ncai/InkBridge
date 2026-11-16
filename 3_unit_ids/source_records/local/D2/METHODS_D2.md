# METHODS_D2 — shared cross-script benchmark pilot (geometric baseline)

Facts only. Run 16 Oct 2026 on CLUSTER-A (Slurm job 24418077, node <node>, partition epyc-128, 16 CPUs; 02:28:43–02:39:14 UTC). Container `$U/images/torch27.sif`, venv `$U/venvs/cpu` (freeze sha256 766a30b8…c758). Provenance: PROVENANCE_D2.json (code, inputs, parameters, commands, output sha256), PROVENANCE_d2_pilot_v1.json, PROVENANCE_d2_writer_maps_v1.json.

## Pilot set (derived dataset d2_pilot_v1)
Up to 500 test units per set; numpy `default_rng(2026)`, a fresh generator per set, `choice(n_eligible, size, replace=False)`, units kept in loader order. No unit was dropped as degenerate. Content sha256 (points, offsets, keys) 99ebadde…158f.

| count | CASIA zh | CASIA en | HPL Tamil | IAM-OnDB lines | UNIPEN-de sie words | OUHD-L v3 ligatures | OUHD-L lines v1c |
|---|---|---|---|---|---|---|---|
| eligible test units | 224559 | 3115 | 26826 | 3525 | 1593 | 4406 | 253 |
| sampled | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| writers in pool | 60 | 60 | 52 | 63 | 4 | 32 | 32 |
| writers in sample | 60 | 60 | 52 | 63 | 4 | 32 | 32 |

Splits and selection:
- CASIA zh: OLHWDB1.1 test, writers 1241–1300 (60 .pot files); labels that are one CJK unified ideograph (U+4E00–U+9FFF).
- CASIA en: the same 60 .pot files; labels A–Z, a–z (52 classes; 3,115 samples). Digits and punctuation are excluded.
- HPL Tamil: IWFHR 2006 test set (26,926 files, anonymised). Writers were recovered by sha256 of the parsed (x, y, t) point sequence against the full hpl-tamil-iso-char-online-1.0 set (derived dataset d2_writer_maps_v1): 26,826 matched to 52 writers, 0 ambiguous. The 100 unmatched files are not eligible.
- IAM-OnDB: t1 test split testset_f (544 forms, 3,859 lines; file extracted from the official task1.tar.gz, sha256 d41191d0…). Writer ids come from original-xml-part (1,561 forms); 3,525 lines from 63 writers have one, and the 334 lines without one are not eligible. **Word units are not available:** lineStrokes XML has no word segmentation, so only lines are used.
- UNIPEN-de: train_r01_v07 `sie` word subset, all 4 writers (gsaa, htca, hwaa, mgca; 1,593 words). train_r01_v07 has no test split, and the geometric baseline is not trained, so no writer is held out. The CIs for this set are bootstrapped over 4 writers.
- OUHD-L ligatures: v3.0 test split (4,406 units, 32 writers).
- OUHD-L lines: ouhdl_lines_v1c test split (253 lines, 32 writers). This is fewer than 500, so all lines are used.
- Orientation: UNIPEN sie y is negated (recorded y-up; checked visually). Tamil test ink is already y-down; checked against the dataset's offline TIFF of the same serial.

Pilot build host: the set was built on <author>'s Mac and uploaded, and the job verified its content hash (verify_pilot.py). An CLUSTER-A build in smoke job 24417418 was identical for 6 sets (index CSV rows and all input sha256, including the 60 .pot files). It differed for UNIPEN sie: CLUSTER-A `data/unipen_train_r01_v07/.../include/<unipen-path>` holds files only for writer htca (494), while `stage/unipen_de.tar` lists all 4 writers (d2_pilot_crosshost_check_D2.json).

## Protocol (identical for all sets)
- Evaluation canvas: isolated units (CASIA zh, CASIA en, HPL Tamil, OUHD-L ligatures) 64×64, via `uhwr_eval.fit_transform_unit` (isotropic, 56×56 box, 4 px margin). Line-canvas units (IAM lines, OUHD-L lines, UNIPEN words) height 64, via `fit_transform_line` (ink bbox height → 56 px, width = ceil(56·aspect)+8). UNIPEN words are put on the line canvas because they are multi-character units.
- Input image: `uhwr_eval.render_image` of the GT at 4× the evaluation canvas, stroke width 8 px (= 2 px per 64 px of canvas height), anti-aliased, black on white. Uniform width, no noise. This is the same renderer and scale as the OUHD-L v3 geometric runs (`rendered_unit` / `rendered_line`, f = 4).
- Method: `uhwr_baselines.geometric.recover` (skeleton graph walk; code unchanged, sha256 in provenance), with `unit_height` = 224 for the dot test (dot_frac 0.18). Dots: `last` for isolated units, `after_body` for line-canvas units. Writing direction: right-to-left for Urdu (rule unchanged). Left-to-right for Latin, Tamil **and Chinese**: the same rule is applied to the horizontally mirrored image and output x is mirrored back (x = W−1−x′). So bodies are visited left to right by leftmost x, each walk starts at the leftmost endpoint (ties → topmost), and the first edge heads rightward. No top-to-bottom or Chinese stroke-order rule is used. Predictions are mapped back to the evaluation canvas by 1/4.
- Scoring: `uhwr_eval.score(..., protocol='standardised', step=1.0)`, with both GT and prediction resampled per stroke at 1 px arc length. Every registered metric is computed: DTW, LDTW, AIoU (Nagamatsu metrics.py, sha256 7e2d3305…), LPIPS (AlexNet, lpips 0.1), Chamfer, Hausdorff, discrete Fréchet, RMSE (K = 320 for 64×64, 2,048 for line canvases), pen-up F1 (3 px), stroke-count errors, stroke match recall/precision (IoU ≥ 0.5), strict order, Kendall tau-b, direction accuracy, loop orientation. Native-protocol scores were not computed.
- InkSight is **not part of this run**.
- Aggregation: mean over units with a defined value (n_valid per cell in results_D2.csv). 95% CI: percentile bootstrap over writers (writers resampled with replacement, all their units kept), B = 2000, seed 0 (`uhwr_baselines.aggregate.boot_ci`). Writers per set: CASIA zh 60, CASIA en 60, Tamil 52, IAM 63, UNIPEN 4, OUHD-L ligatures 32, OUHD-L lines 32.
- Units scored: 3,253; scoring errors 0; empty predictions 0.

## Per-script properties (GT, evaluation space, 1 px resampled)
Definitions (unit_features.py):
- strokes per unit: number of pen-down strokes.
- delayed-stroke fraction: n_delayed / n_strokes. A stroke k ≥ 1 is delayed if the centroid of its writing-direction coordinate u (u = x for LTR, −x for RTL) is more than 2 px behind the frontier max u over all earlier strokes. Single-stroke units count 0.
- self-crossings: proper intersections between segments of the RDP-simplified (ε = 1 px) GT polylines. All pairs between strokes are counted, plus non-consecutive pairs within a stroke. Touching and collinear overlap are not counted.

Mean [95% CI over writers]:

| property | CASIA zh | CASIA en | HPL Tamil | IAM-OnDB lines | UNIPEN-de sie words | OUHD-L v3 ligatures | OUHD-L lines v1c |
|---|---|---|---|---|---|---|---|
| prop_crossings_between | 3.73 [3.24, 4.26] | 0.4 [0.293, 0.537] | 0.21 [0.156, 0.267] | 3.6 [3.29, 3.92] | 0.418 [0.218, 0.689] | 0.052 [0.0207, 0.0891] | 1.21 [0.805, 1.72] |
| prop_crossings_within | 2.68 [2.15, 3.27] | 0.392 [0.295, 0.5] | 1.61 [1.43, 1.82] | 7.21 [6.3, 8.24] | 3.68 [2.71, 4.61] | 0.342 [0.287, 0.4] | 5.65 [5.14, 6.2] |
| prop_delayed_frac | 0.497 [0.468, 0.525] | 0.0972 [0.0782, 0.116] | 0.15 [0.129, 0.172] | 0.0866 [0.0776, 0.0956] | 0.101 [0.0806, 0.144] | 0.307 [0.283, 0.331] | 0.486 [0.471, 0.5] |
| prop_n_delayed | 3.23 [2.91, 3.54] | 0.23 [0.182, 0.282] | 0.324 [0.276, 0.373] | 2.31 [2.02, 2.61] | 0.404 [0.327, 0.534] | 0.946 [0.834, 1.06] | 18.6 [17.5, 19.8] |
| prop_n_strokes | 5.63 [5.23, 6.02] | 1.45 [1.39, 1.52] | 1.52 [1.45, 1.58] | 25.8 [24, 27.4] | 3.16 [2.37, 3.74] | 2.09 [1.97, 2.23] | 37.8 [36.1, 39.6] |
| prop_self_crossings | 6.41 [5.72, 7.15] | 0.792 [0.641, 0.966] | 1.82 [1.63, 2.04] | 10.8 [9.86, 11.9] | 4.1 [3.06, 5.08] | 0.394 [0.33, 0.466] | 6.85 [6.12, 7.76] |

## Results — geometric baseline, standardised protocol, mean [95% CI over writers]
Strict order, direction and loop orientation are in %. All other metrics are in evaluation-canvas pixel units or unitless. The same table in long form, with SD, median and n_valid, is in results_D2.csv.

| metric | CASIA zh | CASIA en | HPL Tamil | IAM-OnDB lines | UNIPEN-de sie words | OUHD-L v3 ligatures | OUHD-L lines v1c |
|---|---|---|---|---|---|---|---|
| dtw | 263 [255, 271] | 219 [208, 231] | 235 [228, 244] | 385 [353, 423] | 418 [355, 488] | 139 [127, 150] | 438 [407, 473] |
| ldtw | 0.647 [0.626, 0.67] | 1.21 [1.12, 1.3] | 0.904 [0.866, 0.951] | 0.25 [0.237, 0.266] | 0.689 [0.607, 0.78] | 0.92 [0.836, 1.01] | 0.292 [0.267, 0.324] |
| aiou | 0.893 [0.891, 0.895] | 0.912 [0.91, 0.914] | 0.912 [0.91, 0.913] | 0.893 [0.891, 0.895] | 0.898 [0.892, 0.906] | 0.904 [0.901, 0.906] | 0.885 [0.882, 0.887] |
| lpips | 0.0111 [0.0104, 0.0119] | 0.00387 [0.00359, 0.00412] | 0.00446 [0.00403, 0.00489] | 0.0105 [0.0096, 0.0113] | 0.00997 [0.00786, 0.0119] | 0.00398 [0.00364, 0.00434] | 0.00927 [0.00864, 0.00997] |
| chamfer | 0.344 [0.338, 0.349] | 0.293 [0.289, 0.297] | 0.298 [0.294, 0.303] | 0.314 [0.309, 0.321] | 0.332 [0.315, 0.344] | 0.282 [0.278, 0.285] | 0.297 [0.293, 0.301] |
| hausdorff | 1.43 [1.38, 1.47] | 0.984 [0.954, 1.01] | 1.12 [1.08, 1.16] | 1.51 [1.47, 1.55] | 1.36 [1.28, 1.44] | 0.86 [0.828, 0.895] | 1.53 [1.48, 1.59] |
| frechet | 36 [35, 36.9] | 37.3 [35.7, 39] | 32 [31.1, 32.9] | 34.8 [32.4, 37.4] | 46.3 [39.2, 54.4] | 26.7 [24.5, 28.8] | 47.6 [41.8, 54.5] |
| rmse | 21.5 [21, 22] | 22.8 [21.7, 23.9] | 22.3 [21.6, 22.9] | 18.2 [17.1, 19.5] | 28.8 [24.6, 34.5] | 16 [14.7, 17.2] | 20 [18.9, 21.2] |
| penup_f1 | 0.259 [0.241, 0.28] | 0.209 [0.182, 0.237] | 0.199 [0.174, 0.225] | 0.351 [0.324, 0.376] | 0.171 [0.149, 0.194] | 0.557 [0.519, 0.593] | 0.522 [0.498, 0.545] |
| stroke_count_abs_err | 8.03 [7.53, 8.53] | 1.62 [1.49, 1.78] | 2.63 [2.4, 2.88] | 18.3 [16, 21] | 7.81 [6.78, 8.46] | 1.01 [0.908, 1.11] | 16.5 [15.3, 17.7] |
| stroke_count_err | 8.02 [7.53, 8.52] | 1.61 [1.48, 1.77] | 2.62 [2.39, 2.88] | 18.3 [16, 21] | 7.81 [6.78, 8.46] | 0.996 [0.891, 1.1] | 16.3 [15.1, 17.5] |
| stroke_match_recall | 0.644 [0.617, 0.671] | 0.828 [0.8, 0.855] | 0.759 [0.716, 0.795] | 0.818 [0.786, 0.849] | 0.54 [0.437, 0.64] | 0.93 [0.911, 0.948] | 0.895 [0.886, 0.905] |
| stroke_match_precision | 0.303 [0.279, 0.329] | 0.505 [0.475, 0.533] | 0.363 [0.325, 0.401] | 0.511 [0.466, 0.554] | 0.199 [0.144, 0.259] | 0.731 [0.704, 0.757] | 0.626 [0.611, 0.642] |
| strict_order_acc | 0.4 [0, 1.02] | 19.6 [16.2, 22.9] | 8 [5.12, 11.2] | 0 [0, 0] | 0 [0, 0] | 38.2 [34.2, 42.1] | 0 [0, 0] |
| kendall_tau | 0.423 [0.354, 0.491] | 0.284 [0.13, 0.439] | 0.802 [0.713, 0.888] | 0.97 [0.964, 0.975] | 0.832 [0.685, 0.895] | 0.452 [0.358, 0.547] | 0.963 [0.958, 0.967] |
| direction_acc | 56.4 [53.7, 59.2] | 53.9 [49.4, 58.5] | 49.4 [45.3, 53.4] | 56.5 [53.7, 59.1] | 56.8 [49.9, 62.1] | 64.1 [59.8, 68.4] | 62.2 [59.5, 64.7] |
| loop_orient_acc | 54.6 [52.3, 57.2] | 55 [50.4, 59.5] | 47.6 [43.6, 51.5] | 57.2 [54.2, 59.8] | 58 [50.6, 61.1] | 62.3 [57.7, 67.1] | 63.2 [60.6, 65.9] |
| n_gt_strokes | 5.63 [5.23, 6.02] | 1.45 [1.39, 1.52] | 1.52 [1.45, 1.58] | 25.8 [24, 27.4] | 3.16 [2.37, 3.74] | 2.09 [1.97, 2.23] | 37.8 [36.1, 39.6] |
| n_pred_strokes | 13.7 [13.1, 14.2] | 3.07 [2.93, 3.22] | 4.14 [3.9, 4.4] | 44.1 [42, 46.5] | 11 [10.3, 11.6] | 3.09 [2.93, 3.26] | 54.1 [52.4, 55.8] |
| n_gt_pts | 316 [306, 326] | 153 [149, 157] | 201 [194, 210] | 1.23e+03 [1.17e+03, 1.29e+03] | 454 [446, 463] | 120 [118, 122] | 1.27e+03 [1.22e+03, 1.31e+03] |
| n_pred_pts | 288 [281, 296] | 149 [145, 152] | 198 [191, 205] | 1.16e+03 [1.11e+03, 1.22e+03] | 427 [415, 443] | 118 [115, 120] | 1.2e+03 [1.16e+03, 1.25e+03] |
| empty_pred | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| n_dir_eligible | 3.76 [3.44, 4.09] | 1.19 [1.13, 1.25] | 1.14 [1.05, 1.21] | 20 [18.2, 21.6] | 1.99 [1.18, 2.67] | 1.52 [1.43, 1.61] | 26.6 [25.6, 27.9] |
| n_loop_eligible | 3.13 [2.85, 3.41] | 1.07 [1, 1.13] | 1 [0.916, 1.09] | 15.6 [14.1, 17.1] | 1.49 [0.876, 2.13] | 1.21 [1.16, 1.28] | 20.8 [20, 21.8] |

n_valid per cell (units with a defined value):

| metric | CASIA zh | CASIA en | HPL Tamil | IAM-OnDB lines | UNIPEN-de sie words | OUHD-L v3 ligatures | OUHD-L lines v1c |
|---|---|---|---|---|---|---|---|
| dtw | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| ldtw | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| aiou | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| lpips | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| chamfer | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| hausdorff | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| frechet | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| rmse | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| penup_f1 | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| stroke_count_abs_err | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| stroke_count_err | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| stroke_match_recall | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| stroke_match_precision | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| strict_order_acc | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| kendall_tau | 408 | 135 | 148 | 499 | 264 | 264 | 253 |
| direction_acc | 473 | 437 | 417 | 500 | 367 | 479 | 253 |
| loop_orient_acc | 450 | 404 | 373 | 498 | 341 | 448 | 253 |
| n_gt_strokes | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| n_pred_strokes | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| n_gt_pts | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| n_pred_pts | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| empty_pred | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| n_dir_eligible | 500 | 500 | 500 | 500 | 500 | 500 | 253 |
| n_loop_eligible | 500 | 500 | 500 | 500 | 500 | 500 | 253 |

Kendall tau needs ≥ 2 matched strokes. Direction accuracy needs ≥ 1 matched stroke with GT start–end distance ≥ 2 px. Loop orientation needs ≥ 1 matched curved stroke.

## Reproduction checks
- Local re-run of the full D2 computation (Mac, 12 workers, 25 s) against the CLUSTER-A run, over all 3,253 units: every metric is identical (max |Δ| LPIPS 8.2e-8, others ≤ 3e-14), except loop orientation on 4 units. Each of these has a matched predicted stroke with signed area at or near 0, where the sign depends on float rounding.
- Code sha256 on CLUSTER-A matches the local files for all 8 D2 scripts. Package versions differ between the hosts (read from PROVENANCE_A2.json package_versions and a version probe in conda env uhwr-eval): CLUSTER-A numpy 2.2.6, scikit-image 0.26.0, opencv 5.0.0, scikit-learn 1.9.1; Mac numpy 1.26.4, scikit-image 0.20.0, opencv 5.0.0, scikit-learn 1.9.1.

## Jobs
- Smoke 1: 24416935 (<node>), failed after 5 s: `data/casia/read_pot.py` is not extracted on CLUSTER-A (only inside stage/loaders_casia.tar).
- Smoke 2 (fix-retry): 24417418 (<node>), TIMEOUT at 60 min. The fix extracted read_pot.py from loaders_casia.tar and testset_f.txt from official task1.tar.gz into exp/D2/ext/. The CLUSTER-A pilot build took 2,916 s (HPL Tamil: 26,926 small files on BeeGFS). D2 on 10 units per set completed (70 units, 0 errors). The A2 smoke step did not complete.
- Full: 24418077 (<node>), completed.

## Deviations
- UNIPEN-de: no test split exists, so all 4 sie writers are used, and the CI rests on 4 writers.
- HPL Tamil: 100 of 26,926 test units (no recoverable writer) are not eligible.
- IAM-OnDB: 334 of 3,859 test lines (no writer id) are not eligible; words are not available.
- OUHD-L lines: 253 < 500, so all are used.
- CASIA en: 3,115 letter samples in the pool (52 classes × 60 writers would be 3,120).
- The pilot set was built locally and uploaded, not built on CLUSTER-A, because CLUSTER-A's extracted UNIPEN copy is partial.
- InkSight was not run. Native-protocol scores were not computed.
- The OUHD-L numbers here are the D2 500-unit / 253-line pilot sample under the D2 protocol only. They are not the full v3 re-score.
