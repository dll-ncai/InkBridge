# FMD phase 1, track Ink: InkSight Small-p zero-shot screen and render-consistency filter (val1000, 3 Nov 2026)

Track Ink. Sub-agent model: <agent> Reason: new pseudo-label filter code and a pre-registered gate read, which are <agent> tasks under the 30 Oct model policy. Pre-registration: `exp/FMD/plans/FMD_PREREG_2026-11-03.md` (sha256 6ad98ab4c1f4..., re-verified). Harness: fmd_harness 1.0.1, unedited. Track code: `exp/FMD/code/fmd_ext_ink/` (tar `exp/FMD/code/fmd_ext_ink_v1.tar.gz`, sha256 9094258920986e21...). Facts only.

## 1. Inputs [reason: prereg 2.3 comparator definition]

- InkSight Small-p val1000 trajectories come from Staging. Array 24828934 (64 tasks COMPLETED) produced 1,000 lines with prompt 'Recognize and derender.' for all 1,000 units (no fallback prompt was used) and 0 empty predictions. Staging inputs are 256x256 crops at f=4, thickness 8 (the G1 'rendered' crop convention), passed through InkSight's own scale_and_pad to 224. Staging scoring job 24831109 used lane_score.py rows031 with uhwr_eval 0.3.1, protocol native. 8 units were thinned to fit 320 points.
- InkSight scoring cross-check (Staging direct-convention scorer): native AIoU 0.82675 vs lane_score 0.82680; strict 0.559 in both; direction 0.5267 in both.
- E0 ms8 EMA, DDIM-50, eta 0, val1000. Seed 0 is the existing `runs/e0/e0_v3_rendered_100k_bs512/eval/val1000_ms8_preds.npz`; its meta shows DDIM-50, eta 0, EMA, seed 0, subset 1000 / subset-seed 0, A30, and its unit order is identical to val1000_units.csv. Seeds 1 and 2 did not exist anywhere on CLUSTER-A: the `exp/E3b` s00/s01 files are shards with seed 0, and `exp/e0q` has test only. They were sampled per prereg 2.3 by job 24829965 (gpuidle, <node>, RTX A4000, 1 GPU, 20 min 57 s) with e0/code_4cd40f57/e0_sample.py unchanged, batch 256. All three seeds were re-scored with lane_score.py rows031 (job 24830474). The seed-0 rescore reproduces the old E0 eval rows: native AIoU 0.79815, strict 0.754.
- Seed-0 reproducibility check (re-sampled on the RTX A4000): bytes_identical = False. 46/1000 units have max |diff| <= 1e-3 vs the A30 file; max abs diff = 2.0. The canonical seed 0 is the existing A30 file. The repro file was not scored.
- nomL1 = lineHT d4 ms4 and nomL2 = Line T d4 ms4, val1000 seeds 0-2. Both come from the Loop rows031 (`exp/CAND/method_round/Loop/score_out/lineHT_d4|lineT_d4/rows`, 4090 sampling, rtl-threshold 0.5, below-threshold backbone). Sidecar checkpoint sha256: 873b6698... (nomL1) and 65847fbe... (nomL2), both equal to the freeze record.
- Other derendering / ink FMs on CLUSTER-A: none. `hf/` contains only InkSight-Small-p, and `exp/FMD/weights/` contains no ink model. `repos/PEN-Net` (ACCV 2022 trajectory-recovery code, CASIA) is present but is not in the prereg FM list and was not assessed.
- InkSight vision tokens: the SavedModel exposes one signature, serving_default (inputs image/encoded + input_text; outputs output_0 string and output_1 float), plus preproc_func and tpu_func (job 24829976). No vision-token output is exposed. Per prereg 5.2, the shape clause is therefore zero-shot AIoU vs E0, and no InkSight probe rows were written.

## 2. Zero-shot InkSight vs E0 and the nominees on val1000 [reason: prereg gate, section 5; ink family 4 order tests]

Per-unit seed means (InkSight 1 sample; E0, nomL1 and nomL2 3 seeds each). Paired writer-cluster bootstrap, B = 2000, rng 20261103, 31 writers. tau_cp uses the Loop convention (NaN -> 0 for units with >= 2 GT strokes).

| metric | InkSight | E0 | delta vs E0 [95% CI] | n | gate_role | verdict | delta vs nomL1 [CI] | delta vs nomL2 [CI] |
|---|---|---|---|---|---|---|---|---|
| aiou | 0.8268 | 0.7968 | 0.0300 [0.0246, 0.0355] | 1000 | shape_NI | pass | 0.0451 [0.0394, 0.0508] | 0.0494 [0.0435, 0.0553] |
| tau_cp | 0.2972 | 0.8378 | -0.5406 [-0.5964, -0.4825] | 635 | order_gate | fail | -0.4143 [-0.4795, -0.3434] | -0.4078 [-0.4714, -0.3358] |
| strict_order_acc | 0.5590 | 0.7460 | -0.1870 [-0.2110, -0.1628] | 1000 | order_gate | fail | -0.0627 [-0.0904, -0.0311] | -0.0693 [-0.0981, -0.0367] |
| penup_f1_v2 | 0.8334 | 0.8283 | 0.0050 [-0.0189, 0.0301] | 1000 | order_gate | fail | 0.1389 [0.1145, 0.1639] | 0.1352 [0.1084, 0.1625] |
| direction_acc | 0.5272 | 0.9620 | -0.4348 [-0.4733, -0.3974] | 992 | order_gate | fail | -0.3908 [-0.4314, -0.3532] | -0.3931 [-0.4348, -0.3564] |
| kendall_tau | 0.3388 | 0.8978 | -0.5591 [-0.6231, -0.4951] | 566 | info | info | -0.4894 [-0.5590, -0.4142] | -0.4994 [-0.5709, -0.4230] |
| ldtw | 0.8454 | 0.1349 | 0.7106 [0.6619, 0.7679] | 1000 | info | info | 0.6000 [0.5514, 0.6551] | 0.6019 [0.5534, 0.6569] |
| stroke_count_err | 0.0520 | -0.1283 | 0.1803 [0.1416, 0.2206] | 1000 | info | info | 0.1620 [0.1186, 0.2030] | 0.1900 [0.1384, 0.2421] |

Absolute means [writer-cluster CI]:

| arm | AIoU | tau_cp | strict | pen-up F1 v2 | direction |
|---|---|---|---|---|---|
| inksight_small_p | 0.8268 [0.8235, 0.8300] | 0.2972 [0.2337, 0.3621] | 0.5590 [0.5210, 0.6009] | 0.8334 [0.8179, 0.8489] | 0.5267 [0.4907, 0.5612] |
| e0_ms8 | 0.7968 [0.7910, 0.8025] | 0.8378 [0.8117, 0.8649] | 0.7460 [0.7171, 0.7751] | 0.8283 [0.8075, 0.8473] | 0.9621 [0.9534, 0.9703] |
| nomL1_lineHT_d4_ms4 | 0.7817 [0.7754, 0.7881] | 0.7115 [0.6735, 0.7459] | 0.6217 [0.5906, 0.6522] | 0.6945 [0.6728, 0.7146] | 0.9175 [0.9012, 0.9337] |
| nomL2_lineT_d4_ms4 | 0.7774 [0.7712, 0.7839] | 0.7051 [0.6676, 0.7393] | 0.6283 [0.5988, 0.6572] | 0.6982 [0.6742, 0.7214] | 0.9199 [0.9047, 0.9355] |

Gate read (screen_rows.family_verdicts on rows_ink.csv): inksight_small_p order_pass = [] (0 of 4 order tests with CI lo > 0), shape_NI = pass (AIoU CI lo +0.0246 > -0.01035), fm_pass = false. **Family ink: FAIL.** Multiplicity: 4 zero-shot order tests, no correction (prereg 4).

## 3. Render-consistency filter on val1000 [reason: prereg section 6 ink route; predictive-validity measures are exploratory, not pre-registered]

Filter: keep a unit if IoU(re-rendered pseudo-trajectory, binarised input) >= 0.80 (threshold pre-registered). The re-render uses uhwr_eval.render_image on the 256-px crop canvas at thickness 8 (= 2 px per 64 px), anti-aliased, then pixel < 128. The input crop is binarised with Otsu.
- Sanity check: GT ink x4 through the same path vs the input mask gives IoU min 0.9943, mean 0.9983, with 0 units < 0.99.
- InkSight IoU over the 1,000 units: mean 0.8521, median 0.8595, q10 0.8158, q90 0.8912, 0 zeros. Info variant iou64 (64-px canvas, thickness 2): mean 0.8116.
- Kept at 0.80: 937/1000 = 0.937 [0.922, 0.952].

| label (InkSight per unit) | n | positives | AUROC of IoU [CI] | AUROC of iou64 [CI] | precision kept@0.80 [CI] | base rate [CI] | kept - all [CI] | kept - rejected [CI] |
|---|---|---|---|---|---|---|---|---|
| strict = 1 (all units) | 1000 | 559 | 0.669 [0.635, 0.702] | 0.698 [0.668, 0.730] | 0.575 [0.535, 0.619] | 0.559 [0.521, 0.601] | 0.0162 [0.0071, 0.0262] | 0.258 [0.122, 0.388] |
| strict = 1 (>= 2 GT strokes) | 635 | 224 | 0.631 [0.586, 0.676] | 0.607 [0.563, 0.651] | 0.367 [0.320, 0.418] | 0.353 [0.308, 0.400] | 0.0138 [0.0034, 0.0251] | 0.204 [0.059, 0.326] |
| direction_acc = 1 | 993 | 383 | 0.535 [0.498, 0.572] | 0.644 [0.608, 0.684] | 0.382 [0.351, 0.412] | 0.386 [0.355, 0.417] | -0.0041 [-0.0123, 0.0034] | -0.068 [-0.196, 0.060] |
| strict = 1 and direction = 1 | 993 | 263 | 0.597 [0.550, 0.642] | 0.692 [0.647, 0.736] | 0.266 [0.235, 0.297] | 0.265 [0.235, 0.295] | 0.0010 [-0.0064, 0.0083] | 0.016 [-0.107, 0.135] |

Spearman correlation of IoU with InkSight per-unit metrics [CI]: tau_cp 0.125 [0.044, 0.217] (n 635); aiou 0.428 [0.368, 0.487] (n 1000); penup_f1_v2 0.268 [0.207, 0.326] (n 1000).

InkSight - E0 (seeds 0-2) on the kept units vs on all units, paired [CI]:

| metric | kept units: delta [CI] (n) | all units: delta [CI] (n) |
|---|---|---|
| aiou | 0.0338 [0.0286, 0.0392] (937) | 0.0300 [0.0246, 0.0355] (1000) |
| tau_cp | -0.5163 [-0.5692, -0.4567] (592) | -0.5406 [-0.5964, -0.4825] (635) |
| strict_order_acc | -0.1818 [-0.2097, -0.1533] (937) | -0.1870 [-0.2110, -0.1628] (1000) |
| penup_f1_v2 | 0.0057 [-0.0184, 0.0301] (937) | 0.0050 [-0.0189, 0.0301] (1000) |
| direction_acc | -0.4402 [-0.4803, -0.4030] (932) | -0.4348 [-0.4733, -0.3974] (992) |

Threshold sweep (info; kept fraction and InkSight label means among the kept units):

| thr | n kept | strict=1 | direction=1 | strict&direction | AIoU | tau_cp | pen-up F1 v2 |
|---|---|---|---|---|---|---|---|
| 0.5 | 997 | 0.561 | 0.386 | 0.265 | 0.829 | 0.298 | 0.836 |
| 0.55 | 996 | 0.561 | 0.386 | 0.265 | 0.829 | 0.298 | 0.836 |
| 0.6 | 993 | 0.563 | 0.387 | 0.266 | 0.830 | 0.298 | 0.838 |
| 0.65 | 992 | 0.564 | 0.387 | 0.266 | 0.830 | 0.299 | 0.838 |
| 0.7 | 989 | 0.564 | 0.385 | 0.266 | 0.830 | 0.300 | 0.838 |
| 0.75 | 979 | 0.568 | 0.386 | 0.267 | 0.831 | 0.308 | 0.839 |
| 0.8 | 937 | 0.575 | 0.382 | 0.266 | 0.833 | 0.321 | 0.842 |
| 0.85 | 634 | 0.645 | 0.391 | 0.297 | 0.840 | 0.377 | 0.864 |
| 0.9 | 50 | 0.920 | 0.660 | 0.600 | 0.856 | 0.286 | 0.948 |
| 0.95 | 0 | - | - | - | - | - | - |

Recorded threshold for the ink pseudo-label route (upti_pool20k / nust_uhwr), as pre-registered: IoU >= 0.80, using the re-render and binarisation rule above. The ink family gate is FAIL, so per prereg 6 the route is not run as a gated arm. Pool images are not drawn at the 2-px-per-64 convention, so their stroke width is not controlled by this rule; it was not measured here.

## 4. Failed or blocked attempts [reason: reporting rule]

- Job 24829997 (first E0 rescoring) FAILED in 1 s. Cause: SyntaxError in e0_prep_preds.py (keyword `n_units_maxdiff_le_1e-3`). The key was renamed to `n_units_maxdiff_le_1em3` and the job was resubmitted as 24830474 (COMPLETED).
- InkSight vision-token probe (T0-T4): not assessed, because the SavedModel does not expose tokens (section 1).
- The InkSight-on-224-render input named in prereg 2.4 was not run. Staging ran InkSight on the 256-px G1 crop convention (f=4, thickness 8), and InkSight's preprocessing resizes that crop to 224. This is declared as a deviation.

## 5. Outputs on CLUSTER-A

- `exp/FMD/screen/rows_ink.csv` (24 rows, schema v1, sha256 8b0b8477...), `exp/FMD/screen/verdicts_ink.json`
- `exp/FMD/screen/units/ink/` per-unit files (InkSight, E0, nomL1, nomL2 zero-shot; filter units/sweep/summary)
- `exp/FMD/runs/ink/` (summary CSV, meta JSONs, signatures, SHA256SUMS_ink_analysis.txt); `exp/FMD/runs/ink_e0_val1000/` (E0 seeds 1-2 and seed-0 repro preds, sidecars, rows031)
- `exp/FMD/PROVENANCE_FMD_ink.json`
