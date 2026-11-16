# FMD pre-registration: foundation-model screen, gate and guided distillation (uhwr, 3 Nov 2026)

Lane: FM distillation (FMD), plan artifact 8f07471e. Track: Prereg (sub-agent model <agent>; reason: design, pre-registration and new harness code are <agent> tasks under <author>'s 30 Oct model policy).
Status: written BEFORE any FM feature extraction, FM probe, FM zero-shot output or distillation run. The only computation before this document was frozen is the harness smoke test on 50 val1000 units with the nominee-encoder / E0 / reference baselines (Section 9). Amendments must be appended (Section 10) with a UTC time stamp before the read they affect; nothing above Section 10 is edited after the sha256 below is recorded.
Harness: CLUSTER-A `exp/FMD/code/fmd_harness_v1/` = fmd_harness 1.0.1 (tar `exp/FMD/code/fmd_harness_v1.0.1.tar.gz`, sha256 a5b69f7b0753b2c486b5f2a297111f1cfdbb68b0ac0c1f2bf9c358bf6fcf53e3; the smoke test ran 1.0.0, tar sha256 8700f704..., which differs only in the T0 margin constant of screen_rows.py, the version string, the added verify_staging.py, README.md and the jobs/ scripts).

## 1. Frozen inputs

| item | value |
|---|---|
| freeze record | CLUSTER-A `exp/CAND/prereg/FREEZE_DECISION_2026-11-03.md`, sha256 869d82e76d02b16501312428ce4043f646772caa1b15bc94937d5d85fe18326a |
| Lane-2 nominee (nomL2) | Line T d4 ms4, ckpt sha256 65847fbef0e649c8aa2ce8d3b86779666834efece888eeef7864c6528633baeb, code loop_code_944a7401 (tar sha256 944a74013d89...). On the VPS only (`runs/CAND/method_round/Loop/lineT_d4_temporal_r16_tconv_r16_tmlp_r16_s20261027/checkpoints/checkpoint_000004.pt`); NOT on CLUSTER-A at prereg time |
| Lane-1 nominee (nomL1) | lineHT d4 ms4 (freeze CORRECTION 00:59Z), ckpt sha256 873b669844f201d8858472a3c7b43fd0b4442b5b1c2bd24bc3fbec35cffe1aec, CLUSTER-A `exp/CAND/method_round/Loop/anchors/lineHT_d4/checkpoints/checkpoint_000004.pt`; code loop_code_944a7401; tier S init tierS_all_r32 ckpt30 (2221c089...) |
| backbone | LUO ms6 EMA, CLUSTER-A `exp/M5/runs/luo/checkpoints/checkpoint_000006.pt` |
| base model code | Nagamatsu Unet1D repo commit 7c5cf62392e36cdc9f86445b4acb16c1e9198c68 (CLUSTER-A `repos/nagamatsu`; tar `tmp/staging_2026-10-24/nagamatsu_repo_7c5cf623_nodata.tar.gz` sha256 88660d42...); adapter_code_b8f84563 (tar sha256 b8f84563...) |
| router / frame | ScriptCNN `exp/D3_ROUTE/router/script_clf.pt`, rtl-threshold 0.5, below-threshold backbone; frame mirror_x from c1_code_c7830637 (`uhwr_frames.py` -> `e0_frames.frame_cond`, 1-px shifted flip) |
| E0 | E0 ms8 EMA, CLUSTER-A `runs/e0/e0_v3_rendered_100k_bs512/checkpoints/checkpoint_000008.pt` |
| data | OUHD-L division v3.0 ligature units: `data/ouhdl_e0_v3/e0_val.npz` (4,349 units, sha256 25451031...), `e0_test.npz` (4,406, sha256 c09560ed...), `e0_train.npz` (33,615, sha256 2c4ddc48...); index CSVs `data/ouhdl_ligatures_v3/ligatures_v3_{val,test}.csv` (sha256 7f166fa0... / 3eaed4af...) |
| val1000 | rows `sorted(np.random.default_rng(0).choice(4349, 1000, replace=False))` of e0_val.npz = `hsr_sample.py --subset 1000 --subset-seed 0` used by every Loop daily read. Materialised as `exp/FMD/data/val1000_units.csv` (+ .meta.json; identity with a Loop preds file checked there) |
| renderer | `uhwr_eval.render_image` (uhwr_eval 0.3.1, CLUSTER-A `shared/uhwr_eval-0.3.1`), called by `e0_adapter.render_condition` (e0/code_2be9e593, sha256 83420e98...) at 64 px, thickness 2, anti-aliased, black on white |

ImageEncoder (models/image_encoder.py, hidden_dims (64,128,256,512)) on the 64-px input returns five maps: 64 ch @64x64 (init conv), 64 @32, 128 @16, 256 @8, 512 @4. All five feed cross-attention: down blocks 0-3 use maps 0-3, the middle block uses map 4, up blocks use maps 4,3,2,1 and the final attention uses map 0. In the nominees the four downsample 1x1 convs of the encoder carry tier S/U LoRA (pathway 'all'), so maps 1-4 depend on the route; map 0 is the frozen LUO conv.

## 2. Task translation (ligature level)

### 2.1 Render and grids
224-px render of each unit = the 64-px evaluation-canvas GT ink (fit_transform_unit, 56-px box, 4-px margin, y down, x not mirrored) scaled by 3.5 (= fit_transform_unit at 224 with a 14-px margin), drawn with the same renderer at thickness 7 (2 px per 64 px x 3.5), anti-aliased, black on white (`render.py`). Check: re-rendering the stored ink at 64 px must reproduce `cond_rendered` byte for byte (count in the render meta file).
Probe grid G = the FM's native token grid if it is 14x14 (patch 16 at 224) or 16x16 (patch 14 at 224); any other native grid is mapped to 16x16. Grid mapping rule (features only): `F.interpolate(mode='bilinear', align_corners=False, antialias=True)`. Targets are computed directly on G from the trajectory (no resampling of targets).

### 2.2 Probe targets (per patch; `targets.py`)
| id | target | label rule | per-unit metric | role |
|---|---|---|---|---|
| T0 | ink occupancy (shape control) | 1 if >= 5 % of the patch's pixels are inked in the binary thickness-7 mask | IoU of predicted vs true ink patches (logit > 0) | shape non-inferiority |
| T1 | signed local stroke direction, 8 bins | vector sum V of GT segments (0.875-px resampling) whose midpoint lies in the patch; labelled if length L >= 0.5 patch and abs(V) >= 0.25 L; bin = round(atan2(Vy,Vx)/45 deg) mod 8 (x right, y down) | 8-way accuracy over labelled patches | order |
| T2 | stroke start-point map | 1 on the patch of each stroke's first point | R-precision (K = number of true start patches) | order |
| T3 | pen-up / stroke end-point map | 1 on the patch of each stroke's last point | R-precision | order |
| T4 | normalised time rank | mean of k/(N-1) over the resampled points (all strokes, writing order) in the patch; labelled if >= 2 points | Spearman rho, predicted vs true, over labelled patches (>= 3) | order |
| info | T1 orientation (mod 180 deg) accuracy; T23 sep_auc = AUC of (logit_T2 - logit_T3) separating start-only from end-only patches | | | reported only |
Patch membership: column j = floor((x + 0.5)/ps) (cv2 pixel-centre convention), same for rows.

### 2.3 Zero-shot tasks
Ink models (InkSight Small-p): full trajectory per unit from the 224-px render, mapped to the 64-px canvas and scored with uhwr_eval 0.3.1 exactly as the Loop reads (lane_score.py rows031, protocol 'native'): AIoU, tau_cp (Kendall tau for units with >= 2 GT strokes, NaN otherwise; Loop convention fillna(0) for failed predictions), strict order accuracy, pen-up F1 v2, direction accuracy. Comparator: E0 ms8 EMA, DDIM-50 eta 0, val1000 seeds 0-2 (existing preds if their sidecar shows these settings, otherwise sampled on CLUSTER-A and recorded).
VLM multiple-choice tasks (labels in `targets.py`; options in this fixed order; greedy decoding, max 8 new tokens; the first A/B/C letter in the answer is parsed, anything else scores 0):
- V1 first body stroke direction. Label RTL / LTR / OTHER from dx of the earliest body stroke (threshold 5.6 px of the 64 canvas). Image: 224 render. Prompt: "This image shows one handwritten Urdu ligature. In which direction did the writer move the pen while drawing the first main (body) stroke? Answer with one letter: A) right to left, B) left to right, C) neither (mostly vertical or a closed loop)."
- V2 which highlighted stroke was written first (units with >= 2 strokes). Pair and colours drawn by `default_rng([20261103, row])`; image `render.render_highlight224` (red (255,0,0) and blue (0,0,255) over black ink). Prompt: "This image shows a handwritten Urdu ligature. One pen stroke is drawn in red and another in blue. Which coloured stroke was written first? Answer with one letter: A) red, B) blue."
- V3 marks after body (units with >= 1 body and >= 1 mark stroke): 'yes' if every mark stroke follows the last body stroke. Prompt: "This image shows a handwritten Urdu ligature with a main body and one or more small marks (dots or diacritics). Were all the marks written after the main body was finished? Answer with one letter: A) yes, B) no."
- V4 transcript CER where `aligned_text` is non-empty. Prompt: "Transcribe the Urdu text in this image. Output only the Urdu characters." (max 32 tokens). Reported only. Expected weakness: a ligature is 1-3 characters cut from a word, so CER is dominated by segmentation and has no word context.
Per-unit score = correctness (0/1). Comparator for V1-V3 = the majority label over the scored units (`stats.majority_units`).

### 2.4 FM list and pre-named layers (repository ids as staged by the Staging track)
| family | FM | input / native grid -> G | layers extracted | gate layer |
|---|---|---|---|---|
| encoder | DINOv2 ViT-L/14 `facebook/dinov2-large` | 224, 16x16 -> 16 | last = last_hidden_state patch tokens (final norm); mid = output of block 12 of 24 | last |
| encoder | DINOv3 ViT-L/16 `facebook/dinov3-vitl16-pretrain-lvd1689m` | 224, 14x14 -> 14 (CLS/register tokens dropped) | last (final norm); mid = block 12 of 24 | last |
| encoder | SigLIP2 so400m `google/siglip2-so400m-patch14-224` (the so400m patch16-224 repo does not exist); fallback `google/siglip2-base-patch16-224` (G 14) only if so400m is blocked | 224, 16x16 -> 16 | last = last_hidden_state (post-LN); mid = hidden_states[13] of 27 (base: 6 of 12) | last |
| encoder | SAM2.1 Hiera-B+ `facebook/sam2.1-hiera-base-plus` | render resized to 1024 by the SAM2 processor; stride-16 64x64 -> 16 | last = image-encoder neck output at stride 16 (256 ch); mid = Hiera trunk stage-3 output | last |
| VLM | Qwen3-VL-8B-Instruct `Qwen/Qwen3-VL-8B-Instruct`; fallback `Qwen/Qwen3.5-4B` | vision tower at 224, patch 16 -> 14x14 pre-merger -> 14 | last = final vision block output before the 2x2 merger; mid = block floor(depth/2) | last; zero-shot V1-V4 on the full model |
| diffusion | SD 2.1 `sd2-community/stable-diffusion-2-1` (stabilityai repo returns 404), DIFT-style | render resized (bilinear) to 768, RGB, VAE encode, noise at t, UNet with the empty-prompt embedding, 4 noise draws averaged (seed 0); up-block output -> 16 | grid t in {101, 261} x up_block in {1, 2} (4 cells) | t = 261, up_block 1 (DIFT default) |
| ink | InkSight Small-p (CLUSTER-A `repos/inksight`) | 224 render with InkSight's own preprocessing | zero-shot trajectories (2.3); vision-encoder last/mid tokens if the exported model exposes them | last (if exposed) |
Rows for non-gate layers/cells are written with gate_role 'info'.

### 2.5 Baselines (`extract_api.py`)
nomL1 and nomL2 ImageEncoder tokens computed exactly as at sampling: 64-px `cond_rendered` input, router p_rtl >= 0.5 -> RTL route (S+U tiers active, mirror_x frame on the input), else route (0,0) = plain backbone; maps of framed units flipped back along W (plain flip; alignment error <= 1 px of 64). Layers: `mid16` = map 2 (128 ch, 16x16) and `cat16` = all five maps resampled to 16x16 and concatenated (1,024 ch); for G = 14 both are mapped 16 -> 14 by the grid rule. E0 ms8 EMA encoder: the same two layers, no router/frame. References (reported, never gating): `pos` (position-only probe) and `px` (raw pixel patches).
"Best nominee-encoder probe" for a given probe type, grid, target and metric = the arm with the highest mean over the units paired with the FM among {nomL1, nomL2} x {mid16, cat16}. Choosing the maximum favours the baseline.

## 3. Splits and probes
- Screening set: val1000 only. 5 writer-disjoint folds inside val1000 (`common.writer_folds`: writers ordered by unit count, ties by a default_rng(20261103) permutation, each assigned to the fold with the fewest units). Each unit is scored by the probe trained on the other four folds. Urdu test, gold and the six test sets are not touched by any FMD step before phase 5.
- Probe input per token: features z-scored with training-fold statistics, concatenated with the patch-centre (x, y) in [-1, 1].
- Linear probe: Linear(D+2, 12) (heads T0 1 | T1 8 | T2 1 | T3 1 | T4 1 with sigmoid), loss BCE + CE + BCE + BCE + MSE (labelled tokens only for T1/T4), AdamW lr 1e-3, weight decay 1e-4, 20 epochs, token batches 4,096.
- Attentive probe: Linear(D+2, 128) + fixed 2-D sin-cos embedding -> one pre-norm TransformerEncoderLayer (d 128, 4 heads, ff 256, dropout 0) -> LayerNorm -> Linear(128, 12); AdamW lr 3e-4, wd 1e-4, 20 epochs, 32 units per batch.
- Seeds 0, 1, 2 for every probe and fold. Hyper-parameters are identical for every FM, layer and baseline; none are tuned.
- Gate rows use the LINEAR probe on the pre-named gate layer. Attentive-probe and non-gate-layer rows are reported with gate_role 'info'.

## 4. Statistics
Per-unit value = mean over probe seeds 0-2 (zero-shot: over sampling seeds where applicable). Contrast = paired per-unit difference FM - comparator over the units where both have a value. 95 % CI = percentile interval of B = 2,000 writer-cluster bootstrap resamples, `numpy.random.default_rng(20261103)`, algorithm identical to `loop_tables.boot_idx` (writers drawn with replacement, all their units enter) (`stats.py`). Rows go to `exp/FMD/screen/fm_screen_rows.csv` (schema v1, `screen_rows.py`). Multiplicity: no correction; per family the number of order tests on the gate layer is encoder 16 (4 FMs x 4 targets), diffusion 4, VLM 4 probe + 3 zero-shot, ink 4 zero-shot (+4 probe if tokens are exposed). These counts are reported with the verdicts.

## 5. Gate
Shape margin delta_T0 = 0.01035. Justification: T0 IoU is an overlap measure of the same kind as AIoU; 0.01035 is 1.5 x MDE80 of AIoU in the 3-vs-3 Urdu test design, the shape margin used by every method-round rule (prereg v2-v4: AIoU 0.01035, direction 0.01245, strict 0.01065, pen-up F1 0.0141).
An FM passes if, on its gate layer with the linear probe:
1. ORDER: at least one of T1 acc8, T2 rprec, T3 rprec, T4 spearman has CI lo of (FM - best nominee-encoder probe) > 0; or, for zero-shot outputs, at least one order task has CI lo > 0 against its comparator (V1, V2, V3 accuracy vs majority; InkSight tau_cp, strict order, pen-up F1 v2 or direction vs E0); AND
2. SHAPE: T0 iou CI lo of (FM - best nominee-encoder probe) > -0.01035. If an FM's vision tokens cannot be extracted (declared blocked), its shape clause is zero-shot AIoU vs E0 with CI lo > -0.01035.
A family passes if at least one of its FMs passes. Verdicts are computed by `screen_rows.family_verdicts`; 'INCOMPLETE' if a listed FM has no shape row.
Known property recorded in advance: T2/T3 can rise through better endpoint (skeleton) detection without order information; T23 sep_auc and T1 orientation accuracy are reported next to them for that reason.

## 6. Distillation routes and arm choice
| family | route (phase 3) |
|---|---|
| encoder, diffusion | REPA-style token alignment: student 16x16 encoder tokens -> 3-layer MLP projector (SiLU, hidden 2,048; discarded after training) -> negative cosine similarity to the frozen FM gate-layer tokens (grid-mapped to 16x16 by the 2.1 rule), precomputed on the training renders; weight lambda = 0.5 on top of the diffusion loss |
| ink | pseudo-trajectories: InkSight on unlabelled offline Urdu images, kept if the re-rendered pseudo-trajectory (project renderer) has IoU >= 0.80 with the binarised input; pooled with the labelled training units |
| VLM | if it passes through a vision-tower probe: REPA alignment to the vision-tower gate layer (as for encoders); if it passes only through V1-V3: auxiliary order-label heads on the student encoder (V1/V3 answers of the VLM on training renders, BCE, weight 0.1) |
At most 2 guided arms (B, C). Ranking among passing FMs: (1) number of order criteria passed on the gate layer (more first); (2) mean over the passed criteria of CI lo divided by the SD over units of the comparator's per-unit values (higher first); (3) T0 delta (higher first). Arm B = rank 1. Arm C = the best-ranked FM from a different family; if only one family passes, the second FM of that family; if only one FM passes, no arm C. If no FM passes, no gated guided arm is run; any arm run anyway is labelled "gate-failed, exploratory" and cannot be promoted. If the chosen FM cannot be run within the phase-3 compute, the next-ranked FM is taken and the skip is logged.

## 7. Phase-3/5 read (fixed now)
Arms: A = student with the new 224-px FM-sized encoder (16x16 tokens) and no FM loss (architecture control); B, C = A + FM guidance (Section 6). A, B and C share the 1-D U-Net initialisation, training data, schedule and seed; they differ only by the FM loss term. Inference uses no FM. Budget: 4,000 steps (the Loop budget: batch 64 x grad-accum 2, lr 2e-4, warm-up 200, EMA, seed 20261027, DDIM-50 eta 0 EMA for sampling). Exact architecture and data are logged by the distillation track in its decision record before arm A starts.
Read on val1000, sampling seeds 0-2, per-unit seed means, writer-cluster bootstrap (Section 4), uhwr_eval 0.3.1 protocol native, metrics AIoU, pen-up F1 v2, strict order, direction, tau_cp, LDTW.
Contrasts: B - A and C - A (primary); B, C and A each minus nomL1 and minus nomL2 (reported; nominee val1000 rows from the Loop day reads, 4090 sampling).
Verdict of a guided arm vs A (mirrors the Loop v4 rule):
- GUIDED-WIN: AIoU CI lo > -0.01035 AND pen-up F1 v2 CI lo > -0.0141 AND strict CI lo > -0.01065 AND direction CI lo > -0.01245 AND at least one of pen-up F1 v2, strict, direction, tau_cp has CI lo > 0, using 97.5 % percentile intervals for this last clause when two guided arms are read (Bonferroni over 2 arms).
- NI-ONLY: the four non-inferiority clauses hold, no order clause wins.
- FAIL: otherwise.
Vs each nominee, the v4 cell clauses are reported (order NI three clauses; AIoU lead CI lo > 0.01035).
Promotion: a GUIDED-WIN arm is promoted to the single exploratory test pass (phase 5). If two arms are GUIDED-WIN, the one with the larger min over the three order NI CI lo values is promoted. If none is GUIDED-WIN, the test pass runs arm B as "not promoted, exploratory". Test pass: test_ouhdl, seeds 0-2, arm A and the promoted arm, same contrasts as on val; labelled exploratory in every table.

## 8. State at prereg time (facts)
- nomL2 checkpoint (65847fbe...) is not on CLUSTER-A. nomL2 baseline rows need it pulled to `exp/FMD/weights/nominees/lineT_d4_ms4/checkpoint_000004.pt` (or `FMD_NOML2_CKPT`) with sha256 check. Until then the best-nominee comparator is chosen within nomL1 only, and rows say so in `baseline`.
- Unlabelled offline Urdu pools on CLUSTER-A: `data/nust_uhwr` (manifest 10,605 handwritten line images: train 8,483 / val 1,061 / test 1,061), `data/pucit_ohul` (manifest 7,402 handwritten line images), `data/upti2/images_upti2_2.zip` (UPTI 2.0 printed text lines, 400,000 train + 80,000 test 'low' PNGs). No 'Urdu News' pool was found under `data/`. All are line-level; ligature-level crops for the ink route need a segmentation step that does not exist yet.
- The Staging track has written `exp/FMD/data/renders_224/upti_pool20k.npz` (20,000 connected components of UPTI 2.0 printed lines, Otsu + 8-connected CCs, fitted into 196x196; not OUHD-L ligature units; unlabelled).
- OUHD-L ligature CSVs carry `aligned_text` (V4) and body/mark stroke lists (V1, V3); stroke counts match the e0_val GT for all 1,000 val1000 units.
- val1000 (materialised by the harness, Slurm 24828905): 1,000 units, 31 writers; unit list identical to the unit_ids of the Loop preds file `val1000_ouhdl__c1c2_composed_p01_mirror_x_ms1_thr0p5_4090__seed0_preds.npz`; unit-id list sha256 a07fd9ba213dcb4fed0145fb53d7d0eb8c88c22e8570967b97228b44b9977d0d. CV folds: 211/199/199/197/194 units, 7/6/6/6/6 writers. 64-px re-render check: 1,000/1,000 units byte-identical to `cond_rendered`.
- val1000 label balance (G16 targets file): V1 RTL 790 / LTR 103 / OTHER 107; V2 red 306 / blue 329 (365 units with one stroke); V3 yes 528 / no 54 (418 not applicable); V4 non-empty 1,000.

## 9. Harness smoke test
Recorded in `fmd_smoke_<jobid>.out` (artifact) and PROVENANCE_FMD_prereg.json: smoke50 = val1000[sorted(default_rng(20261103).choice(1000, 50))], CPU, nominee/E0/reference baselines only. Its rows go to `exp/FMD/screen/smoke/` and never to the screen rows file. Smoke run: Slurm 24828905, cpuidle <node>, 16 CPUs, harness 1.0.0, COMPLETED, wall 00:21:06 (incl. val1000 unit list, renders and targets), MaxRSS 2.0 GB; nomL2 extraction failed as BLOCKED (checkpoint absent); rows rebuilt with 1.0.1 by Slurm 24829074. Baseline val1000 caches and probes (nomL1, e0, pos, px; nomL2 if present): Slurm 24829075 (extract) and array 24829076 (12 probe tasks).

## 10. Amendments

A1 (2026-11-03T06:15:59Z; lead session <agent>; decision by <author> via ask_user after the gate read, gate files exp/FMD/gate/ PROVENANCE_FMD_gate.json): add arm C' = arm A + REPA-style token alignment to the frozen SD2.1 DIFT gate-cell tokens (t = 261, up_block 1, same extraction settings as the screen; projector, lambda 0.5 and precomputation exactly as Section 6). SD2.1 FAILED the gate (T0 iou CI lo -0.0127 < -0.01035; T1/T2/T3 order rows CI lo > 0). C' is therefore labelled "gate-failed, exploratory" in every table and log, is read with the same contrasts as a guided arm (C' - A primary; C' minus nomL1 / nomL2 reported), cannot be promoted, does not enter the phase-5 test pass, and does not count as a guided arm for the Bonferroni clause of Section 7: B (DINOv3 ViT-L/16, the only passing FM) is the only gated guided arm, so B's order clause uses 95 % intervals. No arm C exists (only one FM passed). Sections 1-9 are unchanged; pre-amendment file sha256 6ad98ab4c1f4455535afd468b9144df43bb26ed1aa6a091db79cfc669c896035.

A2 (2026-11-03T07:30:58Z; lead session <agent>; written after the arms launched and BEFORE any arm finished training or was sampled): the race rule placed training on different devices (arm A on the VPS RTX 4090; arms B and C' on CLUSTER-A gpuidle A4000/A30-MIG). Because DDIM sampling is not bitwise reproducible across GPU types and the nominee val1000 rows were sampled on the 4090, the PRIMARY phase-3/5 read uses val1000 (and test) preds sampled on the VPS RTX 4090 for every arm, from each arm's final EMA inference checkpoint (student_infer_ms4.pt), same sampler command and seeds 0-2. Preds sampled on the CLUSTER-A training device for B and C' are scored as a SECONDARY device check and reported, not used for verdicts. The training-device difference cannot be removed and is reported with every contrast. Test pass (Section 7): arms A and B are sampled on test_ouhdl seeds 0-2 on the 4090 regardless of the val verdict (Section 7 runs B on test either as promoted or as "not promoted, exploratory"; C' cannot be promoted); test contrast tables are built only after the val verdict is written.
