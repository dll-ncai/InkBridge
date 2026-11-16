# RESULTS_FMD — foundation-model distillation lane (uhwr, 3 Nov 2026)

Facts only. Level: OUHD-L v3.0 ligature units. Lead session <agent>; each sub-agent's model and reason are in its provenance file.
Plan: {{artifact:8f07471e-e758-404f-bb64-19f9a10508d4}}. Pre-registration: original {{artifact:72376b90-027b-4c64-9196-383919665e68}} (sha256 6ad98ab4…); amended copy with A1 + A2 {{artifact:1a1c02d0-6ec5-4597-9adc-c65cbd9e6438}} (sha256 c5bb785c…). CLUSTER-A: $U/exp/FMD/.

## 1. Acquisition — FM weights and data staging
Reason tag: [acquisition] inputs for the FM screen.
- Weights on CLUSTER-A with pinned HF revisions and per-file sha256: {{artifact:8eb431b2-d4df-4879-adc4-c78a229805a2}}. DINOv2-L/14 @47b73eef; DINOv3 ViT-L/16 @ea8dc286 (gated; fetched via an HF_TOKEN present in the CLUSTER-A account environment; licence restricts redistribution); SigLIP2 so400m/14 @78e40396 (requested so400m/16 repo returns 404); SAM2.1 Hiera-B+ @b7320756; Qwen3-VL-8B-Instruct @0c351dd0; Qwen3.5-4B @851bf6e8 (extra); SD2.1 sd2-community @bb215482 (stabilityai repo returns 404); InkSight Small-p @8514f237 (existing, sha verified).
- Data: val1000 = sorted(default_rng(0).choice(4349,1000)) rows of e0_val.npz, 1,000 units, 31 writers; 224-px renders (f=3.5, thickness 7) for val1000 and the 4,406 test units; 64-px re-render byte-identical to cond_rendered on 1,000/1,000 val1000 units; UPTI pool of 20,000 printed connected components.
- Lane-2 nominee checkpoint pulled from the VPS (sha256 65847fbe… on both ends). Provenance: {{artifact:e07702e6-907e-4338-b73d-0770368c81b7}}.

## 2. Pre-registration and harness
Reason tag: [method] fix tasks, gate and read before any FM output.
- Targets on the 16x16 / 14x14 grid: T0 ink IoU (shape), T1 signed direction acc8, T2 start-point rprec, T3 pen-up rprec, T4 time-rank Spearman; zero-shot V1 writing direction, V2 first component, V3 marks after body, V4 transcript CER; InkSight trajectory vs E0. Gate: linear probe on the pre-named layer, ≥1 order criterion with 95% CI lo > 0 vs the best nominee-encoder probe AND T0 CI lo > −0.01035; writer-cluster bootstrap B=2000, rng 20261103; no multiplicity correction.
- Harness fmd_harness 1.0.1 {{artifact:da647b61-2b8a-4df0-ae6b-3873064058da}}; smoke test 50 units, 1,265 s CPU. Nominee-encoder baseline on val1000 {{artifact:10a7f63b-7bfc-45ff-b545-6f8d02cae2d3}}: nomL1 cat16 linear G16 T0 0.960, T1 0.819, T2 0.514, T3 0.633, T4 0.790; raw-pixel probe T0 0.989; position-only probe T4 0.534.

## 3. FM screen (val1000) and gate read
Reason tag: [experiment] screen. Full table {{artifact:31f0a92e-0ef1-4aa5-ae0d-35bc068b7e5a}}, rows {{artifact:479088de-7ffa-4918-a21e-ef8403d18ff5}}, provenance {{artifact:4f9f9cc9-1726-47f8-b2f4-e4c172aec49c}}.

| family | FM | T0 shape CI lo | order criteria with CI lo > 0 | verdict |
|---|---|---|---|---|
| encoder | dinov2-large | -0.0949 | T2:rprec | FAIL |
| encoder | dinov3-vitl16 | -0.0003 | T2:rprec | PASS |
| ink | inksight_small_p | +0.0246 | none | FAIL |
| VLM | qwen3-vl-8b-instruct | -0.1801 | none | FAIL |
| encoder | sam2.1-hiera-base-plus | -0.0200 | T2:rprec | FAIL |
| diffusion | sd2.1-dift | -0.0127 | T1:acc8;T2:rprec;T3:rprec | FAIL |
| encoder | siglip2-so400m-patch14-224 | -0.1607 | T2:rprec;T3:rprec | FAIL |

- DINOv3 gate rows: T2 rprec +0.0453 [+0.0099, +0.0808]; T0 +0.0038 [−0.0003, +0.0073]; T1 −0.0653, T3 +0.0030 [−0.0165, +0.0229], T4 −0.0167 (CI hi < 0). Info rows: T23 sep_auc +0.0018 [−0.0166, +0.0191]; T1 acc_orient4 −0.0746 [−0.0810, −0.0686].
- SD2.1 DIFT t261/up1: T1 +0.0169 [+0.0092, +0.0240], T2 +0.1188 [+0.0922, +0.1442], T3 +0.0935 [+0.0731, +0.1141], T4 −0.0193 [−0.0287, −0.0092], T0 −0.0102 [−0.0127, −0.0077].
- Qwen3-VL-8B zero-shot: V1 0.156 vs majority 0.790 (920/1,000 answers LTR); V2 0.501 vs 0.518; V3 0.093 vs 0.907 (582/582 'no'); V4 CER 3.726. Vision tower 'last' T0 −0.1757. Info only (mid layer, block 13): T2 +0.128 [0.098, 0.157], T3 +0.097 [0.077, 0.117].
- InkSight vs E0 (val1000): AIoU +0.0300 [0.0246, 0.0355], τ_cp −0.541, strict −0.187, pen-up F1 v2 +0.005 [−0.019, 0.030], direction −0.435. Render-consistency filter at IoU ≥ 0.80 keeps 937/1,000 units; AUROC for strict order 0.669 [0.635, 0.702] (exploratory).
- Family verdicts: encoder PASS (DINOv3 only), diffusion FAIL, VLM FAIL, ink FAIL. Arm choice per prereg §6: B = DINOv3 REPA; no arm C. Amendment A1 (<author>'s decision): arm C′ = SD2.1 REPA, labelled gate-failed, exploratory, never promoted.

## 4. Student build
Reason tag: [method] FM-sized encoder student, FM-free at inference. Decision record {{artifact:e52e61db-b9c9-4b81-bbe5-47b093df25c9}} (frozen 06:48:14Z before any arm), provenance {{artifact:8022d7ef-71de-4371-8597-09124beb64e0}}.
- 224-px ViT-S/14 encoder (16x16 tokens, 12 blocks, width 384) + per-level heads rebuilding 64@64, 64@32, 128@16, 256@8, 512@4; new encoder 25,455,616 params (old 64-px ImageEncoder 3,955,264); student 48,592,388 params; trainable 27,970,624 in every arm (encoder + one fresh LoRA tier).
- Non-encoder weights = Line T d4 ms4 RTL route merged (max |diff| vs routed model 2.2e-6). Data m5_ft_ouhdl_p01 (336 units, 183 writers), mirror_x frame; 4,000 steps, batch 64x2, lr 2e-4, warm-up 200, EMA, seed 20261027. REPA: 3-layer MLP projector (hidden 2,048, discarded), −cos to frozen FM tokens, λ 0.5, targets precomputed on the framed training renders.
- Training lanes (race rule, cand_log.md): A VPS 4090 (cmd 361); B CLUSTER-A gpuidle A4000 (24834408); C′ CLUSTER-A gpuidle (24835426). Float64 check of each inference checkpoint pass (max |diff| 8.2e-8 / 2.7e-7 / 6.8e-8); no projector or FM tensors in any inference checkpoint.

## 5. Distillation read — val1000 (primary, 4090 sampling, seeds 0–2)
Reason tag: [experiment] prereg §7 + A2. Tables {{artifact:87f71842-fadf-4921-898c-e6fca76cbd08}} / {{artifact:2582c963-0a5b-41b2-9fcc-1bb96910b87b}}, absolute {{artifact:28265428-733e-42cf-9a33-6121ab991f32}}, provenance {{artifact:18155951-1bc5-4ff3-ba0d-2c2c21849761}}.

| cell | n units | aiou | penup_f1_v2 | strict_order_acc | direction_acc | tau_cp | ldtw |
|---|---|---|---|---|---|---|---|
| A_4090 | 1000 | 0.5457 | 0.2155 | 0.3060 | 0.9161 | 0.2283 | 0.5097 |
| B_4090 | 1000 | 0.5733 | 0.2855 | 0.4060 | 0.9207 | 0.3601 | 0.4230 |
| Cp_4090 | 1000 | 0.6429 | 0.4078 | 0.4670 | 0.9161 | 0.5036 | 0.3503 |
| nomL1 | 1000 | 0.7817 | 0.6945 | 0.6217 | 0.9175 | 0.7115 | 0.2455 |
| nomL2 | 1000 | 0.7774 | 0.6982 | 0.6283 | 0.9199 | 0.7051 | 0.2436 |

| contrast | aiou | penup_f1_v2 | strict_order_acc | direction_acc | tau_cp | ldtw |
|---|---|---|---|---|---|---|
| B_4090 - A_4090 | +0.0276 [+0.0231, +0.0320] | +0.0701 [+0.0545, +0.0848] | +0.1000 [+0.0829, +0.1155] | +0.0015 [-0.0086, +0.0117] | +0.1318 [+0.1049, +0.1592] | -0.0867 [-0.1156, -0.0605] |
| Cp_4090 - A_4090 | +0.0972 [+0.0918, +0.1027] | +0.1924 [+0.1755, +0.2076] | +0.1610 [+0.1424, +0.1790] | +0.0028 [-0.0086, +0.0136] | +0.2752 [+0.2312, +0.3156] | -0.1594 [-0.1836, -0.1357] |
| A_4090 - nomL1 | -0.2360 [-0.2442, -0.2279] | -0.4791 [-0.5022, -0.4564] | -0.3157 [-0.3465, -0.2859] | -0.0048 [-0.0193, +0.0102] | -0.4831 [-0.5255, -0.4424] | +0.2642 [+0.2354, +0.2940] |
| B_4090 - nomL1 | -0.2084 [-0.2141, -0.2025] | -0.4090 [-0.4295, -0.3879] | -0.2157 [-0.2427, -0.1913] | +0.0012 [-0.0151, +0.0175] | -0.3514 [-0.3841, -0.3180] | +0.1776 [+0.1548, +0.2024] |
| Cp_4090 - nomL1 | -0.1388 [-0.1447, -0.1329] | -0.2867 [-0.3070, -0.2660] | -0.1547 [-0.1818, -0.1299] | -0.0023 [-0.0152, +0.0107] | -0.2079 [-0.2445, -0.1726] | +0.1049 [+0.0844, +0.1260] |
| A_4090 - nomL2 | -0.2317 [-0.2400, -0.2237] | -0.4827 [-0.5071, -0.4578] | -0.3223 [-0.3511, -0.2930] | -0.0068 [-0.0212, +0.0078] | -0.4767 [-0.5126, -0.4399] | +0.2661 [+0.2369, +0.2957] |
| B_4090 - nomL2 | -0.2041 [-0.2100, -0.1982] | -0.4126 [-0.4362, -0.3897] | -0.2223 [-0.2481, -0.1989] | +0.0003 [-0.0150, +0.0156] | -0.3450 [-0.3750, -0.3139] | +0.1795 [+0.1573, +0.2041] |
| Cp_4090 - nomL2 | -0.1345 [-0.1405, -0.1288] | -0.2903 [-0.3145, -0.2662] | -0.1613 [-0.1861, -0.1389] | -0.0036 [-0.0155, +0.0093] | -0.2015 [-0.2416, -0.1634] | +0.1068 [+0.0873, +0.1265] |

- Prereg §7 verdict B − A: GUIDED-WIN (AIoU, pen-up F1 v2, strict and direction non-inferiority clauses hold; order wins on pen-up F1 v2, strict, τ_cp; direction CI includes 0). B promoted to the test pass.
- C′ − A (gate-failed, exploratory): the same clauses hold; not promotable (A1).
- Vs nominees (v4 cell clauses): order non-inferiority fails on pen-up F1 v2 and strict for A, B and C′ against both nominees; AIoU lead clause fails (AIoU deltas −0.1345 to −0.2360).
- Secondary device check (CLUSTER-A A4000 minus 4090 sampling of the same checkpoint): all six metrics' 95% CIs include 0 for B and C′ (B AIoU +0.0002 [−0.0004, +0.0008]).

## 6. Distillation read — test_ouhdl (exploratory, 4,406 units, 32 writers, 4090, seeds 0–2)
Reason tag: [experiment] prereg §7 single exploratory test pass, arms A and B. Tables {{artifact:31e3a7fc-0029-4355-8ebc-80fd5753372d}}, absolute {{artifact:9016fc11-1d6f-40b7-864b-9e6d7af1100a}}, inputs {{artifact:5a1eaa4a-1054-4b96-94f9-4029e2003244}}. Nominee test rows: SealedLane1 (lineHT d4) and SealedLane2 (Line T d4) 4090 rows.

| cell | n units | aiou | penup_f1_v2 | strict_order_acc | direction_acc | tau_cp | ldtw |
|---|---|---|---|---|---|---|---|
| A_4090 | 4406 | 0.5501 | 0.2098 | 0.3212 | 0.9129 | 0.2036 | 0.5076 |
| B_4090 | 4406 | 0.5777 | 0.2899 | 0.4262 | 0.9217 | 0.3367 | 0.3951 |
| nomL1 | 4406 | 0.7823 | 0.7036 | 0.6322 | 0.9216 | 0.6896 | 0.2276 |
| nomL2 | 4406 | 0.7786 | 0.7047 | 0.6403 | 0.9217 | 0.6930 | 0.2207 |

| contrast | aiou | penup_f1_v2 | strict_order_acc | direction_acc | tau_cp | ldtw |
|---|---|---|---|---|---|---|
| B_4090 - A_4090 | +0.0276 [+0.0256, +0.0294] | +0.0801 [+0.0698, +0.0908] | +0.1050 [+0.0937, +0.1171] | +0.0108 [+0.0056, +0.0164] | +0.1331 [+0.1166, +0.1502] | -0.1125 [-0.1244, -0.1006] |
| A_4090 - nomL1 | -0.2322 [-0.2375, -0.2272] | -0.4938 [-0.5159, -0.4731] | -0.3110 [-0.3309, -0.2925] | -0.0104 [-0.0184, -0.0028] | -0.4860 [-0.5047, -0.4671] | +0.2800 [+0.2612, +0.3003] |
| B_4090 - nomL1 | -0.2046 [-0.2103, -0.1993] | -0.4137 [-0.4359, -0.3927] | -0.2060 [-0.2245, -0.1889] | -0.0007 [-0.0082, +0.0067] | -0.3529 [-0.3749, -0.3302] | +0.1675 [+0.1512, +0.1856] |
| A_4090 - nomL2 | -0.2285 [-0.2339, -0.2237] | -0.4949 [-0.5164, -0.4744] | -0.3191 [-0.3392, -0.3006] | -0.0106 [-0.0178, -0.0035] | -0.4894 [-0.5074, -0.4713] | +0.2869 [+0.2692, +0.3063] |
| B_4090 - nomL2 | -0.2009 [-0.2065, -0.1958] | -0.4148 [-0.4361, -0.3947] | -0.2141 [-0.2333, -0.1964] | -0.0007 [-0.0073, +0.0057] | -0.3564 [-0.3781, -0.3348] | +0.1744 [+0.1591, +0.1907] |

## 7. Architecture figure
Reason tag: [documentation]. {{artifact:5621b280-4471-4985-bfa1-f6c7ac15845e}} (PDF {{artifact:7708ef20-e0e6-42e2-bdcd-4eaa18cedd49}}, provenance {{artifact:669bf46d-d92b-4d91-82ad-977f374f9802}}). Example unit 1000_0020_4_L001 (val1000, writer 131, 2 body + 2 mark strokes).

## 8. Failed, blocked and deviating items
Reason tag: [failure/deviation].
- nomL2 baseline was missing at Phase 1 start (checkpoint only on the VPS); pulled at ~03:00Z; encoder rows rebuilt with nomL2 at 04:22Z (nomL1-only copy kept: rows_encoders_nomL1only.csv). The encoder track summary md was written before the rebuild.
- InkSight probe rows not assessed: the exported SavedModel exposes no vision tokens. InkSight input was the 256-px G1 crop convention, not the 224-px FMD render. E0 val1000 seeds 1–2 sampled on an RTX A4000 (seed 0 = existing A30 file).
- VLM: V1–V3 comparator is the majority label only; prompts and image presentation frozen in PROMPTS_VLM_FROZEN.json before any model call.
- Diffusion: features grid-mapped to 16x16 inside the extractor; VAE posterior mean; one shared noise tensor; attentive (info) rows used 3 of 4 nominee arms.
- Encoders: DINOv2/DINOv3/SigLIP2 received the 224 render without HF processor resize/crop; SAM2.1 'last' = neck output without no_memory_embedding.
- Student: no in-job gate at step 2,000; below-threshold (LTR-routed) val units sampled by the LUO ms6 backbone, as in the nominee convention; training devices differ across arms (A 4090, B/C′ A4000).
- Monitor: stock vps_pull.sh failed (Windows path) and stock verify_infer.py failed (float32/float64 dtype); versioned copies used, originals unchanged. Scoring ran on cpuidle (8 CPUs) because epyc-256 start estimate was 4 Nov or later.
- Test-table builder writes FMD_val_* file names; test outputs were renamed to FMD_test_* on CLUSTER-A (exp/FMD/runs/student/tables_test/).
- A plotting-input file cand_inputs.npz (125 KB, one unit's PCA and target maps) was saved to the artifact store by the figure track.
- Security: the monitor track reports that a shell `$_` expansion printed the HF token string from the CLUSTER-A shell environment into a tool output (not written to any file or artifact).
