# FMD gate read — FM screen results (val1000, 3 Nov 2026)

Pre-registration: FMD_PREREG_2026-11-03.md (sha256 6ad98ab4…), section 5 gate, section 6 arm choice. Read by the lead session (<agent>; reason: verdict-setting pre-registered read = <agent>).
Units: OUHD-L v3 val1000 ligatures (1,000 units, 31 writers). CIs: writer-cluster bootstrap B=2000, rng 20261103. Comparator for probe rows = best of {nomL1 lineHT d4 ms4, nomL2 Line T d4 ms4} x {mid16, cat16} per target (column 'baseline'). No multiplicity correction (order tests on gate layers: encoder 16, diffusion 4, VLM 7, ink 4).

## Gate rows (linear probe, pre-named gate layer; zero-shot rows for VLM V1–V3 and InkSight)

| FM | layer | target | metric | FM value | comparator | delta [95% CI] | role | row verdict |
|---|---|---|---|---|---|---|---|---|
| dinov2-large | last | T0 | iou | 0.8688 | nomL2:cat16 | -0.0912 [-0.0949, -0.0877] | shape_NI | fail |
| dinov2-large | last | T1 | acc8 | 0.7483 | nomL1:cat16 | -0.0708 [-0.0784, -0.0636] | order_gate | fail |
| dinov2-large | last | T2 | rprec | 0.5586 | nomL1:cat16 | +0.0450 [+0.0116, +0.0770] | order_gate | pass |
| dinov2-large | last | T3 | rprec | 0.5934 | nomL2:cat16 | -0.0411 [-0.0604, -0.0233] | order_gate | fail |
| dinov2-large | last | T4 | spearman | 0.7662 | nomL2:cat16 | -0.0235 [-0.0316, -0.0146] | order_gate | fail |
| dinov3-vitl16 | last | T0 | iou | 0.9063 | nomL1:cat16 | +0.0038 [-0.0003, +0.0073] | shape_NI | pass |
| dinov3-vitl16 | last | T1 | acc8 | 0.7453 | nomL1:cat16 | -0.0653 [-0.0719, -0.0588] | order_gate | fail |
| dinov3-vitl16 | last | T2 | rprec | 0.5430 | nomL1:cat16 | +0.0453 [+0.0099, +0.0808] | order_gate | pass |
| dinov3-vitl16 | last | T3 | rprec | 0.6501 | nomL1:cat16 | +0.0030 [-0.0165, +0.0229] | order_gate | fail |
| dinov3-vitl16 | last | T4 | spearman | 0.7709 | nomL2:cat16 | -0.0167 [-0.0263, -0.0067] | order_gate | fail |
| siglip2-so400m-patch14-224 | last | T0 | iou | 0.8043 | nomL2:cat16 | -0.1557 [-0.1607, -0.1512] | shape_NI | fail |
| siglip2-so400m-patch14-224 | last | T1 | acc8 | 0.7567 | nomL1:cat16 | -0.0624 [-0.0711, -0.0539] | order_gate | fail |
| siglip2-so400m-patch14-224 | last | T2 | rprec | 0.5789 | nomL1:cat16 | +0.0653 [+0.0411, +0.0914] | order_gate | pass |
| siglip2-so400m-patch14-224 | last | T3 | rprec | 0.6680 | nomL2:cat16 | +0.0335 [+0.0100, +0.0561] | order_gate | pass |
| siglip2-so400m-patch14-224 | last | T4 | spearman | 0.7655 | nomL2:cat16 | -0.0243 [-0.0338, -0.0133] | order_gate | fail |
| sam2.1-hiera-base-plus | last | T0 | iou | 0.9422 | nomL2:cat16 | -0.0178 [-0.0200, -0.0154] | shape_NI | fail |
| sam2.1-hiera-base-plus | last | T1 | acc8 | 0.8210 | nomL1:cat16 | +0.0019 [-0.0045, +0.0079] | order_gate | fail |
| sam2.1-hiera-base-plus | last | T2 | rprec | 0.5685 | nomL1:cat16 | +0.0548 [+0.0296, +0.0786] | order_gate | pass |
| sam2.1-hiera-base-plus | last | T3 | rprec | 0.6474 | nomL2:cat16 | +0.0129 [-0.0066, +0.0327] | order_gate | fail |
| sam2.1-hiera-base-plus | last | T4 | spearman | 0.7670 | nomL2:cat16 | -0.0228 [-0.0324, -0.0122] | order_gate | fail |
| sd2.1-dift | t261_up1 | T0 | iou | 0.9498 | nomL2:cat16 | -0.0102 [-0.0127, -0.0077] | shape_NI | fail |
| sd2.1-dift | t261_up1 | T1 | acc8 | 0.8360 | nomL1:cat16 | +0.0169 [+0.0092, +0.0240] | order_gate | pass |
| sd2.1-dift | t261_up1 | T2 | rprec | 0.6324 | nomL1:cat16 | +0.1188 [+0.0922, +0.1442] | order_gate | pass |
| sd2.1-dift | t261_up1 | T3 | rprec | 0.7279 | nomL2:cat16 | +0.0935 [+0.0731, +0.1141] | order_gate | pass |
| sd2.1-dift | t261_up1 | T4 | spearman | 0.7704 | nomL2:cat16 | -0.0193 [-0.0287, -0.0092] | order_gate | fail |
| qwen3-vl-8b-instruct | zeroshot | V1 | acc | 0.1560 | majority:RTL | -0.6340 [-0.6915, -0.5798] | order_gate | fail |
| qwen3-vl-8b-instruct | zeroshot | V2 | acc | 0.5008 | majority:blue | -0.0173 [-0.0658, +0.0330] | order_gate | fail |
| qwen3-vl-8b-instruct | zeroshot | V3 | acc | 0.0928 | majority:yes | -0.8144 [-0.8723, -0.7547] | order_gate | fail |
| qwen3-vl-8b-instruct | last | T0 | iou | 0.7268 | nomL1:cat16 | -0.1757 [-0.1801, -0.1714] | shape_NI | fail |
| qwen3-vl-8b-instruct | last | T1 | acc8 | 0.7652 | nomL1:cat16 | -0.0455 [-0.0537, -0.0381] | order_gate | fail |
| qwen3-vl-8b-instruct | last | T2 | rprec | 0.4751 | nomL1:cat16 | -0.0225 [-0.0533, +0.0082] | order_gate | fail |
| qwen3-vl-8b-instruct | last | T3 | rprec | 0.6268 | nomL1:cat16 | -0.0203 [-0.0365, -0.0030] | order_gate | fail |
| qwen3-vl-8b-instruct | last | T4 | spearman | 0.6891 | nomL2:cat16 | -0.0986 [-0.1104, -0.0849] | order_gate | fail |
| inksight_small_p | zeroshot | ink_traj | aiou | 0.8268 | e0_ms8_ddim50_s012 | +0.0300 [+0.0246, +0.0355] | shape_NI | pass |
| inksight_small_p | zeroshot | ink_traj | tau_cp | 0.2972 | e0_ms8_ddim50_s012 | -0.5406 [-0.5964, -0.4825] | order_gate | fail |
| inksight_small_p | zeroshot | ink_traj | strict_order_acc | 0.5590 | e0_ms8_ddim50_s012 | -0.1870 [-0.2110, -0.1628] | order_gate | fail |
| inksight_small_p | zeroshot | ink_traj | penup_f1_v2 | 0.8334 | e0_ms8_ddim50_s012 | +0.0050 [-0.0189, +0.0301] | order_gate | fail |
| inksight_small_p | zeroshot | ink_traj | direction_acc | 0.5272 | e0_ms8_ddim50_s012 | -0.4348 [-0.4733, -0.3974] | order_gate | fail |

## FM and family verdicts (recomputed from rows; agree with each track's screen_rows.family_verdicts)

| family | FM | T0 shape CI lo (margin −0.01035) | order criteria with CI lo > 0 | FM verdict |
|---|---|---|---|---|
| encoder | dinov2-large | -0.09493 | T2:rprec | FAIL |
| encoder | dinov3-vitl16 | -0.00030 | T2:rprec | PASS |
| ink | inksight_small_p | +0.02456 | — | FAIL |
| VLM | qwen3-vl-8b-instruct | -0.18006 | — | FAIL |
| encoder | sam2.1-hiera-base-plus | -0.01996 | T2:rprec | FAIL |
| diffusion | sd2.1-dift | -0.01268 | T1:acc8;T2:rprec;T3:rprec | FAIL |
| encoder | siglip2-so400m-patch14-224 | -0.16072 | T2:rprec;T3:rprec | FAIL |

Family verdicts: encoder PASS (DINOv3 ViT-L/16 only); diffusion FAIL; VLM FAIL; ink FAIL.

## Arm choice (prereg section 6)

- Passing FMs: 1 (dinov3-vitl16). Arm B = DINOv3 ViT-L/16, REPA-style token alignment (3-layer MLP projector, SiLU, hidden 2,048, discarded after training; negative cosine to frozen gate-layer tokens grid-mapped to 16x16; lambda 0.5).
- Arm C: none (only one FM passes).
- Arm A: 224-px FM-sized encoder student without FM loss (architecture control).

## Info rows next to the DINOv3 pass (prereg section 5 note: T2/T3 can rise through endpoint detection without order information)

| target | metric | comparator | delta [95% CI] |
|---|---|---|---|
| T1 | acc_orient4 | nomL1:cat16 | -0.0746 [-0.0810, -0.0686] |
| T23 | sep_auc | nomL1:cat16 | +0.0018 [-0.0166, +0.0191] |

## Declared deviations carried from the tracks

- Encoders: DINOv2/DINOv3/SigLIP2 received the 224 render without HF processor resize/crop; SAM2.1 'last' = neck output without no_memory_embedding.
- Diffusion: features mapped to 16x16 inside the extractor; VAE posterior mean; one shared noise tensor; attentive (info) rows used 3 of 4 nominee arms.
- VLM: V1–V3 comparator is majority only (as prereg 2.3); prompt/image choices frozen in PROMPTS_VLM_FROZEN.json; Qwen3.5-4B rows exploratory, not in this table.
- Ink: InkSight input = 256-px G1 crops (f=4, thickness 8), not the 224-px FMD render; InkSight single run; E0 seeds 1–2 sampled on an RTX A4000 (seed 0 = existing A30 file); no InkSight probe rows (SavedModel exposes no tokens).
- Staging: SigLIP2 so400m/14 and sd2-community SD2.1 substituted for 404 repos; DINOv3 fetched via an HF_TOKEN present in the CLUSTER-A account environment; DINOv3 licence restricts redistribution of weights and derived checkpoints.

## Inputs (sha256)

- prereg_gate_excerpt.txt: 7aa9eb1983839ca4291e4f869ea8eb90965dd24a99812d260d450e12fabac31d
- rows_diffusion.csv: 5f4faf206122422ca01557bd96b363073619eb8f49ca0ceded5f361f9ce26e3d
- rows_encoders.csv: 1fa76ecd9ea67dcbcda94a0089fd211bcc654c283fc691b3bdfbf30809481ffd
- rows_encoders_nomL1only.csv: 05a5df4921ffc1352095ab65d93bf2ec64e9778956f6d9c333020e9168f46a31
- rows_ink.csv: 8b0b847730cb32478a232b5e8c26373ecb2c86e1e57d5bc95b1e793c515f664a
- rows_vlm.csv: a2d2b66d3ba41859fb3359db7754d7c37651b4f52221d029560555842d164a22
- rows_vlm_exploratory.csv: 82703668a19afe51f1f96c93dc4a23c63aae1b430661218ec1007386b9d5552d
- verdicts_diffusion.json: 7a6315b8a57d7759fd7300365f7108fe95ac0fa0cbeb694bb9840ccfbd7c7f94
- verdicts_encoders.json: f07b8bd9e0cfc513306c104682656d8cd78f79088d9e2e4c8ec5a572b36eaf34
- verdicts_ink.json: ef29f87d8337c7bc86cd436b11c3cfde3e282cd51b59b8bbb1a7d99e433ec522
- verdicts_vlm.json: edb9e4286257f67507181a1e44eeeedc0380fabb5b122b9c358861b5e2fce18d