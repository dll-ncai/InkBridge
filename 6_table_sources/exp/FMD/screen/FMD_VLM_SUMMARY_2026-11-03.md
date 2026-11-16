# FMD phase 1, track VLM: facts summary (3 Nov 2026)

Pre-registration: `exp/FMD/plans/FMD_PREREG_2026-11-03.md` (sha256 6ad98ab4c1f4…). Set: val1000 (1,000 OUHD-L v3.0 val ligature units, 31 writers). Code: `exp/FMD/code/fmd_ext_vlm` (tar `fmd_ext_vlm_v1.4.tar.gz`, sha256 52c4bac8…) importing fmd_harness 1.0.1 without changes. Frozen prompt/decoding/parse spec: `PROMPTS_VLM_FROZEN.json` (sha256 57e69c96…), written before any model call.
Sub-agent model: <agent> Reason: new prompt harness and verdict-setting pre-registered tasks.

## 1. Pre-registered FM: Qwen3-VL-8B-Instruct (rev 0c351dd0…; 16/16 weight files match the staging manifest sha256)

### 1a. Zero-shot V1-V4 [reason: prereg section 2.3]
- Job 24829989, NVIDIA A30 (<node>), bf16, greedy decoding, batch 1, max_new_tokens 8 (V1-V3) / 32 (V4), 224-px image at its native 14x14 patch grid (49 visual tokens; image processor min_pixels set to 224², without which the stock processor upscales to 256 px). Wall 23 min. 3,217 raw outputs (V1 1,000; V2 635; V3 582; V4 1,000) in `exp/FMD/screen/units/vlm/qwen3-vl-8b-instruct__zeroshot_raw__val1000.jsonl`.
- Parse (first stand-alone A/B/C): V1 1,000/1,000 valid, V2 635/635, V3 582/582; 0 no-letter, 0 out-of-options. The literal variant (first A/B/C character anywhere) gave a different letter for 0 units.
- Answer distribution: V1 LTR 920 / OTHER 66 / RTL 14 (labels RTL 790 / LTR 103 / OTHER 107). V2 blue 466 / red 169 (labels blue 329 / red 306). V3 'no' 582 / 'yes' 0 (labels yes 528 / no 54).
- V4: 1,000 units with aligned_text. Mean reference length 2.04 characters; mean hypothesis length 6.56 (median 5) after NFKC and whitespace removal; 973/1,000 outputs contain Arabic-script characters; 24 outputs reached the 32-token limit.

### 1b. Vision-tower probes [reason: prereg sections 2.4, 3]
- Features: `last` = output of block 27 of 27, before the 2x2 merger (gate layer); `mid` = output of block 13 of 27. Native 14x14 grid -> G = 14. Vision tower only, float32 on CPU (job 24829987, 5 min), cached float16 ([1000,14,14,1152]; max |value| 16,421 for last and 242 for mid). The token re-ordering from the processor's merge-window order to raster order was checked against pixel patch means (max abs difference 7.8e-8).
- Probes: harness linear + attentive, seeds 0-2, 5 writer-disjoint folds, CPU (array 24829988).
- Comparator per row = best of {nomL1, nomL2} x {mid16, cat16} for the same probe type and grid (nomL2 G14 unit files present from 03:13Z; the final scoring job 24830959 includes them).

### 1c. Rows (`exp/FMD/screen/rows_vlm.csv`, 33 rows, sha256 a2d2b66d…)
| layer | probe | task/target | metric | value | comparator | comparator value | delta | 95% CI (writer-cluster, B=2000) | n units | gate_role | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | zeroshot | V1 | acc | 0.1560 | majority:RTL | 0.7900 | -0.6340 | [-0.6915, -0.5798] | 1000 | order_gate | fail |
| zeroshot | zeroshot | V2 | acc | 0.5008 | majority:blue | 0.5181 | -0.0173 | [-0.0658, 0.0330] | 635 | order_gate | fail |
| zeroshot | zeroshot | V3 | acc | 0.0928 | majority:yes | 0.9072 | -0.8144 | [-0.8723, -0.7547] | 582 | order_gate | fail |
| zeroshot | zeroshot | V4 | cer | 3.7263 | none |  |  | [3.4645, 4.0242] | 1000 | info | info |
| zeroshot | zeroshot | V4 | exact | 0.0030 | none |  |  | [0.0000, 0.0066] | 1000 | info | info |
| last | linear | T0 | iou | 0.7268 | nomL1:cat16 | 0.9025 | -0.1757 | [-0.1801, -0.1714] | 1000 | shape_NI | fail |
| last | linear | T1 | acc8 | 0.7652 | nomL1:cat16 | 0.8106 | -0.0455 | [-0.0537, -0.0381] | 1000 | order_gate | fail |
| last | linear | T1 | acc_orient4 | 0.8817 | nomL1:cat16 | 0.8902 | -0.0085 | [-0.0128, -0.0041] | 1000 | info | info |
| last | linear | T2 | rprec | 0.4751 | nomL1:cat16 | 0.4977 | -0.0225 | [-0.0533, 0.0082] | 1000 | order_gate | fail |
| last | linear | T23 | sep_auc | 0.8742 | nomL1:cat16 | 0.8852 | -0.0111 | [-0.0252, 0.0028] | 1000 | info | info |
| last | linear | T3 | rprec | 0.6268 | nomL1:cat16 | 0.6470 | -0.0203 | [-0.0365, -0.0030] | 1000 | order_gate | fail |
| last | linear | T4 | spearman | 0.6891 | nomL2:cat16 | 0.7877 | -0.0986 | [-0.1104, -0.0849] | 1000 | order_gate | fail |
| last | attn | T0 | iou | 0.8768 | nomL2:mid16 | 0.9396 | -0.0628 | [-0.0710, -0.0554] | 1000 | info | info |
| last | attn | T1 | acc8 | 0.7852 | nomL1:cat16 | 0.8224 | -0.0371 | [-0.0469, -0.0271] | 1000 | info | info |
| last | attn | T1 | acc_orient4 | 0.8790 | nomL2:mid16 | 0.8984 | -0.0194 | [-0.0246, -0.0137] | 1000 | info | info |
| last | attn | T2 | rprec | 0.5042 | nomL2:cat16 | 0.5221 | -0.0179 | [-0.0395, 0.0032] | 1000 | info | info |
| last | attn | T23 | sep_auc | 0.8943 | nomL2:cat16 | 0.9075 | -0.0132 | [-0.0223, -0.0044] | 1000 | info | info |
| last | attn | T3 | rprec | 0.6777 | nomL2:cat16 | 0.7044 | -0.0266 | [-0.0460, -0.0067] | 1000 | info | info |
| last | attn | T4 | spearman | 0.7241 | nomL2:cat16 | 0.8060 | -0.0819 | [-0.0920, -0.0720] | 1000 | info | info |
| mid | linear | T0 | iou | 0.9594 | nomL1:cat16 | 0.9025 | 0.0570 | [0.0547, 0.0592] | 1000 | info | info |
| mid | linear | T1 | acc8 | 0.8410 | nomL1:cat16 | 0.8106 | 0.0304 | [0.0242, 0.0364] | 1000 | info | info |
| mid | linear | T1 | acc_orient4 | 0.9274 | nomL1:cat16 | 0.8902 | 0.0373 | [0.0326, 0.0418] | 1000 | info | info |
| mid | linear | T2 | rprec | 0.6254 | nomL1:cat16 | 0.4977 | 0.1277 | [0.0979, 0.1568] | 1000 | info | info |
| mid | linear | T23 | sep_auc | 0.8959 | nomL1:cat16 | 0.8852 | 0.0107 | [0.0011, 0.0195] | 1000 | info | info |
| mid | linear | T3 | rprec | 0.7443 | nomL1:cat16 | 0.6470 | 0.0973 | [0.0766, 0.1174] | 1000 | info | info |
| mid | linear | T4 | spearman | 0.7696 | nomL2:cat16 | 0.7877 | -0.0180 | [-0.0269, -0.0083] | 1000 | info | info |
| mid | attn | T0 | iou | 0.9728 | nomL2:mid16 | 0.9396 | 0.0332 | [0.0307, 0.0357] | 1000 | info | info |
| mid | attn | T1 | acc8 | 0.8562 | nomL1:cat16 | 0.8224 | 0.0338 | [0.0263, 0.0408] | 1000 | info | info |
| mid | attn | T1 | acc_orient4 | 0.9299 | nomL2:mid16 | 0.8984 | 0.0315 | [0.0268, 0.0365] | 1000 | info | info |
| mid | attn | T2 | rprec | 0.6536 | nomL2:cat16 | 0.5221 | 0.1316 | [0.1053, 0.1569] | 1000 | info | info |
| mid | attn | T23 | sep_auc | 0.9197 | nomL2:cat16 | 0.9075 | 0.0122 | [0.0045, 0.0200] | 1000 | info | info |
| mid | attn | T3 | rprec | 0.7839 | nomL2:cat16 | 0.7044 | 0.0795 | [0.0619, 0.0976] | 1000 | info | info |
| mid | attn | T4 | spearman | 0.7855 | nomL2:cat16 | 0.8060 | -0.0204 | [-0.0297, -0.0107] | 1000 | info | info |

### 1d. Gate (prereg section 5; `screen_rows.family_verdicts` on rows_vlm.csv -> `exp/FMD/screen/verdicts_vlm.json`)
- Order tests on the gate layer: 4 probe (T1 acc8, T2 rprec, T3 rprec, T4 spearman) + 3 zero-shot (V1, V2, V3 vs majority) = 7; no multiplicity correction (prereg). Order criteria with CI lo > 0: 0 of 7.
- Shape clause: T0 iou delta CI lo = -0.1801 vs margin -0.01035 -> fail.
- qwen3-vl-8b-instruct: fm_pass = false. VLM family verdict (this track's rows only): FAIL.
- Non-gate rows (`mid` layer, attentive probe) carry gate_role 'info' as pre-registered; they do not enter the verdict.

## 2. Exploratory, not pre-registered: Qwen3.5-4B (rev 851bf6e8…; 14/14 weight files match)
Run only after Qwen3-VL-8B finished (jobs submitted with afterok on 24829989). Same frozen prompts, image presentation, decoding and parse; chat template with enable_thinking=False. Zero-shot job 24830088 (A30, 25.6 min); features job 24830086 (depth 24: last = block 24, mid = block 12; hidden 1024); probes array 24830087; scoring 24830956. All rows have gate_role 'info' and are in `exp/FMD/screen/rows_vlm_exploratory.csv` (sha256 82703668…), separate from the gate file.
- Parse: 0 failures in V1-V3. Answer distribution: V1 LTR 1,000; V2 blue 627 / red 8; V3 'no' 582. V4 mean hypothesis length 1.41 (median 1; 56 empty).

| layer | probe | task/target | metric | value | comparator | comparator value | delta | 95% CI (writer-cluster, B=2000) | n units | gate_role | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | zeroshot | V1 | acc | 0.1030 | majority:RTL | 0.7900 | -0.6870 | [-0.7345, -0.6406] | 1000 | info | info |
| zeroshot | zeroshot | V2 | acc | 0.5118 | majority:blue | 0.5181 | -0.0063 | [-0.0165, 0.0032] | 635 | info | info |
| zeroshot | zeroshot | V3 | acc | 0.0928 | majority:yes | 0.9072 | -0.8144 | [-0.8723, -0.7547] | 582 | info | info |
| zeroshot | zeroshot | V4 | cer | 0.9907 | none |  |  | [0.9666, 1.0169] | 1000 | info | info |
| zeroshot | zeroshot | V4 | exact | 0.0300 | none |  |  | [0.0183, 0.0425] | 1000 | info | info |
| last | linear | T0 | iou | 0.9406 | nomL1:cat16 | 0.9025 | 0.0381 | [0.0357, 0.0404] | 1000 | info | info |
| last | linear | T1 | acc8 | 0.7784 | nomL1:cat16 | 0.8106 | -0.0322 | [-0.0400, -0.0251] | 1000 | info | info |
| last | linear | T1 | acc_orient4 | 0.8973 | nomL1:cat16 | 0.8902 | 0.0071 | [0.0022, 0.0119] | 1000 | info | info |
| last | linear | T2 | rprec | 0.5539 | nomL1:cat16 | 0.4977 | 0.0562 | [0.0327, 0.0792] | 1000 | info | info |
| last | linear | T23 | sep_auc | 0.8752 | nomL1:cat16 | 0.8852 | -0.0100 | [-0.0218, 0.0025] | 1000 | info | info |
| last | linear | T3 | rprec | 0.6720 | nomL1:cat16 | 0.6470 | 0.0250 | [0.0057, 0.0424] | 1000 | info | info |
| last | linear | T4 | spearman | 0.6861 | nomL2:cat16 | 0.7877 | -0.1015 | [-0.1124, -0.0896] | 1000 | info | info |
| last | attn | T0 | iou | 0.9525 | nomL2:mid16 | 0.9396 | 0.0129 | [0.0093, 0.0165] | 1000 | info | info |
| last | attn | T1 | acc8 | 0.8125 | nomL1:cat16 | 0.8224 | -0.0098 | [-0.0171, -0.0027] | 1000 | info | info |
| last | attn | T1 | acc_orient4 | 0.9045 | nomL2:mid16 | 0.8984 | 0.0061 | [0.0011, 0.0113] | 1000 | info | info |
| last | attn | T2 | rprec | 0.5897 | nomL2:cat16 | 0.5221 | 0.0676 | [0.0426, 0.0925] | 1000 | info | info |
| last | attn | T23 | sep_auc | 0.8941 | nomL2:cat16 | 0.9075 | -0.0134 | [-0.0224, -0.0051] | 1000 | info | info |
| last | attn | T3 | rprec | 0.7275 | nomL2:cat16 | 0.7044 | 0.0231 | [0.0022, 0.0433] | 1000 | info | info |
| last | attn | T4 | spearman | 0.7330 | nomL2:cat16 | 0.8060 | -0.0730 | [-0.0826, -0.0631] | 1000 | info | info |
| mid | linear | T0 | iou | 0.9847 | nomL1:cat16 | 0.9025 | 0.0822 | [0.0789, 0.0853] | 1000 | info | info |
| mid | linear | T1 | acc8 | 0.8087 | nomL1:cat16 | 0.8106 | -0.0020 | [-0.0102, 0.0050] | 1000 | info | info |
| mid | linear | T1 | acc_orient4 | 0.9199 | nomL1:cat16 | 0.8902 | 0.0297 | [0.0251, 0.0342] | 1000 | info | info |
| mid | linear | T2 | rprec | 0.6029 | nomL1:cat16 | 0.4977 | 0.1052 | [0.0726, 0.1402] | 1000 | info | info |
| mid | linear | T23 | sep_auc | 0.8886 | nomL1:cat16 | 0.8852 | 0.0033 | [-0.0075, 0.0138] | 1000 | info | info |
| mid | linear | T3 | rprec | 0.7274 | nomL1:cat16 | 0.6470 | 0.0804 | [0.0605, 0.1004] | 1000 | info | info |
| mid | linear | T4 | spearman | 0.7297 | nomL2:cat16 | 0.7877 | -0.0580 | [-0.0669, -0.0473] | 1000 | info | info |
| mid | attn | T0 | iou | 0.9713 | nomL2:mid16 | 0.9396 | 0.0316 | [0.0292, 0.0339] | 1000 | info | info |
| mid | attn | T1 | acc8 | 0.8344 | nomL1:cat16 | 0.8224 | 0.0121 | [0.0040, 0.0192] | 1000 | info | info |
| mid | attn | T1 | acc_orient4 | 0.9238 | nomL2:mid16 | 0.8984 | 0.0254 | [0.0204, 0.0303] | 1000 | info | info |
| mid | attn | T2 | rprec | 0.6316 | nomL2:cat16 | 0.5221 | 0.1095 | [0.0803, 0.1386] | 1000 | info | info |
| mid | attn | T23 | sep_auc | 0.9068 | nomL2:cat16 | 0.9075 | -0.0007 | [-0.0092, 0.0075] | 1000 | info | info |
| mid | attn | T3 | rprec | 0.7717 | nomL2:cat16 | 0.7044 | 0.0673 | [0.0476, 0.0869] | 1000 | info | info |
| mid | attn | T4 | spearman | 0.7477 | nomL2:cat16 | 0.8060 | -0.0583 | [-0.0679, -0.0478] | 1000 | info | info |

## 3. Failed or superseded attempts
- [reason: preview on partial outputs] Slurm 24830073 (scoring while V3 was not yet generated): FAILED with KeyError ('V3','acc'); fixed by skipping tasks without records (score_vlm.py sha c95504fb…).
- [reason: comparator completeness] Scoring job 24830377 (03:00Z) used nomL1 arms only, since nomL2 G14 unit files did not exist yet; its rows_vlm.csv (sha d5995196…) was replaced by job 24830959. Verdict unchanged (FAIL); the rows where a nomL2 arm had the highest mean changed comparator to nomL2.
- [reason: GPU race] 24829990 (gpuidle H100) cancelled while pending; 24829991 (gpuidle 2x A4000) and 24830089 (gpuidle A4000) started, found the lock held by the A30 job and exited after 2-3 s.

## 4. Choices not specified in the prereg (frozen before running, in PROMPTS_VLM_FROZEN.json)
Image size 224 without resize; RGB replication of gray renders; chat format (user turn, image before text, no system message); batch size 1; parse rule = first stand-alone A/B/C; CER = Levenshtein over code points / reference length after NFKC and whitespace removal, uncapped; vision features computed in float32 on CPU; mid = block floor(depth/2) counted 1-based.


Erratum (lead, 3 Nov 2026): the delivered code tar is fmd_ext_vlm_v1.4.tar.gz, sha256 52c4bac882d3761b0f9158c8a469c27d0a66ccfc2b008e3848bd1c23b86b3f05 (CLUSTER-A exp/FMD/code/, artifact version 8a9d9b51); the earlier text named the superseded v1.3 (sha256 2227b29e…).
