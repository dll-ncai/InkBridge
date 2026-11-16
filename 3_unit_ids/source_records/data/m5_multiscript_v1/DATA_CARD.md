# DATA CARD: m5_multiscript_v1 (M5 / X6 multi-script training set)

- **Built:** 2026-10-20T20:57:14Z on CLUSTER-A (<node>, Slurm 24516200), status: full. Provenance: PROVENANCE_m5_multiscript_v1.json.
- **Purpose:** equal-size training sets per script for the M5 joint model (5 scripts) and the leave-Urdu-out model (4 scripts), plus selection val sets, an IAM test sample and fixed-seed OUHD-L fine-tune subsets.
- **Licences:** every source forbids redistribution; this derived set stays on CLUSTER-A and is not an artifact.
  - casia_zh: CASIA-OLHWDB 1.0/1.1 (NLPR CASIA): free for academic research under the CASIA licence agreement; no redistribution
  - casia_en: CASIA-OLHWDB 1.0/1.1 (NLPR CASIA): free for academic research under the CASIA licence agreement; no redistribution
  - hpl_tamil: hpl-tamil-iso-char (HP Labs India): research use only; no publishing/redistribution of any part (data/hpl_tamil/README.md)
  - iam: IAM-OnDB (FKI/IAM): non-commercial research; no redistribution (data/d2_v2/iam/DATA_CARD.md)
  - ouhdl: OUHD-L: terms as in data/ouhdl_ligatures_v3 data card

## Unit format (identical for all scripts; E0 format)
Ink on the 64x64 evaluation canvas (uhwr_eval.fit_transform_unit: bbox fitted into 56x56, centred, 4 px margin, y down, x not mirrored). CASIA / Tamil raw ink first passes uhwr_eval.clean_ink (D2 UF.to_eval_space 'isolated'); IAM and OUHD-L arrays are taken unchanged from data/d2_v2/iam and data/ouhdl_e0_v3. Target `seq` = e0_adapter.encode: per-stroke 1 px arc-length resampling, step x1.05 until <= 320 points (no truncation), Nagamatsu [x, y, pen, eos]. Condition `cond_rendered` = e0_adapter.render_condition (2 px anti-aliased, 64x64, black on white uint8). `sample_ids` = writer key `<script>:<writer>` (e0_score.py rows then carry writer clusters); `writer_ids` = integer index of the sorted writer keys within the file; `scripts` = script tag.

## Sources and sampling (seed 20261016)
- **casia_zh:** CASIA OLHWDB1.0+1.1 train .pot (data/casia/CASIA/train, 576 writer files), labels = single CJK char (U+4E00-9FFF)
- **casia_en:** same .pot files, labels = single ASCII letter a-z/A-Z (D2 casia_en rule)
- **hpl_tamil:** HPL IWFHR06 train online (data/hpl_tamil/extracted/hpl-tamil-iwfhr06-train-online-0.2/new/usr_<id>)
- **iam:** data/d2_v2/iam (X3 component units of IAM-OnDB t1 lines): train = e0_train.npz (trainset), val = e0_val1.npz (testset_v)
- **ouhdl:** data/ouhdl_e0_v3 (E0 arrays of OUHD-L ligature units v3.0): train = e0_train.npz, val = e0_val.npz
- Val writers: CASIA 24 of the train writer files (032, 052, 070, 078, 118, 137, 180, 238, 243, 292, 349, 387, 390, 397, 1014, 1056, 1062, 1071, 1091, 1093, 1137, 1140, 1187, 1198); Tamil 6 users (usr_109, usr_121, usr_156, usr_157, usr_185, usr_211); IAM / OUHD-L official val splits. Train samples come only from non-val writers; val samples only from val writers (CASIA / Tamil).
- Train sample: n_train units uniformly at random without replacement from the train pool; val sample: n_val units from the val pool.

| pool | pool units | pool writers | picked |
|---|---|---|---|
| casia_zh_train | 2105324 | 552 | 20000 |
| casia_zh_val | 91562 | 24 | 250 |
| casia_en_train | 28699 | 552 | 20000 |
| casia_en_val | 1247 | 24 | 250 |
| hpl_tamil_train | 49156 | 111 | 20000 |
| hpl_tamil_val | 1527 | 6 | 250 |
| iam_train | 81659 | n/a (all) | 20000 |
| iam_val | 18624 | n/a (all) | 250 |
| ouhdl_train | 33615 | n/a (all) | 20000 |
| ouhdl_val | 4349 | n/a (all) | 250 |

## Point counts at the E0 setting (64x64 canvas, 1 px arc-length step, L = 320): equal-sequence-length check
n_points_1px = points after 1 px per-stroke resampling before any step growth; n_seq = stored sequence length (<= 320); step = resampling step used (1.0 unless n_points_1px > 320). Computed on the sampled units (train = the 20k training sample of each script). OUHD-L train arrays carry no GT ink: for its units with step > 1 the 1 px count is re-estimated from the decoded sequence (n1px_exact_units says how many are exact).

| script | role | n_units | n_writers | n_errors | n1px_exact_units | n_points_1px_p50 | n_points_1px_p90 | n_points_1px_p99 | n_points_1px_p100 | n_points_1px_mean | frac_n1px_gt_320 | n_seq_mean | n_seq_p100 | step_p90 | step_p100 | frac_step_gt_1 | n_strokes_p50 | n_strokes_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| casia_zh | train | 20000 | 552 | 0 | 20000 | 312.000 | 432.000 | 546.000 | 845.000 | 317.902 | 0.462 | 281.296 | 320.000 | 1.407 | 2.786 | 0.462 | 5.000 | 5.630 |
| casia_zh | val | 250 | 24 | 0 | 250 | 316.000 | 418.500 | 511.060 | 557.000 | 318.232 | 0.468 | 283.508 | 320.000 | 1.340 | 1.796 | 0.468 | 4.000 | 4.992 |
| casia_en | train | 20000 | 552 | 0 | 20000 | 153.000 | 211.000 | 271.000 | 1212.000 | 157.433 | 0.002 | 157.151 | 320.000 | 1.000 | 3.920 | 0.002 | 1.000 | 1.487 |
| casia_en | val | 250 | 24 | 0 | 250 | 151.000 | 200.000 | 262.570 | 280.000 | 153.812 | 0.000 | 153.812 | 280.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.476 |
| hpl_tamil | train | 20000 | 111 | 0 | 20000 | 205.000 | 265.000 | 316.000 | 501.000 | 205.343 | 0.008 | 205.149 | 320.000 | 1.000 | 1.629 | 0.008 | 1.000 | 1.505 |
| hpl_tamil | val | 250 | 6 | 0 | 250 | 225.500 | 276.000 | 330.590 | 361.000 | 220.292 | 0.020 | 219.776 | 320.000 | 1.000 | 1.158 | 0.020 | 2.000 | 1.588 |
| iam | train | 20000 | 382 | 0 | 20000 | 141.000 | 197.000 | 260.000 | 459.000 | 140.297 | 0.001 | 140.269 | 320.000 | 1.000 | 1.477 | 0.001 | 1.000 | 1.690 |
| iam | val | 250 | 62 | 0 | 250 | 137.500 | 195.200 | 253.690 | 271.000 | 137.236 | 0.000 | 137.236 | 271.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.688 |
| ouhdl | train | 20000 | 248 | 0 | 19986 | 124.000 | 176.000 | 230.000 | 1130.000 | 124.606 | 0.001 | 124.537 | 320.000 | 1.000 | 4.322 | 0.001 | 2.000 | 2.196 |
| ouhdl | val | 250 | 31 | 0 | 250 | 119.500 | 166.000 | 224.690 | 277.000 | 118.840 | 0.000 | 118.840 | 277.000 | 1.000 | 1.000 | 0.000 | 2.000 | 2.112 |

## Files
- `build_stats_M5.json`: 5786 B, sha256 3916f2cb5e58a492e370aabe80e1bfd118c7c9b609bb06f1898dfc4dd356001f
- `ft_subsets_M5.json`: 852 B, sha256 71f85df0428c4b662e21a6a81bd7a92885c172a4460ba7f9c966a1fa5cb6b43e
- `m5_ft_ouhdl_p01.npz`: 2086428 B, sha256 0e71475d6094b69a10dd1301bf05662e37816471407d2ee6e08a700a6780aad5
- `m5_ft_ouhdl_p10.npz`: 20921260 B, sha256 3770d4d4cd6cbd5f1131e26d50be9412ce99daea345e3fc01e34d07c4de4696f
- `m5_ft_ouhdl_p30.npz`: 62762604 B, sha256 4e4394ae0b5e1ededf7c4aa40b936ede8530da3f511b663e1366fd118d79d702
- `m5_joint_train.npz`: 725090434 B, sha256 56649a2581bcbc3d47240cc77f39256049dc81fed993ea88ba9c165d6862694b
- `m5_luo_train.npz`: 598358642 B, sha256 690441b6bd7c119d774066622c53f23a0aca6e62825a04014351bfba98260b67
- `m5_test_iam500.npz`: 3747112 B, sha256 0dbd1a613adadb4e5b846d5ecbdb5e7df9b0006d6a7665ee37e1020fc00637de
- `m5_val_casia_en.npz`: 1778752 B, sha256 eea15ff926c70bae84483f940f4491a1f784c8f1e40c9bae76b12f5be9281c8b
- `m5_val_casia_zh.npz`: 2506896 B, sha256 4927393116fb065b559f26323faddc0d3f5323d839b61fad75cbc205136ca2e7
- `m5_val_hpl_tamil.npz`: 2797508 B, sha256 dce93506078cf17f8da84377a5bc5b5ebfc53da8e2025f079e8093683f08d686
- `m5_val_iam.npz`: 1853908 B, sha256 5ecbd8b0b4b159a87a8f65eeb3d298bdf233f708debbbea8a2a9699f03c7f840
- `m5_val_ouhdl.npz`: 3050484 B, sha256 9c6b7dd93fde089fe3023f1539ac8c51d7f053e3d0f819451afb65bbbb73c707
- `members_casia_en_train.csv`: 1074732 B, sha256 5242cc5b5b530a7c5891d0ca3d2a4d442fedee81b66333fe56e7b57b53d0e19a
- `members_casia_en_val.csv`: 13512 B, sha256 0696ccdad11aca0787da1416e78862efeadcacb2b0ded55a81b73c1e3240710e
- `members_casia_zh_train.csv`: 1219962 B, sha256 d843692cd2169c841b83bda51a01abcedbd70849ce087dd2fa8b4efb54e7d52c
- `members_casia_zh_val.csv`: 15319 B, sha256 4a548354d302f9be77a5968cd0f3163cae0d820d41d5bb517c3950a637ac0c55
- `members_hpl_tamil_train.csv`: 1299010 B, sha256 83950b7d9bcdf8dc0fce91255499b092228b36c830663e2175caa183d8a84277
- `members_hpl_tamil_val.csv`: 16583 B, sha256 1e9964f16f5da492197c7c87c4bc7d0d3d560b2ca826df25b61ae2a403233954
- `members_iam_train.csv`: 1029863 B, sha256 c922c535e20ea3a25711ad3151b0cd540979c1e841d6c720974ee61e4c227b46
- `members_iam_val.csv`: 13340 B, sha256 1c303a48297686746ac279ad69a7db6f02f810d879e6b9d6fd4be5430c0ef1a5
- `members_ouhdl_train.csv`: 1041695 B, sha256 22c1636ea39a24dd56da6794e175ac337a08bcf25dd7789becccb3b2e6ed15cf
- `members_ouhdl_val.csv`: 13062 B, sha256 f47bf22f54d7f4016840265b5ecb201fa5cb7e172fb966586e59566d41c37ce4
- `seqlen_check_M5.csv`: 2402 B, sha256 b1615e7c375310bea9628bf2965417aa5fdbc47c78c28a1ca9e9614863cf8111

## Fine-tune subsets (OUHD-L E0 train, nested)
Rule: nested prefixes of one permutation, numpy default_rng([20261016, 9]).permutation(N); k = floor(frac*N + 0.5). Source $U/data/ouhdl_e0_v3/e0_train.npz (sha256 2c4ddc48fb37b3eb3145977d9245f16e2e237f1224e24b5c81b72785d8a1f048).
- p01: 336 units, 183 writers, 305 lines -> m5_ft_ouhdl_p01.npz
- p10: 3362 units, 247 writers, 1569 lines -> m5_ft_ouhdl_p10.npz
- p30: 10085 units, 248 writers, 1903 lines -> m5_ft_ouhdl_p30.npz
- p100: 33615 units, 248 writers, 1911 lines -> $U/data/ouhdl_e0_v3/e0_train.npz

## Eval sets used by M5 (not rebuilt here)
OUHD-L v3 test (data/ouhdl_e0_v3/e0_test.npz, 4,406 units) and gold (e0_gold.npz, 354 units); X4a 500-unit sets of CASIA zh/en (OLHWDB1.1 test writers) and HPL Tamil (IWFHR06 test) in exp/X4a/data; m5_test_iam500.npz (500 IAM testset_f units with a known writer, seed 2026).


## Licence

LICENSE: not redistributable (mixed sources: CASIA-OLHWDB licence agreement, HP Labs India Tamil licence (no redistribution), IAM-OnDB terms, OUHD-L CC BY 4.0). CLUSTER-A only.

(Licence line added 2026-10-22 by the uhwr licence pass; register: shared/licences/licence_register_20261022.csv; dataset: m5_multiscript_v1.)
