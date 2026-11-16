# INVENTORY — reusable checkpoints, preds and comparator seeds for the method round (2026-10-29)

Written 2026-10-28T22:01:04Z by the Prereg track (Phase 0) of PLAN_method_round.json. Sub-agent model: <agent>, under a Fable main session (<agent>).
Sources: CLUSTER-A sha256 job (connector job 19ac0c79-63a2-40e4-82dd-34b5a3170f07, skylake-96; 167 checkpoint files hashed, 93 K1/Route A preds hashed); VPS Get-FileHash on 30 checkpoint files (composed, tier U p01, selfattn); readout v2 preds_manifest.csv (328 rows, sha256 ad6cefcf…) and inputs_cand.csv (263 rows, sha256 a5a4524b…).
All files stay on CLUSTER-A (or the VPS where stated). This document and its two CSVs are the only local copies.

## 1. Arms, selected milestones and checkpoint sha256

selected_sha256 = sha256 of the val1000-selected checkpoint file as hashed on 2026-10-28/29; "manifest match" = equality with checkpoint_sha256 in readout v2 preds_manifest.csv; "VPS copy match" = equality of the CLUSTER-A and VPS copies of the same milestone.

| item | train job | train seed | checkpoints available | selected ms | selected sha256 | manifest match | VPS copy match | preds files (manifest) |
|---|---|---|---|---|---|---|---|---|
| C1C2_p01 composed (init_s=None) | vps:531_cand_c1c2_composed_p01 | 20261027 | ms1-10 (VPS only) | 10 | `48745f6e833e016e4c61ce6cd9a8fbd851252c3a166472e5b371c0dcd3cff50d` | True | None | 13 |
| C2_U_p01 tier U p01 (init_s=None) | vps:520_cand_hsrU_p01_gate2k | 20261027 | ms1-10 | 9 | `f2c56a5e0b20a704f640eac3134e58ad3c117335a4a638d43e903eee6e783ddf` | True | True | 13 |
| C2_U_selfattn_p10 pathway (init_s=None) | vps:573_cand_c2_hsrU_selfattn_p10_rec | 20261027 | ms1-10 | 10 | `a3ab82d77965b1f0cfec2958ad5fc2511b42d62dfd43b639000ec8bf731a0b81` | True | True | 13 |
| C1_p01_FR frame+replay | clusterA:24688253 (B200) | 20261016 | ms1-10 | 10 | `0fa29426311f8447c65c2927c4158e5d01279d47609ebc983ac02c16ebd146ba` | True | None | 13 |
| C1_p01_FO frame only | clusterA:24688254 (B200) | 20261016 | ms1-10 | 4 | `ac11e2df501421d204aecb1a8321b018f3ef5e5be11835277c58a1c56d01cafd` | True | None | 13 |
| C1_p10_FR frame+replay p10 | clusterA:24688255 (B200) | 20261016 | ms1-10 | 10 | `861a745234320bc52f1786d020c75a31f29ab4e8282960aa02e46e2d1c690feb` | True | None | 13 |
| CTRL_s1 LUO-FT p01 (comparator seed) | clusterA:24688007 (B200) | 20261017 | ms1-10 | 3 | `cd60f5c391b808de596b8e3d20dd9b7dda76fd430b4249f6fd8641ac976c4e7b` | True | None | 31 |
| CTRL_s2 LUO-FT p01 (comparator seed) | clusterA:24688009 (B200) | 20261018 | ms1-10 | 4 | `1ba72ad4317c9e510708e0beba278ae00e4f2b9d44113a42a4a3d39451b86057` | True | None | 31 |
| CTRL_cont10k LUO +10k no Urdu | clusterA:24688011 (B200) | 20261016 | ms1-10 | 9 | `b726243b13a4efc4f3065a7a1132636aa520c6115f260944edc90ec8ddf1cf6e` | True | None | 23 |
| C2_S tier S shared LoRA 30k | clusterA:24685216 (A30) | 20261027 | ms5,10,15,20,25-30 | 15 | `ae4ca5b562ef0e798ec47e6768582caefc5c3e2b39dd06eeb4b83402ec866b2c` | True | None | 12 |
| C2_U_p10 tier U p10 (init from tier S ckpt30) | vps:570_cand_c2_hsrU_all_p10 | 20261027 | ms1-10 | 9 | `6fcd56a7fa1f30513d37bf093adfae8bf9338e4f6e11add182c62f2763f172a5` | True | None | 13 |
| C2_U_p30 tier U p30 (init from tier S ckpt30) | vps:571_cand_c2_hsrU_all_p30 | 20261027 | ms1-10 | 10 | `281e7cf414b2b30f3574d5b068dd06ee852176435322f37375fa33f1243f8dfa` | True | None | 13 |
| C3_p01_q25 replay-only (killed at K3, 4k) | clusterA:24686678 (A40) | n/a | ms1-4 | 4 | `e98553cad451501b60aedbe6744e8b4d1fc36c97824835cf094628534a9257ed` | True | None | 2 |
| RA_FT_p01 Route A LUO-FT p01 (comparator seed) | clusterA:24517796 (M5 ft p01) | 20261016 | ms1-10 | 4 | `d770109bb9fcccb36ed13cc68b213cc3d43153255fce5fdc445278d608d65951` | True | None | 20 |
| RA_FT_p10 Route A LUO-FT p10 (reference) | clusterA:24517796 (M5 ft p10) | n/a | ms1-10 | 10 | `9bca5622c97e1f26688c897e4314e0668e458275e3aad6649dac7a99cbed2aed` | True | None | 20 |
| LUO_ms6 backbone | M5 luo | n/a | ms6 only | 6 | `ab6b178d032d90e5ec86f5140e3dd166e0bda497b019d88d6d3fab5da646812c` | True | None | 45 |
| E0 ms8 OUHD-L specialist | clusterA:24417278 | n/a | ms8 (+fp16 EMA copy tmp/teachers_fp16/e0_ms8_fp16.pt) | 8 | `9a9edb199d1c3308996b30a1663f83af3bf8dde3fd8f10deee8a7511f37700d5` | None | None | 0 |

## 2. Paths

| item | CLUSTER-A checkpoint dir | VPS checkpoint dir | preds dir (CLUSTER-A) |
|---|---|---|---|
| C1C2_p01 composed (init_s=None) | (not on CLUSTER-A) | $VPS_ROOT\uhwr\runs\CAND\C1\c1c2_composed_p01_mirror_x_s20261027\checkpoints | exp/CAND/preds/c1c2_composed_p01_mirror_x_s20261027 |
| C2_U_p01 tier U p01 (init_s=None) | exp/CAND/runs/HSR/tierU_all_p01_s20261027_vps/checkpoints | $VPS_ROOT\uhwr\runs\CAND\HSR\tierU_all_p01_s20261027\checkpoints | exp/CAND/preds/tierU_all_p01_s20261027_4090 |
| C2_U_selfattn_p10 pathway (init_s=None) | exp/CAND/runs/HSR/tierU_selfattn_p10_s20261027_rec_vps/checkpoints | $VPS_ROOT\uhwr\runs\CAND\HSR\tierU_selfattn_p10_s20261027_rec\checkpoints | exp/CAND/preds/tierU_selfattn_p10_s20261027_rec_4090 |
| C1_p01_FR frame+replay | exp/CAND/runs/C1/c1_p01_mirror_x_replay_b200/checkpoints | — | exp/CAND/preds/c1_p01_mirror_x_replay_b200_s4090 |
| C1_p01_FO frame only | exp/CAND/runs/C1/c1_p01_mirror_x_noreplay_b200/checkpoints | — | exp/CAND/preds/c1_p01_mirror_x_noreplay_b200_s4090 |
| C1_p10_FR frame+replay p10 | exp/CAND/runs/C1/c1_p10_mirror_x_replay_b200/checkpoints | — | exp/CAND/preds/c1_p10_mirror_x_replay_b200_s4090 |
| CTRL_s1 LUO-FT p01 (comparator seed) | exp/CAND/controls/runs/ctrl_luoft_p01_s1_b200/checkpoints | — | exp/CAND/preds/ctrl_luoft_p01_s1_b200/preds |
| CTRL_s2 LUO-FT p01 (comparator seed) | exp/CAND/controls/runs/ctrl_luoft_p01_s2_b200/checkpoints | — | exp/CAND/preds/ctrl_luoft_p01_s2_b200/preds |
| CTRL_cont10k LUO +10k no Urdu | exp/CAND/controls/runs/ctrl_luo_cont10k_nourdu_b200/checkpoints | — | exp/CAND/preds/ctrl_luo_cont10k_nourdu_b200/preds |
| C2_S tier S shared LoRA 30k | exp/CAND/runs/HSR/tierS_all_r32/checkpoints | — | exp/CAND/preds/tierS_all_r32_4090 |
| C2_U_p10 tier U p10 (init from tier S ckpt30) | exp/CAND/runs/HSR/tierU_all_p10_s20261027_vps/checkpoints | — | exp/CAND/preds/tierU_all_p10_s20261027_4090 |
| C2_U_p30 tier U p30 (init from tier S ckpt30) | exp/CAND/runs/HSR/tierU_all_p30_s20261027_vps/checkpoints | — | exp/CAND/preds/tierU_all_p30_s20261027_4090 |
| C3_p01_q25 replay-only (killed at K3, 4k) | exp/CAND/runs/c3_p01_identity_replay_clusterA_a40/checkpoints | — | exp/CAND/runs/c3_p01_identity_replay_clusterA_a40/eval |
| RA_FT_p01 Route A LUO-FT p01 (comparator seed) | exp/M5/runs/ft/p01/checkpoints | — | exp/D3_ROUTE/routeA/preds |
| RA_FT_p10 Route A LUO-FT p10 (reference) | exp/M5/runs/ft/p10/checkpoints | — | exp/D3_ROUTE/routeA/preds |
| LUO_ms6 backbone | exp/M5/runs/luo/checkpoints | — | exp/D3_ROUTE/routeA/preds ; K1: exp/CAND/K1/preds |
| E0 ms8 OUHD-L specialist | runs/e0/e0_v3_rendered_100k_bs512/checkpoints | — | (E0 eval dir; not CAND) |

CLUSTER-A root for every relative path: $U/. VPS root: $VPS_ROOT\uhwr\.

## 3. Comparator seeds (LUO-FT p01, identity frame, no replay)

| alias | train seed | code path | train job | selected ms | sha256 | preds sampled on |
|---|---|---|---|---|---|---|
| CTRL_s1 | 20261017 | exp/CAND controls code (frame_code_a44e78b0 frame_arm.sh + ctrl_sample.py) | clusterA:24688007 (B200) | 3 | `cd60f5c391b808de…` | RTX 4090 (VPS) final sets; B200 copies also scored |
| CTRL_s2 | 20261018 | exp/CAND controls code (frame_code_a44e78b0 frame_arm.sh + ctrl_sample.py) | clusterA:24688009 (B200) | 4 | `1ba72ad4317c9e51…` | RTX 4090 (VPS) final sets; B200 copies also scored |
| RA_FT_p01 | 20261016 | Route A (M5 ft code); not CAND code | clusterA:24517796 (M5 ft p01) | 4 | `d770109bb9fcccb3…` | RTX 4090 (VPS) final sets; B200 copies also scored |

- Pools used in readout v2: COMP_p01_pool3 = {CTRL_s1, CTRL_s2, RA_FT_p01}; COMP_p01_cand2 = {CTRL_s1, CTRL_s2} (cand_tables.py POOLS).
- Fresh CAND-code comparator seeds 1-3 (COMP-p01) do not exist yet; they are a Phase 1 FewShot item.

## 4. Preds and per-unit rows

- Final-set preds of every landed arm: exp/CAND/preds/<arm>/ (12 npz per arm: test_ouhdl and gold_ouhdl seeds 0-2, six non-Urdu sets seed 0), each with a .json sidecar; readout v2 preds_manifest.csv lists 328 files, sha_match True for all rows with status OK.
- val1000 milestone preds: exp/CAND/runs/<arm>/eval/preds and exp/CAND/controls/runs/<ctrl>/eval/preds.
- K1 frame preds (LUO ms6 zero-shot, identity / mirror_x / VT): exp/CAND/K1/preds, 24 npz hashed (INVENTORY_k1_routeA_preds_sha256_2026-10-29.csv); sets test_ouhdl seeds 0-2, val250_ouhdl, val250_iam.
- Route A preds (LUO ms6, FT p01/p10/p30/p100): exp/D3_ROUTE/routeA/preds, 69 npz hashed; Route A per-unit rows: exp/D3_ROUTE/routeA/score_out/rows.
- CAND per-unit rows (uhwr_eval 0.3.1, rows031): exp/CAND/readout/score_out/{4090,b200,a40}/rows and exp/CAND/K1/score/rows; 309 rows files, 1,006,532 rows (PROVENANCE_cand_tables.json, slurm 24719350).
- ScriptCNN router: exp/D3_ROUTE/router/script_clf.pt, sha256 22bb8802f2da46fc37972eec430a1439ad830905712f22f6d923c38829db45aa (thr_rtl 0.5 in the composed arm sidecar).

## 5. Facts relevant to reuse

- The composed arm (VPS 531) checkpoints exist only on the VPS (10 files, 64.5 MB each); exp/CAND/runs/C1/c1c2_composed_p01_mirror_x_s20261027 on CLUSTER-A holds eval/ only.
- tier U p01, composed and the selfattn pathway arm were trained with init_s=None (no shared tier S adapter); tier U p10/p30 are initialised from tier S checkpoint_000030 (readout v2 §12(k); memory mem_b4eee3ee43fd).
- tierU_all_p10/p30_*_a40 and tierU_selfattn_p10_*_b200 hold only ms1-2 (race losers / preempted); the _vps run dirs are the survivors.
- CTRL_s1, CTRL_s2, cont10k: ms1-10 all present on CLUSTER-A (ms3-10 requested by the plan for step-matched replay readings).
- C1 FR/FO/p10 ms1-10 present on CLUSTER-A (B200 training, 24688253/54/55).
- T_ALL_identity_s0 / T_ALL_mirror_x_s0 checkpoints ms1-10 exist; T_ALL is not landed (not part of readout v2).
- exp/D3_ROUTE/router also contains a macOS AppleDouble file ._script_clf.pt (not a checkpoint).

## 6. Files
- INVENTORY_ckpt_sha256_2026-10-29.csv — every checkpoint hashed (CLUSTER-A 167 + VPS 30 rows): dir, file, sha256, bytes, host.
- INVENTORY_k1_routeA_preds_sha256_2026-10-29.csv — 93 K1 / Route A preds npz sha256.
