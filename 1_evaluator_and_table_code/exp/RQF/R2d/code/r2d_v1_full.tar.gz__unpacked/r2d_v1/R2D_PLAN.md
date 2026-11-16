# RQF R2d — second held-out target script (HPL Tamil): execution plan, fixed before launch (7 Nov 2026)
Parents: PREREG_RQ_finalise_2026-11-07.md (sha256 04766e97...) section 0 + section 2 R2d; amendment A2 (sha256 bface42b..., A2.2 + A2.3 SM-R).
Executed by sub-agent model <agent>. Code: exp/RQF/R2d/code/r2d_v1 (SHA256SUMS.txt).
Nothing below changes a rule, margin, arm or seed of the parents; items marked (impl) are implementation choices the parents leave open.

## Data (PROVENANCE_<name>.json next to each file, exp/RQF/R2d/data)
- m5_lto_train.npz: m5_joint_train minus all 20,000 hpl_tamil units (casia_zh, casia_en, iam, ouhdl = 80,000; Urdu included). Non-Urdu units byte-checked against m5_luo_train.
- tamil_p01.npz: 1 % of the M5 Tamil train pool (49,156 units, 111 non-val users), writer-stratified (largest remainder), seed 20261107; K = 492 expected.
- tamil_val1000.npz (impl): 1,000 units drawn (seed 20261107) from the M5 Tamil val pool (6 held-out val users, 1,527 units; writer-disjoint from train and IWFHR06 test). Used ONLY for FT milestone selection.
- 224-px renders (FMDX A4.2 rule): lto_train_render224.npy (non-Urdu rows copied from luo_train_render224.npy, Urdu rows from OUHD-L GT ink), tamil_p01_render224.npy, tamil_val1000_render224.npz, val250_ouhdl_render224.npz.

## Arms (recipes copied, code executed unchanged)
- LTO: M5 LUO recipe (m5_train.py code_e33511dd): 100k steps, save 10k, batch 128 x accum 4, lr 8e-5, seed 20261016, 2 GPUs. Selection = LUO rule (lowest mean native LDTW, ties AIoU) on val250 of the training scripts {casia_zh, casia_en, iam, ouhdl} (impl: Tamil excluded as the held-out script, Urdu included because it is a training script — mirrors LUO, which excluded only its held-out script).
- LTO_W: FMD A4.2 WARM, NOFM (bb_train.py regime warm, vits14 224-px encoder, enc-lr 2e-4 warm-up 200, 40k steps, 512 sequences/step) init = LTO selected checkpoint; same selection rule/sets as LTO (bb_sample).
- LTO-FT p01: COMP-p01 recipe (frame_code e0_train.py, init LTO selected, frame identity, 10k steps, save 1k), seeds 20261034/35/36.
- LTO_W-FT p01: A4.3(e) FT (bb_train regime ft, enc-lr 8e-5, 10k steps, save 1k) init LTO_W selected bb_ms, seeds 20261034/35/36.
- SM-R (A2.3): single checkpoint, 10k steps, each batch 50 % tamil_p01 / 50 % m5_lto_train (uniform, seed = training seed). On LTO: e0_train --replay-npz m5_lto_train --urdu-share 0.5 (impl). On LTO_W: bb_train ft --pseudo-npz m5_lto_train --pseudo-frac 0.5 (fixed-share second source; impl). Seeds 20261034/35/36.
- FT/SM-R selection: E0 rule (lowest native LDTW) on tamil_val1000, seed 0, DDIM-50 eta 0, EMA. Test sets sealed until the run's own selection exists.

## In-job GATE_FAIL (impl; gate_watch.py)
Non-finite loss at any step, or median of the last 50 logged losses > 0.10 at step >= 5,000 (LTO) / 2,000 (LTO_W) / 1,000 (FT, SM-R). One gate line per milestone checkpoint in gate_log.txt. GATE_FAIL stops the run and its chain (exit 20 -> afterok dependents cancelled).

## Sampling and scoring
- 7 sets x sampling seeds 0-2: test_hpl_tamil (X4a 500), test_ouhdl, test_casia_zh, test_casia_en, test_iam500, german_unipen_de_test, german_onhw_all. LTO also test_hpl_tamil in mirror_x (zero-shot, descriptive).
- Scorer uhwr_eval 0.3.1 via routeA lane_score.py (vendored unchanged). CIs: writer-cluster percentile bootstrap B=2000 (unit bootstrap on OnHW), paired on the unit intersection, per-unit mean over training and sampling seeds. Device per preds file recorded; (M) tag where method and comparator devices differ.
- U-WIN' (prereg R2d, CONFIRMATORY): test_hpl_tamil, LTO_W-FT pooled vs LTO-FT pooled: AIoU lo > 0, pen-up F1 v2 lo > -0.0141, strict lo > -0.01065, direction lo > -0.01245.
- X-NI' (FMD A4.5 verbatim): composed LTO_W system vs LTO on each of the 6 other sets, AIoU lo > -0.010 on every set. Composition (impl, A4.4 analogue): ScriptCNN router script_clf.pt (sha 22bb8802), per-unit p_hpl_tamil >= 0.5 -> LTO_W-FT, else LTO_W backbone. Reported next to it: LTO_W backbone vs LTO, LTO_W-FT vs LTO-FT without router, LTO-FT composed vs LTO.
- A2.2 for SM-R arms: target test_hpl_tamil vs LTO-FT pooled (U-WIN clauses); other 6 sets vs LTO (LUO-equivalent) X-NI + order NI clauses reported; SOLVES = both.

## Lanes
- Copy A CLUSTER-A: CPU data -> CPU smoke -> afterok chain (LTO 2x A30 'gpu' partition, LTO_W 2x A30, FT/SM-R/sampling gpuidle A30 1-GPU, scoring CPU).
- Copy B CLUSTER-B: 1-day segments with continuation and trigger chain (same DAG); LTO waits for admin/SMOKE_PASS (uploaded after the CLUSTER-A smoke passed). 2-GPU H100-RP/H200-SDS/A100-SDS, 1-GPU H100-RP/A100-SDS/A100-RP/H200-SDS.
- Race: the first LTO copy to START wins (START lines in admin/r2d_events.log of each lane); the loser copy is cancelled with its chain (CLUSTER-A: scancel the LTO job, afterok dependents fall away; CLUSTER-B: admin/STOP_clusterB + scancel). No cross-cluster lock exists (CLUSTER-A has no route to the VPS lock hub), so the race is resolved by the monitor from the START times; both cancels logged in exp/RQF/queue/rqf_log.md.
- VPS 4090: not used for R2d (every R2d stage needs a lane-local backbone that does not exist yet).
- Freeze (A2.6): tables read at 10 Nov 18:00Z; anything not landed is 'not assessed'.
