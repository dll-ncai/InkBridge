# HANDOFF — FMDX backbone lane monitor (track 'Backbone build', written 2026-11-03 ~17:05Z)
Facts only. CLUSTER-A root R=$U; CLUSTER-B root N=$U_B (pyxis mounts N at R); VPS stage $VPS_ROOT\uhwr\stage (= hub).
X = R/exp/FMD/xs. Authoritative live tables: X/admin/arm_queue.csv (CLUSTER-A job ids, state; controller-maintained), CLUSTER-B N/exp/FMD/xs/admin/clusterB_submissions.tsv
(mirror X/admin/clusterB_submissions_mirror.tsv sha256 cbe3838b…), X/admin/priority.json (rank), X/admin/controller_log.md, arm_log.md, cancellation_log.md,
CLUSTER-B admin/arm_log_clusterB.md, cancellation_log_clusterB.md, controller_log_clusterB.md. Cross-track log: R/exp/CAND/queue/cand_log.md.

## 1. Code / records
- Trainer code: X/code/fmdx_backbone_v1 (frozen; SHA256SUMS.txt sha256 96149ca7aee95bdd68a2545f4f1d6d74d028fb0de20ddf5b561bdf35fe89d9ad; tar 2c49ecdc…). Same on CLUSTER-B (CODE_SHAOK) and VPS $VPS_ROOT\uhwr\code\fmdx_backbone_v1.
- Controller CLUSTER-A: X/code/fmdx_controller_v1.2 (gate_controller.py bd3365c3…; prereg A7: --cap 14 --lane-wide over ALL running GPU jobs of the account; PASS*/control WARM arms reserve GPUs against SCRATCH only), job 24846629 (until 2026-11-10T23:59Z). CLUSTER-B side still v1.1 (clusterB sbatch 5485b502…, --cap 8 fx_ GPUs). CLUSTER-B job 3700469 (A100-RP,H100-RP CPU-only, PENDING at 17:00Z).
- Pseudo job fix: X/code/fmdx_pseudo_jobs_v1.0.1. Ad-hoc job files: X/code/_jobs (VPS cmds, waiters, CPU targets, audit), CLUSTER-B N/exp/FMD/xs/code/_jobs/clusterB_tgt.
- Decision record: X/runs/DECISION_RECORD_backbone.md (frozen 42dab0b6…) + addendum_1 (322d171e…) + addendum_2 (ef00320c…) + addendum_3 (4eeac936…).
- Prereg: X/plans/FMD_PREREG_2026-11-03.md current sha256 b5af7ee97706d4a7c8d5c880c7612b769712e9d1a41de4808ee4158e95fd04c8 (A7 cap 14; pre-A7 b84412d8…; CLUSTER-B + VPS copies sha-verified) (A6 3c1beaa8…, A3.a aa9c6c7b…). CAPACITY_NOTE.md 58b7833f….

## 2. Arms and lanes
- 132 arms (arms.tsv f90f9b02…): 66 WARM (40k steps, save 10k) + 66 SCRATCH (100k steps, save 10k). CLUSTER-A: one Slurm job per arm, name fx_<arm>. CLUSTER-B: 102 race copies (no DINOv3-family). VPS: cmds 440 (dinov3-vitl16_W) and 441–446 (C-RADIO WARM).
- Race lock: atomic mkdir of VPS stage\exp\FMD\xs\race\<arm> (owner.txt = lane). The first copy to start trains. Later copies write RACE_LOST.json and exit 0.
- Cancelled (A4.3a, verdict FAIL): 10 FMs from controller v1.0 + lightonocr-2-1b (CLUSTER-A 24843771/24843845; CLUSTER-B 3700397/3700449). BLOCKED/held: sd3-medium-diffusers, flux1-dev, flux1-schnell, sam3, inkfm (both regimes). flux1-dev = non-commercial licence; checkpoints stay on CLUSTER-A/CLUSTER-B/VPS.
- Running at 17:00Z:
  - NOFM_W CLUSTER-A 24843707: 2×A4000 <node>, step ~3200 at 16:33Z, 1.427 s/step, expected end ~07:10Z 4 Nov plus in-job selection.
  - dinov3-vitl16_W VPS 440: start 15:52:11Z, 2.39 s/step, expected end ~18:30Z 4 Nov plus val250 sampling.
- PASS_XS WARM arms (rank by priority_score): c-radiov3-h > c-radiov3-g > c-radiov4-h > c-radiov3-l > c-radiov4-so400m > c-radiov3-b.

| arm | CLUSTER-B copy | CLUSTER-A copy (2 GPU) | VPS cmd | CLUSTER-B extraction job | SCRATCH CLUSTER-A/CLUSTER-B |
|---|---|---|---|---|---|
| c-radiov3-h_W | 3700361 | 24843729 | 441 | 3700746 | 24843809 / 3700412 |
| c-radiov3-g_W | 3700362 | 24843730 | 442 | 3700747 | 24843810 / 3700413 |
| c-radiov4-h_W | 3700364 | 24843732 | 443 | 3700748 | 24843812 / 3700415 |
| c-radiov3-l_W | 3700360 | 24843728 | 444 | 3700646 | 24843808 / 3700411 |
| c-radiov4-so400m_W | 3700363 | 24843731 | 445 | 3700749 | 24843811 / 3700414 |
| c-radiov3-b_W | 3700359 | 24843727 | 446 | 3700646 | 24843807 / 3700410 |

- Controls: NOFM_W CLUSTER-A 24843707 (running) / CLUSTER-B 3700349 (held). PEINIT_W CLUSTER-A 24843710 (pending) / CLUSTER-B 3700351 (released, nice 90).
- Other top-4 warm: dinov3-vitl16_W CLUSTER-A 24843708 (2 GPU), VPS 440. cleandift-sd21_W CLUSTER-A 24843709 (2 GPU) / CLUSTER-B 3700350.

## 3. REPA targets (S40k = 40,000 units, positions X/data/backbone/repa_subset_S40k_positions.npy sha 7ec5b08d…)
- Files: X/targets/repa/<fm>/repa_<fm>_gate_G16__S40k.npy (+ __unit_ids.npy, .npy.meta.json, .DONE.json with the sha256).
- DONE: dinov3-vitl16 (bc161a92…, 15:17Z), c-radiov3-b (d14c54cb…, 16:32Z, CPU).
- Running: c-radiov3-l re-merge 24847054 (first merge EBUSY at os.replace); merges of v3-h/v3-g/v4-h/v4-so400m (24846255); cleandift-sd21 and other FMs in main GPU array 24843687 (throttle 1).
- CLUSTER-A CPU array 24846255: c-radiov3-h/g, c-radiov4-h/so400m. Chunk-0 + verdict (+ weights for v3-h, v4-so400m) are pushed to the hub by refpush 24846256; pulled on CLUSTER-B by ONE sequential tmux stream (session fmdx_pull, code/_jobs/clusterB_tgt/pull_seq_1.sh; .PULLED markers). Hub rule: one concurrent stream per track.
- CLUSTER-B-side extraction jobs (section 2 table) write X/targets/repa/<fm>/ on CLUSTER-B plus chunk0_check_vs_clusterA.json. Pass rule: cos_min ≥ 0.999 and maxabs ≤ 0.01·max|ref|. PASS → scontrol release of the CLUSTER-B WARM copy (logged in arm_log_clusterB.md). FAIL → copy stays held; fallback = hub relay pull: `bash N/exp/FMD/xs/code/fmdx_backbone_v1/jobs/hub_pull.sh tgt_<fm> exp/FMD/xs/targets/repa/<fm>` on CLUSTER-B login1, after removing the fm entry from N/exp/FMD/xs/admin/controller_state_clusterB.json (pre-seeded so the CLUSTER-B controller does not overwrite CLUSTER-B-extracted files).
- CLUSTER-A controller relay pushed c-radiov3-b to the hub at 16:33Z (tgt_c-radiov3-b). Check controller_log.md for rc.
- VPS: waiter v3 24846288 (begins 18:30Z, sequential by rank). It pushes final target files only when the VPS sha differs, verifies the VPS sha, inserts the load-time sha gate (exit 34 on mismatch) into the cmd, and enqueues 441–446.
- Incident 15:48–16:23Z (foreign pushes into the VPS targets dir): audit 24846202 found all dinov3 files MATCH. The dinov3_W load-time sha (run_config.json) = bc161a92…. No restart.

## 4. Expected outputs per arm run dir (<arm>_<lane>)
- CLUSTER-A X/runs/<arm>_clusterA, CLUSTER-B N/exp/FMD/xs/runs/<arm>_clusterB, VPS stage\exp\FMD\xs\runs\<arm>_vps.
- Contents: run_config.json (inputs and their sha), train_log.jsonl, latest.pt every 1000 steps (resume), bb_ms<k>.pt (fmdx_bb/1, EMA, projector discarded) every 10k, training_complete.json, GATE_FAIL / RACE_LOST.json if applicable, eval/val__<set>__ms<k>_preds.npz/.json (val250 × 4 LUO sets, seed 0, DDIM-50, eta 0).
- CLUSTER-A and CLUSTER-B also write eval/selection.json + best_milestone.txt (m5_select: lowest mean native LDTW over the 4 sets).
- VPS runs: pull the run dir to CLUSTER-A (sha check), then score + select on CLUSTER-A with the same m5_select rule. e0_score ~7 min per call, so run it as an array.

## 5. After a WARM arm finishes
- FT (prereg A4, COMP-p01 recipe), 1 GPU 128×4 (≥40 GB) or the A6 layout for the GPU:
  `bb_train.py --regime ft --init-ckpt <run>/bb_ms<best>.pt --train-npz <R>/data/.../m5_ft_ouhdl_p01.npz --render-npy X/data/backbone/p01train_render224.npy --train-steps 10000 --save-every 1000 --seed 20261034 --enc-lr 8e-5 --enc-warmup 0` (E0 rule on val1000 selects the FT milestone).
- Sampling: bb_sample.py (FM-free, e0_sample format, sidecar .json with sha). Test sets / seeds per prereg.

## 6. Pending / watch list
1. CLUSTER-B A100-RP slots. Earliest start estimate 18:23Z (sbatch --test-only at 16:32Z); it slipped once.
2. CLUSTER-A cap: 14 lane-wide (prereg A7, controller v1.2). Do not start SCRATCH arms by hand; v1.2 releases them after the higher-ranked WARM arms.
3. InkSight pseudo pipeline v1.0.1:
   - Pool 24844987 COMPLETED: 56,121 images, against 20,000 per source in A4.2 (casia_zh 20,000; casia_en 8,699 = whole pool; hpl_tamil 7,422 of 20,000 drawn, 12,578 rebuild exceptions; iam 20,000). Reported to the lead.
   - Cause of the Tamil failures: macOS AppleDouble '._*.txt' files (tamil_diag.json). Redraw 24847074 → InkSight extra array 24847075 → build v1.0.1 24847076 (afterany 24844988 + 24847075). Check data/pseudo/pseudo_pool_hpl_tamil_redraw.json.
   - INKSIGHT_W/S become eligible when data/pseudo/pseudo_train.npz.meta.json exists.
4. Gated FMs (sd3/flux/sam3/sd3.5) stay BLOCKED until weights exist and the verdict is not BLOCKED. Then: update roster, make_arms, submit_arms --only, add target tasks.
5. Not-assessed A3.a.2 FMs: no arms unless an extractor exists before 2026-11-05T12:00Z.
6. Expected first full U/X/F read: ~4 Nov 15:00–20:00Z if FT and test sampling run on CLUSTER-B/CLUSTER-A, or ~5 Nov 06:00–12:00Z if they wait for the 4090 (estimates; CLUSTER-B s/step not measured).

## 7. Update 19:45Z (addendum_5)
- New top-ranked PASS_XS arms:
  - vit-mae-large: CLUSTER-A WARM 24843748 (2 GPU), CLUSTER-B WARM 3700378, CLUSTER-B extraction 3701232.
  - webssl-mae300m-224: CLUSTER-A WARM 24843747 (2 GPU), CLUSTER-B WARM 3700377, CLUSTER-B extraction 3701233.
  - dinov3-vitb16: CLUSTER-A WARM 24843719 only (2 GPU).
  - S40k targets for all three are DONE on CLUSTER-A (19:23–19:31Z).
- VPS: trains nothing beyond 440 (lead addendum_4). Waiter v3 is cancelled; cmds 441–446 are never enqueued.
- Target dispatcher 24850228 (fmdx_target_dispatch_v1) extracts targets for every PASS* FM in priority.json order. Log: admin/target_dispatch_log.md; state: admin/target_dispatch_state.json.
- CLUSTER-B side: tmux session fmdx_extract_loop (code/_jobs/clusterB_tgt/clusterB_extract_loop.sh; log logs/targets/clusterB_extract_loop.log). If login1 is rebooted, restart it with `tmux new-session -d -s fmdx_extract_loop 'bash <path>'`.

## 2026-11-05T14:14:57Z addendum_8
- WARM requeue after <node> failures: 7 encoder arms CLUSTER-B (3709983,84,89,88,94,97,98), 4 CLUSTER-A (24904693-96). Controller v1.6 job 24904697 (auto-resubmit <=2). clusterB_arm.sbatch v1.0.3. FLUX/SD3 targets 24866395 failed (blocked). See DECISION_RECORD_backbone_addendum_8.md.
