# DECISION_RECORD_backbone — addendum 8 (2026-11-05, ~14:00–14:20Z): requeue of WARM arms after <node> failures; controller v1.6; clusterB_arm.sbatch v1.0.3
Track: Backbone build (FMDX). Times UTC unless marked CET (Slurm estimates on both clusters print CET).

## J1 Incident (facts) [reason: lead request, 9+ WARM jobs FAILED]
- CLUSTER-A sacct 4 Nov 19:00–21:30Z, all on llms node <node> (RTX PRO 6000 Blackwell): 11 WARM jobs FAILED exit 7 (training crash; lead: CUDA illegal memory access in 2-GPU DDP) — cleandift-sd21_W 24843709, c-radiov3-b_W 24843727, c-radiov3-g_W 24843730, c-radiov3-l_W 24843728, c-radiov4-h_W 24843732, c-radiov4-so400m_W 24843731, dinov3-vitb16_W 24843719, pe-spatial-b16-512_W 24843735 (1 GPU), qwen3.5-4b_W 24843761 (1 GPU), vit-mae-large_W 24843748, webssl-mae300m-224_W 24843747.
- PEINIT_W 24843710 FAILED exit 8 (VAL_SAMPLING_FAILED, <node>, 18:16Z). Training was complete (step 40000); milestone 4 selected by the monitor's sel_job 24877089 at 18:53Z; arm_queue state DONE. Not requeued.
- CLUSTER-A latest.pt at failure: vit-mae-large_W step 19250 (863,651,292 B), webssl-mae300m-224_W step 12850 (863,651,356 B); the other 9 had no checkpoint (step 0).

## J2 Lane estimates, 5 Nov ~14:00Z (sbatch --test-only, one job)
- CLUSTER-A gpuidle,b200idle on cc>=8 nodes, excluding <node>: 2 GPU 2026-11-06T17:13 CET; 1 GPU 2026-11-06T13:57 CET. llms 1 GPU excluding <node>: 2026-11-29 (llms = <node> + <node> only).
- CLUSTER-B 2 GPU: A100-SDS 2026-11-06T13:19 CET; H100-RP 2026-11-07T06:20 CET; A100-RP 2026-11-14; RTXB6000 2026-11-20. V100-32GB-SDS (2026-11-05 15:00 CET) and batch (RTX6000) are compute capability <8 -> arm.sh exits 4 (GPU_CC_LT_8); not usable.
- CLUSTER-B->CLUSTER-A route other than VPS: none. CLUSTER-A<->CLUSTER-B ssh is closed both ways (task brief); CLUSTER-A ~/.ssh/authorized_keys lists a key '<author>@clusterB-clusterA' but CLUSTER-B ~/.ssh holds only id_ed25519 (GitHub key) and uhwr_staging_vps_ed25519 (VPS key, VPS off); no test connection was made.
- Checkpoint sizes for any relay (NOFM_W_clusterA): bb_ms1..4.pt 389,091,621 B each (1.56 GB per arm), checkpoints/latest.pt 778,649,092 B.

## J3 Decision: lanes per arm [reason: earliest start; static lanes; licence/targets]
| arm | lane | job | start (est.) | from step |
|---|---|---|---|---|
| vit-mae-large_W | CLUSTER-B A100-SDS,H100-RP,A100-RP | 3709983 | A100-SDS est. 6 Nov 13:19 CET | 0 (CLUSTER-A 19250 not transferable) |
| webssl-mae300m-224_W | CLUSTER-B | 3709984 | after rank 1 | 0 (CLUSTER-A 12850 not transferable) |
| c-radiov3-g_W | CLUSTER-B | 3709989 | queue | 0 |
| c-radiov4-h_W | CLUSTER-B | 3709988 | queue | 0 |
| c-radiov3-l_W | CLUSTER-B | 3709994 | queue | 0 |
| c-radiov4-so400m_W | CLUSTER-B | 3709997 | queue | 0 |
| c-radiov3-b_W | CLUSTER-B | 3709998 | queue | 0 |
| dinov3-vitb16_W | CLUSTER-A gpuidle,b200idle 2 GPU | 24904693 | 7 Nov 10:18 CET (squeue) | 0 |
| cleandift-sd21_W | CLUSTER-A | 24904694 | 7 Nov 20:35 CET | 0 |
| pe-spatial-b16-512_W | CLUSTER-A | 24904695 | 9 Nov 15:13 CET | 0 |
| qwen3.5-4b_W | CLUSTER-A | 24904696 | 9 Nov 17:29 CET | 0 |
- CLUSTER-B arms: S40k targets on CLUSTER-B DONE with chunk-0 check pass=True for all 7 FMs (CLUSTER-B extraction jobs 3705254-61 COMPLETED). WARM init LUO checkpoint_000006.pt present on CLUSTER-B (434,608,395 B).
- CLUSTER-A-only: dinov3-vitb16 (DINOv3 licence, CLUSTER-A only); cleandift-sd21, pe-spatial-b16-512, qwen3.5-4b (S40k targets exist on CLUSTER-A only).
- A6 layouts (512 units/step): CLUSTER-B A100-40GB/A100-80GB/H100 (>=40 GB) -> 64/proc x 2 GPU x accum 4; CLUSTER-A A4000/A30/A40 etc. -> arm.sh rule (<40 GB: 32/proc x 2 x 8). No llms 1-GPU runs were used.
- arm_queue.csv: 7 rows PINNED_CLUSTER-B (clusterA_job cleared, clusterB_job set); 4 rows SUBMITTED with new clusterA_job. Backup admin/arm_queue.csv.pre_add8. Every change logged in admin/arm_log.md.

## J4 Failed attempts during the requeue (own sections) [reason: facts]
- CLUSTER-B 3709959-65 (submitted with clusterB_arm.sbatch default --mem=200G; V100 nodes had 79-111 GB free) -> cancelled, resubmitted --mem=64G (CLUSTER-A MaxRSS 4.9-5.5 GB; targets are memory-mapped).
- CLUSTER-B 3709973-79 started on V100-32GB-SDS (garda) and exited 1 within 1 s: clusterB_arm.sbatch sources common.sh on the host, which sets X=$SCRATCH_ROOT/... (host race_lock: "mkdir: cannot create directory '$SCRATCH_ROOT'", race_lock ERR) and IMG=$SCRATCH_ROOT/.../torch27.sif (pyxis: image not found). The continuation segments 3709983-98 were moved to A100-SDS,H100-RP,A100-RP, MinMemoryNode=64G.
- CLUSTER-A 24904414-17 FAILED exit 4 (GPU_CC_LT_8) after 3 s on <node>-122 (GTX 1070): the explicit --exclude=<node> replaced clusterA_arm.sbatch's own #SBATCH --exclude list. Resubmitted as 24904693-96 with --exclude=dgx[001-005],gpu[004-005,008,013-014,017-023,119-148].
- requeue_arms.py first clusterB-mode write to arm_queue.csv was overwritten by a concurrent controller v1.5 write; re-applied after the controller was stopped.

## J5 Code changes
- clusterB_arm.sbatch v1.0.3: after `source common.sh` one line resets X=$N/exp/FMD/xs and IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh. sha256 d2eeb0e39a512c497045fa8cd9db167e85c8933ecd2998c2d18df8ad20db8f8a; SHA256SUMS.txt f74d9e75da78e39cf2c5c03b4b3f90c14ff1ae9db661ed2d0b8366a272383fb8 (36/36 OK on CLUSTER-B). Originals: jobs/_pre_clusterBpath/ (both clusters). Also applies to the pending CLUSTER-B SCRATCH arm segments.
- Controller v1.6 (X/code/fmdx_controller_v1.6/gate_controller.py sha256 f7c82196eaa91640f9b2a535c6fe0ac4c4db6b1e8e1bea698e28d30a73d0b4d0; diff vs v1.5 controller_v1.5_to_v1.6.diff): an CLUSTER-A arm with state RUNNING/SUBMITTED, no live Slurm job, and last job sacct state FAILED/NODE_FAIL/OUT_OF_MEMORY/TIMEOUT/BOOT_FAIL/PREEMPTED, no GATE_FAIL marker -> resubmitted held (nice 100+100*arms.tsv position; normal cap/priority release), same GPU count, -p gpuidle,b200idle, --exclude = base list + every node this arm failed on; at most 2 per arm, then state FAILED_MAX_RESUBMIT. Logged in admin/resubmit_log.md, controller_log.md, cand_log; counts in controller_state_clusterA.json 'resubmits'. CLUSTER-A job 24904697 (v1.5 job 24875898 cancelled). Not covered: CLUSTER-B arms (no CLUSTER-B controller runs; clusterB_arm.sbatch continuation segments resubmit each arm until done).
- requeue_arms.py (X/code/fmdx_requeue_v1, sha256 77e163b07a2771c735eec2cd88a7dac62dfe34eb48898085f6c6bd2e2903a8bb).
- Pending CLUSTER-A fx_ jobs: ExcNodeList now includes <node>; the 6 pending 2-GPU jobs that listed llms -> Partition=gpuidle,b200idle; 3 pending 1-GPU jobs keep gpuidle,llms,b200idle.
- SD3.5 target array 24878952: pending tasks ExcNodeList=<node>; task 26 was already running on <node> (not cancelled).

## J6 Blocked (found during this check) [reason: facts]
- FLUX/SD3 target array 24866395: 33 FAILED, 3 COMPLETED. flux1-schnell and flux1-dev tasks ended rc 1 between 4 Nov 12:46 and 13:51Z with no traceback in the .out logs; sd3-medium-diffusers 3 of 4 shards COMPLETED, 1 FAILED (<node>), 153/157 chunks present, no merged target. Dispatcher v1.1 retries fail with "sbatch: error: Invalid job array specification" (0 tasks computed). flux1-schnell_W, flux1-dev_W and sd3-medium_W have no S40k targets.

## J7 CLUSTER-B arms: backbone selection + Urdu p01 FT + relay manifest on CLUSTER-B [reason: lead request 5 Nov]
- Backbone selection (A4.3(d), val250 LUO rule) already runs inside arm.sh on the training lane (CLUSTER-B for CLUSTER-B arms).
- Inputs verified on CLUSTER-B against CLUSTER-A, sha256 identical: m5_ft_ouhdl_p01.npz (2,086,428 B), p01train_render224.npy (16,859,264 B), m5_val_{casia_zh,casia_en,hpl_tamil,iam}.npz, val250_{same}_render224.npz, fmd_render224_val1000.npz (3,103,722 B), e0_val.npz (16,777,421 B), e0_score.py, m5_select.py; Python files of uhwr_eval-0.3.1, nagamatsu/src, fmd_harness_v1 and e0/code_4cd40f57 identical. venv e0 python -> /opt/conda/bin/python (in the container).
- Missing on CLUSTER-B: only the warm monitor's ft_job.sh (4,570 B). It is now copied unchanged (sha256 92aa9831d32140a3...) to CLUSTER-B exp/FMD/xs/code/fmdx_monitor_v1/ft_job.sh.
- New code X/code/fmdx_clusterB_post_v1 (both clusters): clusterB_ft.sbatch (sha256 8b0ce7e127e8dc23c273fa356a91d3e91f95eec0be43854256c36b52832f9a5f; 1 GPU, A100-SDS,H100-RP,A100-RP, 64G, 1-day segments with continuation) runs ft_job.sh <ARM> clusterB <run>/bb_ms<best>.pt 20261034 (A4.3(e): 10,000 steps, save 1,000, E0 val1000 selection; FT run dir runs/fmdx_monitor/ft/<ARM>_FT_clusterB), then relay_manifest.sh (sha256 b210a356be0217578255...) which writes runs/<ARM>_clusterB/RELAY.json with ONLY the selected bb_ms and the selected FT bb_ms (path, CLUSTER-B path, size, sha256, selection.json + sha256, parts with per-part sha256, reassemble command) and splits each file with split -b 200M -a 2 into <file>.part_aa, .part_ab, ... next to it.
- Trigger: clusterB_arm.sbatch v1.0.4 (sha256 85c6a9085933fe247ebd4975ed49f4d3e8f170d55eeb45d59c06ffe74954613a; BB SHA256SUMS.txt 3b7457681fe3746bd71e45078ea20a4360fbf1392d50cf309054e3b7719c1805, 36/36 OK on both clusters) submits fxft_<ARM> once (marker <run>/.ft_submitted) as soon as eval/best_milestone.txt exists. This applies to every CLUSTER-B arm (WARM and SCRATCH; prereg A4.2: every arm gets the Urdu p01 FT).
- Slurm stores the batch script at submission, so all 17 pending CLUSTER-B fx_ segments (they carried the pre-v1.0.3 script) were cancelled and resubmitted with v1.0.4, keeping name/nice/partition/mem/GPUs: WARM vit-mae-large 3710045, webssl 3710046, c-radiov3-g 3710047, c-radiov4-h 3710048, c-radiov3-l 3710049, c-radiov4-so400m 3710050, c-radiov3-b 3710051; SCRATCH NOFM 3710052, PEINIT 3710053, vit-mae-large 3710054, webssl 3710055, c-radiov3-h 3710056, c-radiov3-g 3710057, c-radiov4-h 3710058, c-radiov3-l 3710059, c-radiov4-so400m 3710060, c-radiov3-b 3710061. arm_queue.csv clusterB_job updated; arm_log.md lines written.

## J8 FLUX/SD3 targets: cause and fix [reason: lead request]
- Cause (from code/_jobs/slurm-24866395_<task>.err): torch.OutOfMemoryError while placing the transformer on 16 GB A4000s. fm_registry_dit.json had no entries for the DiT FMs, so kw={} and the extractor placed all blocks on the GPU (no vram_budget_gb streaming); the screen had used micro=1, vram_budget_gb=10.5 on A4000 (placement record in xs/features/<fm>/ouhdl_val1000/*.meta.json). The SD3.5 array 24878952 had the same failure on <node> (A30 MIG slices, 11.75 GB) and "AssertionError: Invalid device id" on <node>; 13 SD3.5 tasks completed on larger GPUs.
- Fix: fm_registry_dit.json (sha256 8f021a10aa1de64f..., backup fm_registry_dit.json.pre_add8) entries for flux1-schnell, flux1-dev, sd3-medium-diffusers, sd3.5-large/-large-turbo/-medium: kw micro=1, vram_budget_gb=10.5 (= screen placement; extractor docstring: streaming changes only tensor residence). Exclude list for DiT jobs adds <node> and <node> to the clusterA_targets_dit.sbatch default list.
- Chunk state before resubmission (157 chunks per FM): flux1-schnell 0, flux1-dev 0, sd3-medium 153 (shard 3 missing), sd3.5-large 50 (shards 0,1,4-8,12-15 missing), sd3.5-large-turbo 95 (shards 11-15 missing; 9,10 running as 24878952_26/27), sd3.5-medium 0.
- Resubmitted in rank order as one array: CLUSTER-A 24904943, --array=1-53%6, -p gpuidle, tasks file code/_jobs/targets_tasks_dit_r2.txt (sha256 9c8de5d696df6c50...): flux1-schnell 16 shards, flux1-dev 16, sd3-medium shard 3, sd3.5-large 11, sd3.5-large-turbo 5, sd3.5-medium 4. Pending SD3.5 tasks 24878952_[28-36] cancelled (their shards are in the new array). Slurm estimate for the first task: 2026-11-07T20:35 CET.
- Cost at the screen rate (12.95 s/unit, A4000): 2,500 units per FLUX shard ≈ 9 GPU-h; 32 FLUX shards at %6 ≈ 6 waves.
- Target dispatcher v1.1 (24878954) keeps running for dinov3-vit7b16. Its state shows attempts 3 for flux1-schnell, flux1-dev and sd3-medium, so it does not resubmit them. SD3.5 FMs (attempts 1) may receive failing dispatcher attempts after 24878952 ends; those attempts are logged in target_dispatch_log.md.

## J9 Arms for three FMs missing from arm_queue (lead 5 Nov gate read) [reason: lead request]
- FMs (verdict status, priority_score, family, gate layer): webssl-mae1b-224 PASS_XS 3051221521 encoder last; dinov3-vitl16-sat493m PASS_XS 3041090565 encoder last (DINOv3 licence, CLUSTER-A only); chandra-ocr-2 PASS_URDU_ONLY 1010956509 VLM last (non-promotable, exploratory).
- Roster copy admin/fm_roster_A3_add8.json (from fm_roster_A3_sd35.json + the three FMs; sha256 490f08bbedd7a8bc...). make_arms.py was run on a temp copy (/tmp/fmdx_mk4) and diffed: exactly 6 rows added (init_rank 70/71/72). Installed as admin/arms.tsv sha256 857380551865387bd324520e3db9fecb8c9471e79ad7784c56d5468fd76fed3d; previous copy arms.tsv.pre_add8_<sha8>. 6 arm_queue.csv rows (QUEUED) with arm_log.md lines. Controller v1.6 was stopped for the install (24904697) and restarted as CLUSTER-A job 24914909.
- Static lane CLUSTER-A for all six arms: S40k targets are extracted on CLUSTER-A only (DINOv3 licence for sat493m; no CLUSTER-A->CLUSTER-B transfer route for chunk-0 refs/weights with the VPS off).
- CLUSTER-A arm jobs (held; the controller releases them by priority once targets are DONE): WARM (2 GPU, gpuidle,b200idle, cc>=8 exclude list incl. <node>/<node>) webssl-mae1b-224_W 24914901, dinov3-vitl16-sat493m_W 24914902, chandra-ocr-2_W 24914903; SCRATCH (clusterA_arm.sbatch defaults) 24914904, 24914905, 24914906.
- Targets: fm_registry_dit.json entries layer=last, kw {} (same as the qwen3.5-4b S40k targets; the screen ran chandra-ocr-2 on CPU float32, the targets run on GPU with the extractor defaults) — registry sha256 14ddd9136a7705e3..., backup fm_registry_dit.json.pre_add8b. Arrays with clusterA_targets_dit.sbatch, -p gpuidle, same exclude list: 24914907 (targets_tasks_add8_enc.txt sha256 d2ccdfe8...: webssl-mae1b-224 2 shards, dinov3-vitl16-sat493m 1 shard; nice 0; Slurm estimate 6 Nov 08:01 CET) and 24914908 (targets_tasks_add8_chandra.txt sha256 e4bd5823...: chandra-ocr-2 2 shards; nice 200; estimate 7 Nov 00:40 CET). To keep rank order, the DiT target array 24904943 (lower priority_score than webssl-mae1b-224) was set to nice 100. Dispatcher state seeded (attempts 1, jobs = these arrays).
- A4.3(f): arms that cannot finish by 2026-11-10T18:00Z end 'not assessed'.
