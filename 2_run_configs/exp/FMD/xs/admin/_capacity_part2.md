## 4. Proposed fixed shared training subset for REPA targets

Ordering file: `exp/FMD/xs/data/luo_repa_order_v1.csv` (80,000 rows; sha256 + code sha in `.meta.json`; unit_ids sha256 `95b656b81594f9bcc7bdf72db1fc776c7409d0567e16dba2d461f3c4bae4a2e7`).  Rule: per script, rows in file order are permuted by `default_rng([20261103, 7, i])` (scripts sorted: casia_en, casia_zh, hpl_tamil, iam); `rank_in_script` = position; subset of size N = all rows with `rank_in_script < N/4`; nested (10k within 20k within 40k within 80k); rows of the csv are interleaved by rank, so every csv prefix is script-stratified.

Writers (sample_ids) covered by each prefix: N=10000: 1406; N=20000: 1495; N=40000: 1561; N=80000: 1597 (of 1,597 in the full mix).

Budget test, reference GPU = a4000 (the only type measured on all FMs at writing; slower than A100/H100, so a conservative bound).  Sum of GPU-h over the FMs of each set, one pass over N units:

| N units (per script) | priority set (5 FMs) | all roster candidates with a rate (20 FMs) |
|---|---|---|
| 10,000 (2,500) | 4.0 GPU-h | 6.3 GPU-h |
| 20,000 (5,000) | 8.1 GPU-h | 12.7 GPU-h |
| 40,000 (10,000) | 16.2 GPU-h | 25.3 GPU-h |
| 80,000 (20,000) | 32.4 GPU-h | 50.7 GPU-h |

**Proposal: N = 40,000 units (10,000 per script, first 10,000 ranks of `luo_repa_order_v1.csv`)** - the largest nested size whose all-candidate sum is within ~8% of the 24 GPU-h target (<= 26 GPU-h) on the slowest measured GPU (A4000); the next smaller nested size is the fallback if the sum must stay strictly <= 24.  Faster GPUs (A30/A100/H100) reduce the sum.  Where the priority set alone is the target, N = 80,000 costs the figure in the table above.  Cost drivers on the A4000 (units/s): sd21-dift-gate 1.24, cleandift-sd21 1.74, cleandift-sd15 4.36; every ViT-L-class FM is >= 30 units/s.

Storage of the 16 x 16 mapped gate-layer tokens (prereg grid rule), fp16, for the proposed N (native diffusion maps 48 x 48 x 1280 would be 9x larger and are not stored):

| FM | channels | MB per unit | GB for N |
|---|---|---|---|
| dinov3-vitl16 | 1024 | 0.52 | 21.0 |
| cleandift-sd21 | 1280 | 0.66 | 26.2 |
| sd21-dift-gate | 1280 | 0.66 | 26.2 |
| pe-spatial-l14-448 | 1024 | 0.52 | 21.0 |
| c-radiov4-h | 1280 | 0.66 | 26.2 |
| cleandift-sd15 | 1280 | 0.66 | 26.2 |
| pe-spatial-b16-512 | 768 | 0.39 | 15.7 |
| pe-spatial-g14-448 | 1536 | 0.79 | 31.5 |
| c-radiov3-l | 1024 | 0.52 | 21.0 |
| c-radiov3-h | 1280 | 0.66 | 26.2 |
| c-radiov3-g | 1536 | 0.79 | 31.5 |
| c-radiov4-so400m | 1152 | 0.59 | 23.6 |
| siglip2-giant-opt-patch16-256 | 1536 | 0.79 | 31.5 |
| dinov3-vith16plus | 1280 | 0.66 | 26.2 |
| dinov3-vit7b16 | 4096 | 2.10 | 83.9 |
| tipsv2-so400m14 | 1152 | 0.59 | 23.6 |
| tipsv2-g14 | 1536 | 0.79 | 31.5 |
| dinov3-convnext-large | 1536 | 0.79 | 31.5 |
| siglip2-so400m-patch14-224 | 1152 | 0.59 | 23.6 |
| dinov2-large | 1024 | 0.52 | 21.0 |
| **sum** | | | **569** |

Dependency: LUO training renders at 224 px must be derived from the stored sequences (section 1); they are not byte-identical to the stored 64-px conditions, so a student/backbone trained on derived renders sees a slightly different image than the LUO backbone saw.

## 5. Live cluster state

Snapshot taken 2026-11-03T13:28Z (CLUSTER-A Slurm prints local time CET = UTC+1; CLUSTER-B Slurm prints CET as well; the VPS clock runs at UTC-8).  Raw text: `admin/live_state_2026-11-03.json`.

### CLUSTER-A (account <user>)

- `squeue` for the account at capture: only CPU jobs (`cf2_scor`, `cf2_watc` on skylake-96, other track) and this track's staging jobs; 0 GPU jobs running or pending -> 0 of the 16-GPU QoS cap in use at capture (held/confirmatory jobs of other tracks: none visible in squeue).
- b200idle: single node <node> (8 x B200), state MIXED+DRAIN, reason 'Reboot' set 2026-11-02T14:57:49 (local), 4 of 8 GPUs allocated to other users.  `sbatch --test-only` for 1 x and 2 x B200 returned 'Requested node configuration is not available'.
- A100: <node> IDLE+DRAIN (reason 'CUDA-Update', 2026-10-26), <node> ALLOCATED+DRAIN (reason 'Reboot', 2026-11-02): `--test-only` for 1 x and 2 x A100 (with and without `-p gpuidle`) returned 'Requested node configuration is not available'.

| request (sbatch --test-only) | estimated start (local CET) |
|---|---|
| 2xB200 b200idle 24h | allocation failure: Requested node configuration is not available |
| 2xA100 (no partition) 24h | allocation failure: Requested node configuration is not available |
| 1xA100 (no partition) 12h | allocation failure: Requested node configuration is not available |
| 1xA30 (no partition) 12h | 2026-11-07T06:35:48 |
| 1xA40 (no partition) 12h | allocation failure: Requested node configuration is not available |
| 1xH100 gpuidle 12h | 2026-11-05T13:08:33 |
| 1xA4000 gpuidle 12h | 2026-11-03T17:09:03 |
| 1xgpu any gpuidle cc>=8 12h | 2026-11-03T17:09:04 |
| 2xB200 b200idle 24h -c8 mem64 | allocation failure: Requested node configuration is not available |
| 2xB200 b200idle -c16 mem256 (<node> node) | allocation failure: Requested node configuration is not available |
| 1xB200 b200idle 12h | allocation failure: Requested node configuration is not available |
| 2xA100 gpuidle 24h | allocation failure: Requested node configuration is not available |
| 1xA100 gpuidle 12h | allocation failure: Requested node configuration is not available |
| 1xA40 gpuidle 12h | 2026-11-03T22:23:04 |
| 1xA30 gpuidle 12h | 2026-11-04T04:24:37 |
| 2xA30 (no partition) 24h | 2026-11-07T06:42:37 |
| CLUSTER-A 4xH100 gpuidle 12h | 2026-11-07T13:08:34 |
| CLUSTER-A 2xH100 gpuidle 24h | 2026-11-05T13:56:33 |
| CLUSTER-A 2xA40 gpuidle 24h | 2026-11-04T17:12:04 |
| CLUSTER-A 2xA30 gpuidle 24h | 2026-11-04T06:13:37 |
| CLUSTER-A 2xV100 gpuidle 24h | 2026-11-04T01:03:20 |
| CLUSTER-A 4xH100 gpuidle 12h | 2026-11-07T13:08:34 |
| CLUSTER-A 2xH100 gpuidle 24h | 2026-11-05T13:56:33 |
| CLUSTER-A 2xA40 gpuidle 24h | 2026-11-04T17:12:04 |
| CLUSTER-A 2xA30 gpuidle 24h | 2026-11-04T06:13:37 |
| CLUSTER-A 2xV100 gpuidle 24h | 2026-11-04T01:03:20 |

A4000 timing job 24842562 was submitted at ~14:45 local with an estimate of 17:09 and started after minutes; estimates for gpuidle slots are therefore upper bounds.  gpuidle jobs are preemptible (three of this track's CPU staging jobs were preempted once on <node>).

### CLUSTER-B (user <author>)

| request (sbatch --test-only) | estimated start (local CET) |
|---|---|
| 2xA100-80GB 24h | 2027-05-27T06:49:13 |
| 1xA100-80GB 12h | 2027-05-27T06:03:04 |
| 2xH100 24h(max 1d) | 2026-12-12T18:05:24 |
| 1xH100 12h | 2026-12-12T17:45:49 |
| 2xH200 1d | 2027-02-24T14:27:52 |
| 1xA100-RP 12h | 2026-11-03T14:29:52 |
| 1xH100-RP 12h | 2026-11-03T15:43:49 |
| 1xH100-PCI 12h | 2026-11-06T05:35:14 |
| 1xA100-40GB 12h | 2026-11-05T06:51:09 |
| 2xA100-RP 24h | 2026-11-03T17:03:53 |
| 4xA100-RP 24h | 2026-11-03T19:46:39 |
| 2xH100-RP 23h | 2026-11-03T16:03:24 |
| 4xH100-RP 23h | 2026-11-03T16:03:24 |
| 2xA100-PCI 24h | 2026-11-06T14:48:32 |
| 1xH200-SDS 24h | 2026-11-06T15:21:35 |

Normal partitions (A100-80GB, H100, H200) are estimated to start in Dec 2026 - May 2027 (queue priority of this account); the `-RP` partitions (A100-RP, H100-RP) and A100-PCI/H200-SDS start within hours-days.  `-RP` jobs can be preempted: arms there must be requeue-safe.

### VPS RTX 4090 (read-only)

- queue dir empty; running: 406_cf2_B_s20261033_sample.cmd; held_user_confirm empty; GPU RTX 4090 7.7/24.6 GB used, 89% util; last jobs.csv rows 401-405 (cf2, 3000 s each), 370/371/381 (FMD student, done 08:54-09:23Z); numbers used in 400-499 range: 400,401-406,410,420,430; VPS disk free 2,524 GB
- Number ranges: 400-406 are used by the cf2 track; free for FMDX: 440-499 (and 407-409).  The sealed-read cmds 050/300-349 are not in the queue directory any more (jobs.csv shows them done).

## 6. Lane assignment

| workload | primary lane | race / secondary lane | facts behind the choice |
|---|---|---|---|
| screen extraction of the new FMs on the 6 screen sets (val1000 + 4 x 250 + 3,927 = 5,927 units per FM) | CLUSTER-A gpuidle array (A4000/A30/A40; cap <= 6 GPUs) | CLUSTER-B A100-RP / H100-RP for FMs >= 7 GB (PE-Spatial-G14, C-RADIOv3-g, SigLIP2-giant, CleanDIFT / DIFT) | A4000 rates in section 2: 5,927 units take ~1.8 min (DINOv3-L, 56 u/s), ~12 min (PE-Spatial-G14, 8.4 u/s) and ~82 min (sd21-dift-gate, 1.2 u/s) on one A4000; A4000 slots started within minutes today |
| REPA-target extraction on the fixed subset (section 4) | CLUSTER-A gpuidle arrays, one job per FM | CLUSTER-B A100-RP / H100-RP copies of the two diffusion FMs (largest GPU-h) | diffusion FMs hold 90%+ of the GPU-h in section 3 |
| backbone arms (2 GPUs, 100k steps; control, DINOv3, CleanDIFT/SD2.1, PE-init, ...) | CLUSTER-B H100-RP (2 x and 4 x estimated within hours, see table) and A100-RP | CLUSTER-A gpuidle 2 x A30 (est. 4 Nov 06:13) / 2 x V100 / 2 x H100 (est. 5 Nov 13:56); b200idle only after <node> leaves DRAIN | B200 and A100 are drained at CLUSTER-A; CLUSTER-B `-RP` is preemptible -> every arm must checkpoint and resume; wall time per 100k steps on H100/A100/A30 has NOT been measured in this track (LUO on 2 x B200: ~15 h) |
| FT (p01 / p10 / p30 / p100) + sampling on 7 test sets | VPS RTX 4090 (cmds 440-499) | CLUSTER-A gpuidle A30/A4000 race copies | 4090 queue empty except cf2 job 406 at capture |
| transfers CLUSTER-A <-> CLUSTER-B | VPS hub from CLUSTER-B login1 / CLUSTER-A <node> (sha256 both ends) | CLUSTER-B direct download of public weights (sha-checked against CLUSTER-A manifest) | hub path was blocked 13:35-15:05Z by the VPS `from=` rule (restored by the lead with the user's approval) |

## 7. Initial arm priority order

1. **Control** (no-FM backbone arm): no target extraction; first to launch so its 100k steps are not blocked.
2. **DINOv3-L/16** (only FM that passed the Urdu gate): REPA targets for N units cost 0.20 GPU-h on an A4000; weights stay on CLUSTER-A (licence), so CLUSTER-B copies can only receive target files.
3. **CleanDIFT SD2.1** and the SD2.1 DIFT gate layer (largest Urdu order deltas in FMD, gate-failed on T0 shape only): targets cost 6.40 (CleanDIFT, full-UNet load-test pass) and 9.00 (DIFT gate) GPU-h on an A4000 at the proposed N.
4. **PE-init** (PE-Spatial / PE-Core encoder as initialisation of the 224-px encoder; PE-Spatial-L14 gate-layer targets 0.24 GPU-h; PE-Spatial-B16-512 and PE-Core-B16-224 have 14 x 14 maps and a ViT-B body).
5. Others in the order of their Urdu / cross-script screen strength once verdict files exist (`exp/FMD/xs/verdicts/`): candidates already staged and timed are C-RADIOv3-{B,L,H,g}, C-RADIOv4-{H,SO400M}, PE-Spatial-{T,S,B,L,G}, SigLIP2-giant-opt-256, DINOv3 {ViT-S,B,H+,7B, ConvNeXt}, TIPSv2 {B,L,so400m,g}, and the extras tier.  No screen result is used in this order yet.

## 8. Open items

- A30, A100 and H100 extraction rates were NOT measured: CLUSTER-A A30 timing job 24842563 (gpuidle) was still pending with an estimated start of 2026-11-07 11:17 local; CLUSTER-B timing jobs 3699969 (A100-RP) and 3699970 (H100-RP) were pending without a start estimate after ~40 min, and all three were cancelled so they would not take queue priority from the backbone arms.  Scripts to rerun: exp/FMD/xs/code/timing_a30.sbatch and exp/FMD/xs/code/clusterB_time_{a100rp,h100rp}.sbatch (CLUSTER-B copy at $U_B/exp/FMD/xs/code/).  Section 3 therefore rests on the A4000 only (the slowest card measured); A100/H100 are not extrapolated here.
- Blocked weights: SD3-medium, FLUX.1-dev, FLUX.1-schnell, SAM3 (HF licence not accepted by the account behind the CLUSTER-A token); InkFM unreleased; C-RADIOv4-{B,L,g} do not exist.