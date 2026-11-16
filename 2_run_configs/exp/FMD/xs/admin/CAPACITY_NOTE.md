# CAPACITY NOTE - FMDX staging (track XS staging, <agent>)

Built 2026-11-03T14:40Z.  Facts only.  Sources: admin/timing/*.json (CLUSTER-A), admin/timing_clusterB/*.json (CLUSTER-B), admin/loadtest/*.json, admin/live_state_2026-11-03.json, admin/luo_render_probe.json, data/m5_multiscript_v1/DATA_CARD.md.

## 1. Training data of the LUO backbone (units per source)

LUO backbone = `exp/M5/runs/luo` trained on `data/m5_multiscript_v1/m5_luo_train.npz` (sha256 690441b6bd7c119d774066622c53f23a0aca6e62825a04014351bfba98260b67; 80,000 units; Urdu left out; 100k steps; effective batch 512 sequences/step per prereg A6).  Counts read from the npz (`scripts`, `sample_ids`):

| source (script tag) | units | writers (sample_ids) | train pool units | units with step > 1 px (val250 rate) |
|---|---|---|---|---|
| CASIA-OLHWDB zh (casia_zh) | 20,000 | 552 | 2,105,324 | 46.8% (val) |
| CASIA-OLHWDB en (casia_en) | 20,000 | 552 | 28,699 | 0.0% (val) |
| HPL Tamil (hpl_tamil) | 20,000 | 111 | 49,156 | 2.0% (val) |
| IAM-OnDB components (iam) | 20,000 | 382 | 81,659 | 0.0% (val) |
| **total LUO** | **80,000** | | | |

Not in the LUO mix: OUHD-L Urdu (20,000 sampled in the M5 joint set; p01 FT subset = 336 units / 183 writers, p10 = 3,362, p30 = 10,085, p100 = 33,615), German UNIPEN-de and OnHW (zero-shot).  Screen sets rendered at 224 px (this track): casia_zh / casia_en / hpl_tamil / iam val250 and the UNIPEN-de train stand-in (3,927 units, 12 writers); 64-px re-render byte-identical for every unit.  Files: `exp/FMD/xs/data/xs_render224_*.npz` (+ `.meta.json`).

**Training-render fact (LUO units carry no ground-truth ink):** `m5_luo_train.npz` has `seq` (Nagamatsu) and `cond_rendered` (64 px) but no `gt_points`.  224-px renders therefore have to be derived by `e0_adapter.decode(seq)` -> `fmd_harness.render.render224`.  Probe (admin/luo_render_probe.json, 125 random units per script, seed 20261103):

| script | units probed | decoded-seq 64-px render byte-identical to stored cond_rendered | max abs diff | mean abs diff (grey levels) |
|---|---|---|---|---|
| casia_zh | 125 | 0 | 207 | 6.7653 |
| casia_en | 125 | 0 | 243 | 5.0217 |
| hpl_tamil | 125 | 0 | 220 | 3.0015 |
| iam | 125 | 0 | 193 | 3.9704 |

Speed (1 CPU core): decode 0.11 ms, render64 0.66 ms, render224 0.51 ms per unit -> 80,000 units ~ 0.8 CPU-min.  RTL/mirror_x framing of training renders is not included in this figure.

## 2. Per-FM extraction rate (1,000 val1000 renders, strict fp32, TF32 off, forward + copy of returned layers to CPU, weight load excluded)

| FM | token map at 224 px | channels | a4000 u/s | code sha (first run) |
|---|---|---|---|---|
| dinov3-vitl16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 1024, "mid": 1024} | 56.4 | 4b629160 |
| sd21-dift-gate | {"t261_up1": [48, 48]} | {"t261_up1": 1280} | 1.2 | 4b629160 |
| cleandift-sd21 | {"up1": [48, 48], "up2": [96, 96]} | {"up1": 1280, "up2": 640} | 1.7 | 0ccce9a0 |
| cleandift-sd15 | {"up1": [32, 32], "up2": [64, 64]} | {"up1": 1280, "up2": 640} | 4.4 | 0ccce9a0 |
| pe-spatial-b16-512 | {"last": [14, 14]} | {"last": 768} | 190.8 | 0ccce9a0 |
| pe-spatial-l14-448 | {"last": [16, 16]} | {"last": 1024} | 45.4 | 0ccce9a0 |
| pe-spatial-g14-448 | {"last": [16, 16]} | {"last": 1536} | 8.4 | 0ccce9a0 |
| c-radiov3-l | {"last": [14, 14]} | {"last": 1024} | 65.8 | 0ccce9a0 |
| c-radiov3-h | {"last": [14, 14]} | {"last": 1280} | 32.0 | 0ccce9a0 |
| c-radiov3-g | {"last": [14, 14]} | {"last": 1536} | 17.9 | 0ccce9a0 |
| c-radiov4-h | {"last": [14, 14]} | {"last": 1280} | 31.9 | 0ccce9a0 |
| c-radiov4-so400m | {"last": [14, 14]} | {"last": 1152} | 47.0 | 0ccce9a0 |
| siglip2-giant-opt-patch16-256 | {"last": [14, 14], "mid": [14, 14]} | {"last": 1536, "mid": 1536} | 17.4 | 0ccce9a0 |
| dinov3-vith16plus | {"last": [14, 14], "mid": [14, 14]} | {"last": 1280, "mid": 1280} | 20.4 | 0ccce9a0 |
| dinov3-vit7b16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 4096, "mid": 4096} | 18.3 | 0ccce9a0 |
| tipsv2-so400m14 | {"last": [16, 16]} | {"last": 1152} | 30.3 | 0ccce9a0 |
| tipsv2-g14 | {"last": [16, 16]} | {"last": 1536} | 11.6 | 0ccce9a0 |
| dinov3-convnext-large | {"last": [7, 7], "mid": [14, 14]} | {"last": 1536, "mid": 768} | 87.7 | 0ccce9a0 |
| siglip2-so400m-patch14-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1152, "mid": 1152} | 31.5 | 0ccce9a0 |
| dinov2-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 42.1 | 0ccce9a0 |
| aimv2-large-patch14-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 47.0 | 0ccce9a0 |
| c-radiov3-b | {"last": [14, 14]} | {"last": 768} | 221.9 | 0ccce9a0 |
| deepseek-ocr | {"sam_last": [4, 4], "clip_last": [4, 4]} | {"sam_last": 1024, "clip_last": 1024} | 493.3 | 0ccce9a0 |
| depth-anything-v2-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 46.3 | 0ccce9a0 |
| dinov2-reg-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 43.8 | 0ccce9a0 |
| dinov3-vitb16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 768, "mid": 768} | 181.9 | 0ccce9a0 |
| dinov3-vits16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 384, "mid": 384} | 537.1 | 0ccce9a0 |
| pe-core-b16-224 | {"last": [14, 14]} | {"last": 768} | 186.9 | 0ccce9a0 |
| pe-spatial-s16-512 | {"last": [14, 14]} | {"last": 384} | 598.1 | 0ccce9a0 |
| pe-spatial-t16-512 | {"last": [14, 14]} | {"last": 192} | 1146.9 | 0ccce9a0 |
| qaari-urdu-vision | {"pre_merge": [16, 16], "last": [8, 8]} | {"pre_merge": 1280, "last": 1536} | 77.2 | 0ccce9a0 |
| qwen2-vl-2b-instruct | {"pre_merge": [16, 16], "last": [8, 8]} | {"pre_merge": 1280, "last": 1536} | 76.9 | 0ccce9a0 |
| starvector-1b-im2svg | {"last": [16, 16]} | {"last": 1024} | 46.1 | 0ccce9a0 |
| tipsv2-l14 | {"last": [16, 16]} | {"last": 1024} | 42.8 | 0ccce9a0 |
| trocr-large-handwritten | {"last": [24, 24]} | {"last": 1024} | 20.8 | 0ccce9a0 |
| vit-mae-large | {"last": [14, 14], "mid": [14, 14]} | {"last": 1024, "mid": 1024} | 64.9 | 0ccce9a0 |
| webssl-dino1b-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1536, "mid": 1536} | 12.0 | 0ccce9a0 |

GPUs: a4000 = NVIDIA RTX A4000.  `bf16`: dinov3-vit7b16 is loaded in bf16 (27 GB fp32 does not fit a 16/24 GB card); every other row is fp32.  sd21-dift-gate = DIFT gate layer only (768 px, 4 noise draws x 1 timestep, UNet stopped after up_block 1); cleandift-sd21 = one clean-latent pass at t = 0 through the full UNet at 768 px (load-test convention, NOT the official `get_features` path).

## 3. GPU-hours for REPA targets

GPU-h = units / (units/s) / 3600 on one GPU of that type, one pass over the units, extraction only (no probe/training time).  Rows use the measured rate of the FM on that GPU type.

### priority set (control has no FM; DINOv3-L, CleanDIFT-SD2.1, SD2.1-DIFT gate, PE-Spatial-L14 as PE proxy, C-RADIOv4-H)

| FM | a4000: 80k units GPU-h |
|---|---|
| dinov3-vitl16 | 0.39 |
| cleandift-sd21 | 12.80 |
| sd21-dift-gate | 17.99 |
| pe-spatial-l14-448 | 0.49 |
| c-radiov4-h | 0.70 |
| **sum over rows with a rate** | **32.37** |
| rows without a rate | 0 |

Subset size that keeps the sum at <= 24 GPU-h (per tag, linear scaling): a4000: 59305 units

### all roster candidates with a measured rate

| FM | a4000: 80k units GPU-h |
|---|---|
| dinov3-vitl16 | 0.39 |
| sd21-dift-gate | 17.99 |
| cleandift-sd21 | 12.80 |
| cleandift-sd15 | 5.10 |
| pe-spatial-b16-512 | 0.12 |
| pe-spatial-l14-448 | 0.49 |
| pe-spatial-g14-448 | 2.63 |
| c-radiov3-l | 0.34 |
| c-radiov3-h | 0.69 |
| c-radiov3-g | 1.24 |
| c-radiov4-h | 0.70 |
| c-radiov4-so400m | 0.47 |
| siglip2-giant-opt-patch16-256 | 1.28 |
| dinov3-vith16plus | 1.09 |
| dinov3-vit7b16 | 1.22 |
| tipsv2-so400m14 | 0.73 |
| tipsv2-g14 | 1.91 |
| dinov3-convnext-large | 0.25 |
| siglip2-so400m-patch14-224 | 0.71 |
| dinov2-large | 0.53 |
| **sum over rows with a rate** | **50.69** |
| rows without a rate | 0 |

Subset size that keeps the sum at <= 24 GPU-h (per tag, linear scaling): a4000: 37879 units
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