# DECISION_RECORD_backbone — addendum 7 (2026-11-04 ~17:40Z, track 'Backbone build'; <author>: use CLUSTER-B GPUs, static lanes, no VPS)
## H1. CLUSTER-B start estimates (sbatch --test-only, 24 h, 8 cpu, 64 GB; CLUSTER-B local time CET; run 17:36Z)
| partition | GPUs | earliest start (CET) |
|---|---|---|
| A100-40GB | g1 | 2026-11-18T05:19:37 |
| A100-40GB | g2 | 2026-11-18T06:19:25 |
| A100-80GB | g1 | 2027-06-08T11:50:36 |
| A100-80GB | g2 | 2027-06-08T14:54:24 |
| A100-PCI | g1 | 2026-11-07T20:32:02 |
| A100-PCI | g2 | 2026-11-07T21:48:38 |
| A100-RP | g1 | 2026-11-04T19:46:36 |
| A100-RP | g2 | 2026-11-05T02:55:02 |
| A100-SDS | g1 | 2026-11-05T06:07:44 |
| A100-SDS | g2 | 2026-11-05T06:26:49 |
| B200 | g1 | 2026-11-07T22:12:17 |
| B200 | g2 | 2026-11-07T23:02:31 |
| batch | g1 | 2026-11-04T19:06:56 |
| batch | g2 | 2026-11-04T19:06:56 |
| H100 | g1 | 2026-12-18T11:48:56 |
| H100 | g2 | 2026-12-18T11:48:56 |
| H100-PCI | g1 | 2026-11-20T16:40:41 |
| H100-PCI | g2 | 2026-11-20T16:44:42 |
| H100-RP | g1 | 2026-11-07T03:36:56 |
| H100-RP | g2 | 2026-11-07T03:36:56 |
| H200 | g1 | 2027-02-23T01:47:56 |
| H200 | g2 | 2027-02-23T01:47:56 |
| H200-PCI | g1 | 2026-12-03T18:19:21 |
| H200-PCI | g2 | 2026-12-03T18:58:15 |
| H200-SDS | g1 | 2026-11-07T18:39:41 |
| H200-SDS | g2 | 2026-11-07T18:39:41 |
| L40S | g1 | 2026-11-10T15:48:43 |
| L40S | g2 | 2026-11-11T01:30:06 |
| RTX3090 | g1 | 2026-12-07T16:07:34 |
| RTX3090 | g2 | 2026-12-08T03:56:03 |
| RTXA6000 | g1 | 2026-11-11T05:21:50 |
| RTXA6000 | g2 | 2026-11-11T05:21:50 |
| RTXA6000-SDS | g1 | 2026-11-12T20:13:16 |
| RTXA6000-SDS | g2 | 2026-11-13T01:03:00 |
| RTXB6000 | g1 | 2026-11-05T00:36:56 |
| RTXB6000 | g2 | 2026-11-05T00:36:56 |
| V100-32GB | g1 | 2026-11-07T22:20:35 |
| V100-32GB | g2 | 2026-11-07T22:20:35 |
| V100-32GB-SDS | g1 | 2026-11-04T21:40:35 |
| V100-32GB-SDS | g2 | 2026-11-04T21:40:35 |
## H2. CLUSTER-A controller v1.5 (code/fmdx_controller_v1.5; gate_controller.py sha256 4bcc010c...; diff controller_v1.4_to_v1.5.diff; CLUSTER-A job 24875898)
- v1.3/v1.4 treated a released arm that waits only on Slurm as a blocker. After the gpuidle requeue of vit-mae-large_W and webssl-mae300m-224_W, they were re-held behind PEINIT_W (rank 2, Slurm Reason=Priority).
- v1.5: only an arm waiting for LANE GPUs (cap) blocks lower ranks. A released arm keeps its cap slots, and Slurm orders the released arms by nice = rank.
- vit-mae-large_W has a latest.pt and resumes.
## H3. Lane assignment
- CLUSTER-A first wave by rank (2 GPU each; cap 14 minus ~2-4 other FMD GPUs): PEINIT_W, vit-mae-large_W, webssl-mae300m-224_W, c-radiov3-g_W, c-radiov4-h_W.
- CLUSTER-A second wave (~16-17 h later, i.e. ~5 Nov 16-24Z if Slurm grants the 2-GPU slots; gpuidle 2-GPU estimate 23:45 CET 4 Nov): dinov3-vit7b16_W, c-radiov3-l_W, c-radiov4-so400m_W, c-radiov3-b_W, then flux1-schnell_W once its targets land (~6 Nov 00Z).
- So no PASS_XS WARM arm is expected to start on CLUSTER-A later than ~6 Nov 12Z. Exceptions are flux1-dev and sd3 (targets ~7 Nov; DiT, excluded from CLUSTER-B), dinov3 (licence) and INKSIGHT (pseudo data CLUSTER-A-only).
- No WARM arm is moved to CLUSTER-B. This is to be re-checked if the second wave has not started by 6 Nov 06Z.
- SCRATCH arms rank after all WARM arms. On CLUSTER-A they are 1-GPU jobs: 100k steps at the 1xA4000 measured 2.98 s/step = ~83 h. Starting after the WARM waves (>= 6 Nov 12Z), they cannot finish by 10 Nov 18Z.
- Pinned to CLUSTER-B, 2 GPU, partitions A100-RP,H100-RP,RTXB6000,A100-SDS,batch:
  - controls NOFM_S 3705244, PEINIT_S 3705245 (released);
  - vit-mae-large_S 3705246, webssl-mae300m-224_S 3705247, c-radiov3-h_S 3705248, c-radiov3-g_S 3705249, c-radiov4-h_S 3705250, c-radiov3-l_S 3705251, c-radiov4-so400m_S 3705252, c-radiov3-b_S 3705253 (held until their CLUSTER-B targets pass the chunk-0 check).
- Their CLUSTER-A copies were scancelled (24843785 24843790 24843826 24843825 24843809 24843810 24843812 24843808 24843811 24843807). arm_queue.csv state is PINNED_CLUSTER-B with clusterB_job set, and the controller skips these arms because they have no CLUSTER-A job.
- Excluded from CLUSTER-B: DINOv3 family (licence), DiT/SD/FLUX (no weights/token on CLUSTER-B), INKSIGHT (pseudo data on CLUSTER-A only).
- clusterB_arm.sbatch continuation segments use its own #SBATCH partitions (H100-RP,A100-RP).
## H4. CLUSTER-B-side S40k targets
- clusterB_targets_fm.sbatch, 1 GPU, partitions A100-RP,H100-RP,RTXB6000,A100-SDS,batch; jobs 3705254-3705261 (vit-mae-large, webssl-mae300m-224, c-radiov3-h, c-radiov3-g, c-radiov4-h, c-radiov3-l, c-radiov4-so400m, c-radiov3-b).
- Each job releases its FM's SCRATCH arm after the chunk-0 check vs the CLUSTER-A chunk 0. A reference exists on CLUSTER-B for all 8 FMs (targets/repa_clusterA_chunk0; pulled with sha OK on 3 Nov). Pass rule: cos_min >= 0.999 and maxabs <= 0.01 max|ref|.
- The c-radiov3-l/-b inputs came from clusterB_cradio_extract_in (pulled 2026-11-03T16:04Z, sha OK); their .PULLED markers were added so the jobs do not attempt a hub pull.
## H5. Checkpoint relay
- When a CLUSTER-B arm finishes, its bb_ms<k>.pt paths and sha256 are sent to the lead for relay to CLUSTER-A (no VPS).

## H6. CLUSTER-A partitions widened (lead ~18:05Z: gpuidle preempted ~16:00Z; llms starts now)
- sbatch --test-only at 18:07Z, CET, 2 days:
  - llms (RTX PRO 6000 96 GB nodes): 1 and 2 GPU at 2026-11-04T19:07.
  - b200idle: 1 GPU 2026-11-04T23:13; 2 GPU 2026-11-07T03:16.
  - gpuidle: 1/2 GPU 2026-11-05T12:57.
  - gpu, dgx: not permitted (error).
- All 18 pending CLUSTER-A WARM jobs (fx_*_W) and the DiT target array 24866395 now have Partition=gpuidle,llms,b200idle.
- Cap 14 (lane-wide, A7) and strict rank order are unchanged (controller v1.5). SCRATCH and other jobs are unchanged.
- Resume layout (arm.sh v1.0.2, job script only; jobs/arm.sh sha256 b5006f16…, SHA256SUMS.txt 660de55f…; original kept in jobs/_pre_layoutkeep/; deployed on CLUSTER-A and CLUSTER-B, sha256sum -c OK):
  - With a latest.pt, arm.sh reuses per_gpu from the run's last batch_layout line for the same GPU count. So PEINIT_W, vit-mae-large_W and webssl-mae300m-224_W (A4000: 2 GPU x 32/proc x accum 8) keep that layout if they resume on 96 GB cards.
  - Code, seed, data and global 512 sequences/step are unchanged.
  - Arms without progress take the device rule (>= 40 GB: 64/proc x accum 4, prereg A6).
- DiT targets: shards that start on llms/b200idle run on a different GPU type than A4000 shards of the same FM. The device is recorded per extraction in the target meta.

## H7. SD3.5 DiT arms (lead ~19:45Z: PASS_XS sd3.5-large-diffusers-layout 3041115518, sd3.5-large-turbo-diffusers-layout 3041112517, sd3.5-medium-diffusers-layout 3041085513; CLUSTER-A only)
- Roster copy admin/fm_roster_A3_sd35.json (sha256 bd5796bb…; the three FMs added: family diffusion (DiT), lanes clusterA).
- make_arms.py on a temp copy, diffed: exactly 6 rows added. Result installed as arms.tsv, sha256 396d0290…; old copy arms.tsv.pre_sd35_33c8e7e0. arm_queue.csv rows added.
- CLUSTER-A arms:
  - WARM (2 GPU, gpuidle,llms,b200idle): medium 24878944, large 24878945, large-turbo 24878946.
  - SCRATCH: 24878947, 24878948, 24878949.
  - Ranked by priority_score under controller v1.5.
- S40k targets: array 24878952 (clusterA_targets_dit.sbatch; tasks targets_tasks_sd35.txt sha256 71ebb110…: large 16 shards, large-turbo 16, medium 4; %2; partitions gpuidle,llms,b200idle; rank order).
- Dispatcher state seeded so the dispatcher waits on 24878952.
- Cost estimate (not measured):
  - Scaled from the lead-measured FLUX 12.95 s/unit by transformer size up to the gate block.
  - large/turbo ~9 s/unit = ~100 A4000 GPU-h each; medium ~2.6 s/unit = ~29 GPU-h.
  - Targets: large/turbo 40,000x256x2432 fp16 = 49.8 GB each; medium 31.5 GB.
- ETA at 2 concurrent A4000 (estimate; faster on llms):
  - Targets: large ~7 Nov 00Z, turbo ~9 Nov 02Z, medium ~9 Nov 16Z.
  - WARM end ~17–22 h after each, then FT + selection + test ~1 day.
  - large is expected to finish ~9 Nov. turbo and medium are expected to miss 10 Nov 18:00Z and end 'not assessed' (A4.3 f) unless llms speeds them up. Not shortened.
- 2026-11-04T19:48:03Z dispatcher v1.1 restarted as 24878954 (SD3.5 seeded)
