# HANDOFF - FMDX warm$SCRATCH_ROOT monitor (written 2026-11-10T17:13:19Z, replacement monitor track, <agent>)
Facts only. X = CLUSTER-A $U/exp/FMD/xs; MR = X/runs/fmdx_monitor.

## 1. Where things are
- Tables: MR/tables/FMDX_warm_{absolute,contrasts}.csv, _contrasts.md, _inputs.json, _SHA256.txt; A4.3f list MR/tables/FMDX_not_assessed_A43f.csv; provenance MR/tables/PROVENANCE_FMDX_warm.json.
- Builder: code/fmdx_monitor_v1/build_tables_a9.py via tables_job_a9.sbatch (no verdicts; device_a/device_b/DEVICE_MIXED; controls by regime suffix: _W -> NOFM_W/PEINIT_W, _S -> NOFM_S/PEINIT_S).
- State: MR/state/warm_state.json (per arm bb_ms, ft_ms, rows, npreds, repairs, tables_history); events MR/state/autopilot_events.log, MR/ACTIONS_LOG.md; summary MR/state/autopilot_summary.txt; status X/runs/WARM_MONITOR_STATUS.md.
- Autopilot: code/fmdx_monitor_v1/fxm_autopilot.py (+ .sbatch), job 25036911 on cpuidle, exits at 2026-11-10T18:00Z. To continue: edit --until in fxm_autopilot.sbatch and sbatch it; it resumes from warm_state.json.
- Per-arm pipeline (unchanged recipes): training_complete + eval/best_milestone.txt -> FT (ft_clusterA.sbatch, 3 earliest --test-only copies; local lock in ft_job.sh) + A9 bb sampling (arm_sample_a9.sbatch, 3 copies, per-seed mkdir locks) -> FT val1000 selection -> A9 ft sampling -> adapt_sidecars + lane_score_array_mon.sh -> compose_preds.py (A4.4) -> score cmp -> tables.
- Polling rule (lead 8 Nov): CLUSTER-A at most hourly, one short command per poll (e.g. cat MR/state/autopilot_summary.txt). No CLUSTER-A<->CLUSTER-B relays.

## 2. Arms with full rows (bb 18 + FT 24 + composed 18 files scored) in table job 25049213
| arm | FM status | bb ms | FT ms |
|---|---|---|---|
| NOFM_W | control | 3 | 1 |
| PEINIT_W | control | 4 | 2 |
| c-radiov3-b_W | PASS_XS | 4 | 3 |
| c-radiov3-h_W | PASS_XS | 4 | 4 |
| c-radiov4-so400m_W | PASS_XS | 4 | 3 |
| cleandift-sd21_W | PASS_URDU_ONLY | 4 | 3 |
| dinov3-vitb16_W | PASS_XS | 4 | 2 |
| dinov3-vitl16-sat493m_W | PASS_XS | 4 | 5 |
| pe-spatial-b16-512_W | PASS_URDU_ONLY | 4 | 2 |
| vit-mae-large_W | PASS_XS | 4 | 5 |
| webssl-mae300m-224_W | PASS_XS | 4 | 2 |

## 3. Partial
- webssl-mae1b-224_W: bb ms4, FT ms3; FT rows 24/24 scored; bb sampling seed0 done (6 files), seeds 1-2 pending (fxm_as9_webssl-mae1b-224_W_bb); bb and composed rows absent from the table.

## 4. Not assessed by 2026-11-10T18:00Z (A4.3f), last stage
| arm | regime | last stage |
|---|---|---|
| dinov3-vitl16_W | warm | training not started on CLUSTER-A (queue state SUBMITTED) |
| dinov3-vits16_W | warm | training step 6850 of 40000 (slurm PENDING(JobHeldUser)) |
| dinov3-vit7b16_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| c-radiov3-l_W | warm | training step 36200 of 40000 (slurm PENDING(Priority)) |
| c-radiov3-g_W | warm | training step 36100 of 40000 (slurm PENDING(Priority)) |
| c-radiov4-h_W | warm | training step 36050 of 40000 (slurm PENDING(Priority)) |
| sd3-medium-diffusers_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| flux1-dev_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| flux1-schnell_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| qwen3.5-4b_W | warm | training step 0 of 40000 (slurm PENDING(JobHeldUser)) |
| INKSIGHT_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| inkfm_W | warm | training not started on CLUSTER-A (queue state QUEUED) |
| NOFM_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| PEINIT_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| dinov3-vitl16_S | scratch | training step 2600 of 100000 (slurm PENDING(JobHeldUser)) |
| cleandift-sd21_S | scratch | training step 5850 of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vits16_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vitb16_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vit7b16_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| c-radiov3-b_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-l_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-h_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-g_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov4-so400m_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov4-h_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| pe-spatial-b16-512_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| webssl-mae300m-224_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| vit-mae-large_S | scratch | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| sd3-medium-diffusers_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| flux1-dev_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| flux1-schnell_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| qwen3.5-4b_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| INKSIGHT_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| inkfm_S | scratch | training not started on CLUSTER-A (queue state QUEUED) |
| sd3.5-large-diffusers-layout_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-turbo-diffusers-layout_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-medium-diffusers-layout_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-diffusers-layout_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-turbo-diffusers-layout_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| sd3.5-medium-diffusers-layout_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| webssl-mae1b-224_W | warm | FT selected ms3; incomplete: bb rows 0/18; cmp rows 0/18 (sampling PENDING(Priority)) |
| webssl-mae1b-224_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vitl16-sat493m_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| chandra-ocr-2_W | warm | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| chandra-ocr-2_S | scratch | training step None of 100000 (slurm PENDING(JobHeldUser)) |

## 5. Known issues
- CLUSTER-A SSH stopped answering during agent sftp transfers of 50-200 MB (8 Nov); downloads of a few hundred kB work.
- DEVICE_MIXED rows: arm sides on A30/B200 vs A4000 comparators (A9.2 note applies).
- CLUSTER-B SCRATCH arms (3705244-53 lineage; fx_*_S on CLUSTER-B) all PENDING at 15:3xZ 10 Nov; no RELAY.json under CLUSTER-B runs/.
- c-radiov3-g_W, c-radiov4-h_W, c-radiov3-l_W at ~36k/40k steps, PENDING (Priority) on CLUSTER-A at 17:12Z.
