# DECISION_RECORD_backbone_addendum_6 (FMDX warm monitor track)
Facts only. Reason: <author> decision 4 Nov 2026, relayed by the lead: VPS stays OFF; cross-cluster race locks end; static lane assignment (each arm/FT/sampling job on exactly one lane).
Written by the warm-monitor track (<agent>; monitoring/existing recipes). Executed 2026-11-04T14:14:35Z.
## D6.1 Pinning of warm arms with copies on both clusters
No warm arm had a RUNNING copy on CLUSTER-B; no copy of these arms was RUNNING on CLUSTER-A either (the CLUSTER-A copies are held by controller v1.3 until a cap slot is free; running CLUSTER-A warm arms NOFM_W, PEINIT_W, c-radiov3-h_W have no CLUSTER-B copy in squeue). Rule applied: pin to the lane with the earlier Slurm start estimate. CLUSTER-A: sbatch --test-only for a 2xA4000 gpuidle job gave 2026-11-05T08:45Z; CLUSTER-B: squeue --start gave 2026-11-08T04:53 (c-radiov3-l), 2026-11-18T07:06 (c-radiov3-g), N/A for the rest (held or no estimate). All 12 arms pinned to CLUSTER-A; CLUSTER-B copies cancelled (re-pin rule: if an CLUSTER-A copy has not started 6 h after its estimate, 2026-11-05T14:45Z, the lead decides).
- cleandift-sd21_W: CLUSTER-A copy 24843709 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700350 (PENDING (JobHeldUser), est. N/A) scancelled
- c-radiov3-b_W: CLUSTER-A copy 24843727 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700359 (PENDING (JobHeldUser), est. N/A) scancelled
- c-radiov3-l_W: CLUSTER-A copy 24843728 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700360 (PENDING (Priority), est. 2026-11-08T04:53:00) scancelled
- c-radiov3-g_W: CLUSTER-A copy 24843730 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700362 (PENDING (Priority), est. 2026-11-18T07:06:00) scancelled
- c-radiov4-so400m_W: CLUSTER-A copy 24843731 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700363 (PENDING (Priority), est. N/A) scancelled
- c-radiov4-h_W: CLUSTER-A copy 24843732 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700364 (PENDING (Priority), est. N/A) scancelled
- pe-spatial-b16-512_W: CLUSTER-A copy 24843735 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700367 (PENDING (JobHeldUser), est. N/A) scancelled
- sd3-medium-diffusers_W: CLUSTER-A copy 24866386 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3703379 (PENDING (JobHeldUser), est. N/A) scancelled
- flux1-dev_W: CLUSTER-A copy 24866387 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3703378 (PENDING (JobHeldUser), est. N/A) scancelled
- flux1-schnell_W: CLUSTER-A copy 24866388 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3703377 (PENDING (JobHeldUser), est. N/A) scancelled
- qwen3.5-4b_W: CLUSTER-A copy 24843761 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700388 (PENDING (JobHeldUser), est. N/A) scancelled
- INKSIGHT_W: CLUSTER-A copy 24843782 (PENDING, held/pending; CLUSTER-A 2xA4000 --test-only est. 2026-11-05T08:45Z) kept; CLUSTER-B copy 3700399 (PENDING (JobHeldUser), est. N/A) scancelled
Not touched: SCRATCH copies and arms outside the warm queue; CLUSTER-B extraction/target jobs; any controller file; VPS (no connection made in this action).
## D6.2 Monitor jobs
All monitor jobs (comparator re-sampling, arm sampling, FT, scoring) already run on CLUSTER-A only. FT copies use a cluster-local lock (mkdir RUN/.local_lock + .local_owner); the VPS race_lock call in ft_job.sh returned ERR (VPS off/unreachable) and does not gate anything.
## D6.3 CLUSTER-B-trained arms
None so far: no arm has trained on CLUSTER-B. If one does, its bb_ms<k>.pt (about 390 MB) must be relayed to CLUSTER-A for FT/A4000 sampling (A8: primary test preds on CLUSTER-A A4000).

## F4. Static lane assignment (<author> decision 4 Nov: VPS stays OFF; cross-cluster race locks end; lead relay ~14:15Z)
- The warm monitor pinned the 12 warm arms (addendum_6, the lead's entry).
- DiT arms follow the same rule. None was running. Slurm --test-only at 14:14Z: CLUSTER-A gpuidle 1/2-GPU 2026-11-04T23:45 CET; CLUSTER-B H100-RP,A100-RP 1-GPU 2026-11-05T14:12 CET, 2-GPU 14:38 CET. DiT targets exist only on CLUSTER-A.
- So the DiT arms are pinned to CLUSTER-A (WARM 24866388/24866387/24866386, SCRATCH 24866391/24866390/24866389). The CLUSTER-B copies flux1-schnell_W 3703377, flux1-dev_W 3703378 and sd3-medium-diffusers_W 3703379 were scancelled at ~14:16Z. No CLUSTER-B SCRATCH copies existed.

## D6.4 (2026-11-04T18:05Z, lead decision: no nice/fairshare manipulation; resubmit stalled FT/selection to the earliest partition)
- Cause: ~16:00Z all account GPU jobs in gpuidle were preempted; sbatch --test-only starts (CLUSTER-A local time = UTC+1): gpuidle A4000/A30 2026-11-05T15:50 local, H100 18:47, A40 11-06T03:30; partition gpu A30 11-07T03:41, A100 11-16; b200idle B200 11-04T23:13 local; llms (node <node>) 11-04T19:02 local (= 18:02Z); idle 11-06. CLUSTER-B not used (lead: relay only if essential).
- Action: llms partition (RTX PRO 6000 Blackwell Server Edition, 95 GB): FT c-radiov3-h_W job 24877088 (--gres gpu:rtx6000_2:1, FT_LAYOUT=32x16 = same micro-batch layout, same code sha, seed 20261034, resumed from latest.pt) started 18:03:41Z; PEINIT_W selection (monitor copy sel_job.sh, same logic as the arm.sh block) job 24877089 started 18:03:40Z; it holds controller job 24843710 (scontrol hold) to avoid two writers on eval/ and releases it at the end. NOFM_W FT: original job 24863028 restarted at 18:01:34Z on gpuidle A4000 <node> (resume from latest.pt); llms copy 24877087 exited LOCAL_RACE_LOST; extra copies 24875478/79/80 cancelled.
- No file relay needed; no CLUSTER-B job submitted.
- Note: preemption-resume restarts the batch sampler seed (bb_train: seed + step) as in any requeue; the layout is unchanged (32x16).

