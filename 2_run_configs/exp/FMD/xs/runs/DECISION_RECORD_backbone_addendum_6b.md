# DECISION_RECORD_backbone — addendum 6b (track 'Backbone build'; F1–F4)
The shared file DECISION_RECORD_backbone_addendum_6.md on CLUSTER-A was rewritten by the warm-monitor track after 11:20Z, so its D6.x sections replaced F1–F3. This file restores F1–F3 verbatim from the 11:20Z version (artifact e4485d6c, sha256 dc089e38…) and adds F4.

## F1. DiT arms unblocked (lead: PASS_XS flux1-schnell 3051203517, flux1-dev 3051188518, sd3-medium-diffusers 3051105512; prereg A8: CLUSTER-A/CLUSTER-B only)
- Roster copy admin/fm_roster_A3_unblock_dit.json: gated flag cleared, lanes clusterA,clusterB.
- make_arms.py was run on a temp copy and diffed. Exactly the 6 DiT rows changed (BLOCKED -> QUEUED, lanes clusterA,clusterB). Result installed as arms.tsv, sha256 33c8e7e023047b78ccc73849b93ffe8e0b39b4ac501bf3f3904b9bc04396eaf5; the old copy is arms.tsv.pre_dit_unblock_f90f9b02.
- CLUSTER-A jobs via submit_arms.py --only:
  - WARM (2 GPU): flux1-schnell 24866388, flux1-dev 24866387, sd3-medium-diffusers 24866386.
  - SCRATCH: 24866391, 24866390, 24866389.
- CLUSTER-B WARM copies, held: 3703377, 3703378, 3703379.
- CLUSTER-B arms.tsv is still the old version. The hub push failed (F3), so the CLUSTER-B copies stay held until it is updated.
- flux1-dev: non-commercial licence; checkpoints stay on CLUSTER-A/CLUSTER-B.
## F2. DiT S40k targets (A4 convention = the screen extraction: 1024-px, 4 noise draws, gate layer s261_b28 FLUX / s261_b12 SD3)
- Job: GPU array 24866395 (code/_jobs/clusterA_targets_dit.sbatch, registry copy fm_registry_dit.json = fm_registry.json + _import dit_extractors; dit_extractors.py sha256 cf2be7cd…).
- Shards: flux1-schnell 16, flux1-dev 16, sd3-medium-diffusers 4. Throttle %4, rank order.
- Cost on A4000:
  - FLUX: 40,000 units x 12.95 s (lead-measured) = 518,000 s = 143.9 GPU-h per FLUX FM. Targets 40,000x256x3072 fp16 = 62.9 GB each.
  - SD3-medium: per-unit time not measured. ~2.2 s/unit is an estimate scaled by transformer FLOPs to the gate block, giving ~24 GPU-h. Targets 40,000x256x1536 fp16 = 31.5 GB.
- ETA at 4 concurrent A4000 (estimate; assumes the GPUs are granted):
  - flux1-schnell DONE ~5 Nov 24:00Z.
  - flux1-dev DONE ~7 Nov 13:00Z.
  - sd3-medium DONE ~7 Nov 19:00Z.
- WARM 40k on 2xA4000 at ~1.5-2 s/step, ~17-22 h, then FT + selection + A4000 test sampling, ~1 day. Expected finish (estimates; all before 2026-11-10T18:00Z if the lanes deliver):
  - flux1-schnell ~7-8 Nov.
  - flux1-dev ~9 Nov.
  - sd3-medium ~9 Nov.
- Not shortened. Any arm that cannot finish ends 'not assessed' (A4.3 f).
- Target dispatcher restarted as 24866399, with the DiT FMs seeded to array 24866395. A failed DiT array would not be re-dispatched correctly by v1, which uses the main tasks file; this is a known limit.
## F3. Blocker: CLUSTER-A VPS ssh key missing
- $HOME/.ssh/uhwr_staging_vps_ed25519 does not exist on CLUSTER-A. The ~/.ssh dir was modified 2026-11-04 06:54 CET. The first controller hub push failure was at 2026-11-04T06:00:15Z (48 failures by 11:18Z).
- Effects:
  - No CLUSTER-A->hub pushes: CLUSTER-B relay, priority/arms publish, CLUSTER-B arms.tsv update.
  - No CLUSTER-A race locks: arm.sh logs 'race_lock ERR' and trains anyway. Affected: vit-mae-large_W 24843748 (start 08:06:30Z), webssl-mae300m-224_W 24843747 (10:53:19Z), PEINIT_W 24843710 restart (10:39:40Z; it already owned its lock since 18:59Z on 3 Nov).
- Mitigation: from CLUSTER-B (key present) the CLUSTER-A ownership was written to the VPS race dirs for vit-mae-large_W and webssl-mae300m-224_W ('clusterA <job> <start>'), so their CLUSTER-B copies lose the race.
- The key was not restored by this track. That is the user's/lead's decision.

## F4. Static lane assignment (<author> decision 4 Nov: VPS stays OFF; cross-cluster race locks end; lead relay ~14:15Z)
- The warm monitor pinned the 12 warm arms (addendum_6, the lead's entry).
- DiT arms follow the same rule. None was running. Slurm --test-only at 14:14Z: CLUSTER-A gpuidle 1/2-GPU 2026-11-04T23:45 CET; CLUSTER-B H100-RP,A100-RP 1-GPU 2026-11-05T14:12 CET, 2-GPU 14:38 CET. DiT targets exist only on CLUSTER-A.
- So the DiT arms are pinned to CLUSTER-A (WARM 24866388/24866387/24866386, SCRATCH 24866391/24866390/24866389). The CLUSTER-B copies flux1-schnell_W 3703377, flux1-dev_W 3703378 and sd3-medium-diffusers_W 3703379 were scancelled at ~14:16Z. No CLUSTER-B SCRATCH copies existed.

## G1. No-VPS patch (lead approval after <author> decision 4 Nov: VPS stays off)
- Check before the patch (job_history of every CLUSTER-A run dir since 2026-11-04T06Z):
  - Three arms hit race_lock ERR and logged train_start in the same second: vit-mae-large_W 08:06:30Z, PEINIT_W 10:39:40Z, webssl-mae300m-224_W 10:53:19Z.
  - No arm exited or waited, so nothing was requeued.
- Job scripts in code/fmdx_backbone_v1/jobs (bb_train and the other Python code are unchanged; originals kept in jobs/_pre_novps/ and jobs/_pre_novps_SHA256SUMS.txt):
  - common.sh race_lock now uses an atomic mkdir in <xs>/race_local/<arm>, with owner.txt = lane + job + utc, on the cluster's own filesystem.
  - hub_push.sh and hub_pull.sh exit 3 ('hub disabled') without contacting the VPS.
  - New SHA256SUMS.txt sha256 23090037...; common.sh 01b0b058..., hub_push.sh ce4b572a..., hub_pull.sh 0def8518....
  - The same files are deployed on CLUSTER-B (sha256sum -c OK).
  - Test: WIN / WIN for the same lane / LOSE for the other lane.
- Controller v1.4 (code/fmdx_controller_v1.4; gate_controller.py sha256 df9f5a45...; diff controller_v1.3_to_v1.4.diff):
  - --no-vps: no hub pull/push/publish/relay, no VPS queue edits; race owners are read from <xs>/race_local.
  - CLUSTER-A job 24870589 replaces v1.3 24850613.
- Dispatcher v1.1 (code/fmdx_target_dispatch_v1.1; target_dispatch.py sha256 58ae6bd0...): --no-clusterB, so no hub refpush and no CLUSTER-B requests. CLUSTER-A job 24870597 replaces 24866399.
- CLUSTER-B-side shutdown:
  - clusterB_extract_loop.sh retired (exit stub; original saved as clusterB_extract_loop.v1_vps.sh). Its tmux server was no longer running.
  - CLUSTER-B controller 3700469 and CLUSTER-B extraction jobs 3700646, 3700746-3700749, 3701232 and 3701233 scancelled: static lanes, no CLUSTER-B copies remain for those arms.
- Static lane rule for the remaining copies: 14 pending CLUSTER-B copies (SCRATCH arms) were scancelled and their CLUSTER-A copies kept, because the CLUSTER-A estimate was earlier. Listed in arm_log.md.
