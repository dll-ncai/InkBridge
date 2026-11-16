# DECISION_RECORD_backbone — addendum 2 (2026-11-03 ~16:00–16:40Z, track 'Backbone build')
Follows addendum 1 (sha256 322d171e5ee518ebf2e7958e4670f9b69a054cd477e1c1effd3b576bfbc84836). The frozen DR is unchanged.

## B1. CLUSTER-B-side extraction of S40k REPA targets (lead ~16:00Z)
- Reason: hub link ~6–9 MB/s, while the C-RADIO targets are 15.7–31.5 GB each.
- CLUSTER-B already holds the S40k inputs from bb_data_1, sha-checked: luo_train_render224.npy (+unit ids) and repa_subset_S40k_positions.npy.
- CLUSTER-B had weights for c-radiov3-b/-l/-g and c-radiov4-h. The weights for c-radiov3-h and c-radiov4-so400m are pushed through the hub (job 24846256).
- Pushed extractor: fmd_ext_xs_encoders.py, sha256 e02fdf462ae6a61d0869a19cdca067a2f8b2e8adec06318e5f6f242decbba5c4. fmd_ext_encoders/*.py and fmd_harness_v1/*.py are byte-identical on CLUSTER-A and CLUSTER-B (sha256 compared).
- Extraction setup:
  - prep_repa_targets.py from code fmdx_backbone_v1 (sha-pinned), chunk 256, same positions.
  - Container: CLUSTER-B pyxis image pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh with CLUSTER-A paths mounted.
  - PYTHONPATH: fmdx_backbone_v1, fmd_harness_v1, pylibs_clusterB (transformers 5.22.1, timm 1.0.30, as CLUSTER-A xs/pylibs).
  - The extractor runs fp32 with tf32 off on GPU; the CLUSTER-A copies were extracted on CPU (fp32).
- Jobs (code/_jobs/clusterB_tgt):
  - clusterB_targets_cradio.sbatch (c-radiov3-l, -b): job 3700646, sbatch sha256 e2bbb31f….
  - clusterB_targets_fm.sbatch (one FM, pulls its own inputs from the hub, sha256 d30c372a…): c-radiov3-h 3700746, c-radiov3-g 3700747, c-radiov4-h 3700748, c-radiov4-so400m 3700749.
- Chunk-0 provenance check (chunk_check.py, sha256 0b064b356652c475fbfe01cd5db1b51f4e972a2ecb33defe16e59d390790bc05):
  - Compares CLUSTER-B rows [0:256] with the CLUSTER-A chunk_00000.npy of the same FM.
  - The pass rule was fixed before any comparison: min per-token cosine ≥ 0.999 AND max|Δ| ≤ 0.01·max|CLUSTER-A|.
  - PASS → scontrol release of the CLUSTER-B WARM copy. FAIL → the copy stays held and the fallback is the hub relay.
  - Each result is written to targets/repa/<fm>/chunk0_check_vs_clusterA.json on CLUSTER-B and copied into the handoff.
- CLUSTER-B controller relay pull is skipped for these six FMs. Their entries are pre-seeded in admin/controller_state_clusterB.json (rc "clusterB_local_extraction") before the CLUSTER-B controller's first start, so a relayed copy cannot overwrite a CLUSTER-B-extracted file under a running arm.
- Exception: if the relayed CLUSTER-A file reaches CLUSTER-B first, it is installed by a manual hub_pull and feeds the arm. Which copy fed each CLUSTER-B arm is recorded in arm_log_clusterB.md.

## B2. CLUSTER-A CPU extraction (gpuidle full)
- 24845068 (addendum 1). 24846255: c-radiov3-h 16 shards, c-radiov3-g 16, c-radiov4-h 12, c-radiov4-so400m 8; tasks file sha256 8cba9d5b….
- Main GPU array tasks 24843687_32–35 cancelled. The main array throttle stays at 1.

## B3. Incident: foreign pushes into the VPS stage (lead relay ~16:30Z)
- Screen-encoders jobs 24845238 (15:48–16:14Z) and 24846129 (16:14–16:23Z), plus a login push at ~16:23Z, tar-streamed exp/FMD/xs/targets into the VPS stage and were killed mid-stream.
- Audit job 24846202 (admin/xfer/vps_repa_audit_24846202.tsv, listing vps_targets_listing_24846202.txt, 248 files) compared the VPS certutil SHA256 with CLUSTER-A for every final file of every CLUSTER-A FM dir. Only dinov3-vitl16 had final files: npy bc161a92…, unit_ids d4904e13…, meta 126ed557…, DONE.json 639e0ccc…, all MATCH.
- VPS runner jobs since 15:48Z: only 440_fx_dinov3-vitl16_W (start 15:52:11Z). Its run_config.json records the target sha at load (bb_train, ~15:53Z): bc161a92… / unit_ids d4904e13…, equal to CLUSTER-A and to the post-incident VPS file. The run is not restarted.
- Load-time gate for VPS cmds (bb_train is sha-pinned, so the gate is in the cmd): before the race lock and on every restart, python computes the SHA256 of the targets .npy and compares it with CLUSTER-A DONE.json. Mismatch → exit /b 34, logged in job_history.txt.
- Waiter v1 24845046 and v2 24846213 were cancelled before enqueuing anything. Waiter v3 24846288 (vps_enq_cradio_v3.sbatch, sha256 4062e72d…):
  - Begins 18:30Z, so it does not compete with the CLUSTER-B relay for hub bandwidth.
  - Handles cmds sequentially in verdict-rank order.
  - Pushes final files only when the VPS sha differs, re-checks up to 3 attempts, checks unit_ids, inserts the gate, then enqueues.

## B4. Verdict batch 5 and VPS cmd numbering
- PASS_XS: c-radiov3-h 3051231558, c-radiov3-g 3051222553, c-radiov4-h 3051206532 (file read 16:30Z; the lead's message had PENDING), c-radiov3-l 3051167544, c-radiov4-so400m 3051158522, c-radiov3-b 3051143538. The controller ranks by priority_score automatically (no change).
- VPS cmds by rank (the runner runs lexical order after the running 440):
  - 441 c-radiov3-h_W 8fa190f4…, 442 c-radiov3-g_W 99b87b1f…, 443 c-radiov4-h_W b826f98d…, 444 c-radiov3-l_W 0dea9865…, 445 c-radiov4-so400m_W 34595cda…, 446 c-radiov3-b_W 448b7617…. These are sha256 before the gate is inserted; the gated sha is logged by the waiter.
  - 407/408 superseded (renamed *.superseded_by_444/446, never enqueued).
- CLUSTER-A WARM copies of the C-RADIO arms were raised to 2 GPUs: 24843727–24843732. They remain subject to the 4-GPU cap. No running arm was preempted (A4).
- Licences:
  - nvidia/C-RADIOv3-*: license other, license_name nvidia-open-model-license, link https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf (README front matter of the -b/-l dirs; read for -h/-g/v4 in the provenance file).

## B5. CLUSTER-A GPU cap question (lead ~16:45Z) — not changed
- The lead asked to raise the CLUSTER-A controller cap toward the account QoS (sacctmgr u-clusterA gres/gpu=16).
- The frozen prereg A4.2 (plans/FMD_PREREG_2026-11-03.md line 221) fixes "CLUSTER-A at most 6 GPUs for the whole FMD lane"; CAPACITY_NOTE.md line 237 says "cap <= 6 GPUs". The controller --cap 4 counts fx_ arm GPUs only.
- At 16:48Z the FMD tracks held 6 CLUSTER-A GPUs: gv_diffx 1, gv_zs 1, fmdx_enc_w 1, fmdx_enc_smoke 1, fx_NOFM_W 2. Final-code smoke 24843695 (2 GPU) was cancelled at 16:49Z after all train-path checks had passed (sample/score/select passed in 24843130).
- sbatch --test-only, 2-GPU gpuidle: earliest start 2026-11-04T06:47Z.
- Cap left at 4 pending a prereg amendment from the lead. If one arrives, the controller becomes v1.2: lane-wide GPU count, ≥ 2 kept for the screen workers.

## B6. CLUSTER-B GPU precedence (lead ~16:52Z), applied with scontrol nice to this track's jobs only
- Extraction jobs: 3700746 (v3-h) 0, 3700747 (v3-g) 1, 3700748 (v4-h) 2, 3700646 (v3-l, v3-b) 3, 3700749 (v4-so400m) 4.
- WARM copies: v3-h 3700361 100, v3-g 3700362 110, v4-h 3700364 120, v3-l 3700360 130, v4-so400m 3700363 140, v3-b 3700359 150.
- PEINIT_W 3700351: nice 90. A4.3(b) puts controls at the top; this deviates from the lead's order, and the lead was informed.
- All other CLUSTER-B copies of this track stay held.

## B7. Corrections to earlier entries (reviewer findings, 17:15Z)
- Addendum 1 A4 and the lead message of 15:53Z said the GPU job 24845031 had a "gpuidle start estimate of 2026-11-04T00:00Z". The squeue START_TIME of 24845031 was 2026-11-04T04:47:00, CLUSTER-A local time (CET), i.e. 03:47Z. The ~00:00 figure came from a separate sbatch --test-only probe (job 24845056 with the same node exclusions: 2026-11-04T00:01:47 CET = 23:01Z on 3 Nov).
- Addendum 1 A5: "fp32 rows, array 24844928" is the lead's statement (lead message ~15:35Z). It was not verified by this track. Job 24844928 is named gv_diffx in squeue.
- arm_log.md line of 15:20Z: the CLUSTER-B controller sbatch sha256 after the retarget and continuation guard is 5485b5025688308296e412f66c160c869a19844f2e3e1dc9ad046801825e7e9d. 7cedb4b1… was an intermediate edit.
