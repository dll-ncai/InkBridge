# DECISION_RECORD_backbone — addendum 3 (2026-11-03 ~16:45–17:10Z, track 'Backbone build')
Follows addendum 2 (sha256 ef00320cb60de9f40b82d02bfbe575b91604d9b10829e8a8d6f7de0722e3bdf2). The frozen DR is unchanged.

## C1. Prereg A7: CLUSTER-A FMD-lane GPU cap 14 (<author> approved; lead relay ~16:44Z)
- Prereg exp/FMD/xs/plans/FMD_PREREG_2026-11-03.md is now sha256 b5af7ee97706d4a7c8d5c880c7612b769712e9d1a41de4808ee4158e95fd04c8 (A7 appended by the lead).
- PREREG_FROZEN.flag sha256 703dda2c4dc78db8d792c0484382d9aff0379f2e454fcea29f37527ecba457d3; pre-A7 flag copy is PREREG_FROZEN.flag.pre_A7.
- A7: the cap counts ALL FMD/FMDX GPU jobs (arms, controls, screen workers, smokes, extraction) and leaves 2 of the QoS 16 free.
- Hub copies (push prereg_A7, manifest sha256 896b8160…) were sha-checked on both sides:
  - CLUSTER-B pull 16:45:26Z "3 files sha256 OK"; CLUSTER-B sha256sum = b5af7ee9… / 703dda2c….
  - VPS stage Get-FileHash = b5af7ee9… / 703dda2c….

## C2. Controller v1.2 (exp/FMD/xs/code/fmdx_controller_v1.2; diff controller_v1.1_to_v1.2.diff)
- `--lane-wide` (CLUSTER-A only): the cap counts the GPUs of every RUNNING/CONFIGURING/COMPLETING GPU job of the account, from squeue `%b`. This is conservative: it would include a non-FMD GPU job of the account, if any.
- Released-but-pending fx_ jobs take slots as in v1.1. The breakdown is logged on every change ([cap] lines).
- Rank order is unchanged:
  1. Controls first (A4.3 b).
  2. FM arms by verdict priority_score desc.
  3. WARM before SCRATCH.
- WARM arms with inputs present are released in rank order as their DONE.json lands.
- A WARM arm of a control or of a PASS* FM whose targets are not DONE yet reserves its GPUs against SCRATCH arms only. SCRATCH arms therefore start only from GPUs not needed by the higher-ranked WARM arms.
- Running jobs are never touched. Other tracks' jobs are only counted.
- Build history (all logged in controller_log.md):
  - 24846583 (gate_controller.py 25db8876…, no reservation) released SCRATCH arms NOFM_S 24843785, PEINIT_S 24843790, c-radiov3-b_S 24843807 and dinov3-vitl16_S 24843787 at 16:46Z. All 4 were re-held by hand at 16:47Z, before any started.
  - 24846598 (5f7bba5e…) reserved against WARM arms too and held c-radiov3-b_W.
  - Final: 24846629 (gate_controller.py bd3365c36cee4dc4679da51062556c7de1671a968326ce51a89e47df3d0384d6, started 16:57:52Z). First cycle: lane-wide 5 of 14 (fmdx_enc_w 1, fx_NOFM_W 2, gv_diffx 1, gv_zs 1); released fx_c-radiov3-b_W 24843727 (rank 8). SCRATCH arms stay held.
- v1.1 → v1.2 restart: the v1.1 relay push of c-radiov3-b to the hub (pid 1765039) died with job 24844303. A dry-run test had recorded its rc as 0 because it was not that process's child. The state entry is now "killed_at_controller_restart_v1.2" and hub/targets_ready.txt is empty. The VPS copy goes through waiter v3 with its sha gate; CLUSTER-B does not use the relay (addendum 2 B1).

## C3. Hub streams (lead 16:45Z: hub stalled at 16:43Z with 3 parallel streams; one stream per track)
- This track: refpush 24846256 finished 16:43:28Z; prereg push at 16:45:03Z.
- The CLUSTER-B extraction inputs are pulled by ONE sequential login-node stream (code/_jobs/clusterB_tgt/pull_seq_1.sh: prereg_A7, then clusterB_in_c-radiov3-h, -g, c-radiov4-h, -so400m). It sets the .PULLED markers, so the CLUSTER-B extraction jobs do not pull in parallel when they start.
- Waiter v3 24846288 (begins 18:30Z) is sequential.
- CLUSTER-B PEINIT_W 3700351 stays at nice 90 (A4.3 b), as the lead confirmed.

## C4. hpl_tamil pseudo-pool failures: cause, and redraw per lead case (b)
- Diagnostic tamil_diag.py (job 24846587; code/_jobs/tamil_diag; output data/pseudo/tamil_diag.json, 25 tracebacks kept):
  - It re-derived the same drawn 20,000 (sha256 of the id list 3a466618…) and reproduced 7,422 ok / 12,578 failed.
  - Exception histogram: 12,578 × `ValueError: empty ink`, raised in m5_build.encode_raw.
  - Every failing id inspected is a macOS AppleDouble file `._<name>.txt` (e.g. usr_100/._000t01.txt). The pool glob `*.txt` in prep_gt_ink.pool matched these. They are metadata files, not ink, so this is not a read_unipen/encode_raw bug.
- Lead rule (b) applied:
  1. The 7,422 v1 successes are kept (rows 0..7421 of pseudo_pool_hpl_tamil.npz unchanged; the v1 file is kept as pseudo_pool_hpl_tamil.v1_7422.npz).
  2. Candidates = pool minus the 20,000 v1-drawn ids, sorted. Order = numpy default_rng(20261103).permutation(len(candidates)).
  3. Candidates are rebuilt in that order with the same reader path and accepted until 20,000 succeed or the pool is exhausted. Failures during the redraw, AppleDouble included, are counted in pseudo_pool_hpl_tamil_redraw.json.
- The script checks that rows 0..7421 of img224/cond64 are byte-identical in the combined file, and that sample new rows unpack to their rebuilt ink.
- Jobs: tamil_extra.py (sha256 fef9b034…) 24847074 → InkSight extra array 24847075 (manifest_tamil_extra.json, 0-99%50, nice 200; data/pseudo/inksight_extra/) → build v1.0.1 24847076 (out-glob `inksight*/out_*.jsonl`; afterany 24844988 + 24847075). The v1 build 24844989 was cancelled.
- pseudo_pool_summary.json gets hpl_tamil_v1 (old) and hpl_tamil (with the redraw record).
- casia_en: 8,699 = the whole eligible pool; recorded as is.
- The fix for prep_gt_ink's glob is not applied to the sha-pinned code. The redraw handles the AppleDouble entries by the lead's failure rule.

## C5. Other events 17:00–17:20Z
- c-radiov3-l CLUSTER-A merge 24845068_3 failed at `os.replace(tokens.npy → final)` with OSError [Errno 16] EBUSY on $SCRATCH_ROOT, after tokens.npy (20,971,520,128 B) was complete. MERGE.lock and tokens.npy were removed; the merge was re-run as 24847054. All 157 chunks were already present.
- The CLUSTER-B login pull (nohup) of the extraction inputs died silently after the prereg step. It was restarted inside tmux (session fmdx_pull). Log logs/targets/pull_seq_1.log: clusterB_in_c-radiov3-h, -g and c-radiov4-h were pulled with sha256 OK by 17:18:30Z.

## C6. AppleDouble provenance check (lead ~17:25Z)
- AppleDouble entries `._<name>.txt` in data/hpl_tamil/extracted/…/new/usr_*/ are macOS resource-fork metadata files, not ink.
- They were drawn into the v1 Tamil pseudo pool because prep_gt_ink.pool lists Tamil files with the glob `*.txt`. read_unipen returns no pen strokes for them, so encode_raw raises "empty ink".
- The lead accepted the redraw. Continuing the same permutation order equals taking the first 20,000 valid units in that order.
- Check script appledouble_check.py (sha256 4ee7fdbbffb3f41ad01bc1ba8bcc73a9592c1f76f6fc1133d5e5d60de26240a6), output data/pseudo/appledouble_check.json. It counts ids with a `._` name component (regex `(^|[|/_\\])\._`). Result: 0 in every set.
  - m5_luo_train: 0/80,000 (each source 0/20,000).
  - xs_render224 val250 sets: casia_zh, casia_en, hpl_tamil, iam each 0/250. unipen_de tr3927: 0/3,927.
  - Pseudo pools: casia_zh 0/20,000, casia_en 0/8,699, iam 0/20,000, hpl_tamil (v1) 0/7,422.
  - luo_repa_order_v1.csv: 0/80,000. Its S40k subset: 0/40,000.
