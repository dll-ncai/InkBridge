# DECISION_RECORD_backbone — addendum 1 (2026-11-03, track 'Backbone build')
The frozen record DECISION_RECORD_backbone.md (sha256 42dab0b645e725e9e8fbbae3cc2a79edd08ab46c7aaa745fe2d88fb8812b3c3f) is unchanged. Entries below are later events, in time order (UTC).

## A1. CLUSTER-B controller partition (15:20Z)
- Job 3700460 (partition batch) had a start estimate of 2026-12-05; cancelled.
- clusterB_controller_v1.1.sbatch now requests `-p A100-RP,H100-RP --requeue --time=1-00:00:00`, CPU only (1 cpu, 2 GB). A continuation is queued only if no fmdx_controller job is pending. That guard prevents duplicate continuations after a preemption requeue.
- New job: 3700469. Controller SHA256SUMS.txt updated: clusterB_controller_v1.1.sbatch sha256 5485b5025688308296e412f66c160c869a19844f2e3e1dc9ad046801825e7e9d. gate_controller.py is unchanged (95d613fa…).

## A2. VPS runner cmd verdict path (15:22Z)
- gen_vps_cmd.py (code v1, sha-pinned) writes `--verdict-file %STG%\exp\FMD\xs\verdicts\verdict_<fm>.json`. That path does not exist on the VPS stage.
- The controllers keep the live verdict copies in `%STG%\exp\FMD\xs\hub\verdicts\`. Generated cmds are patched after generation (sed) to that path.
- 440_fx_dinov3-vitl16_W.cmd: v0 sha256 c4c63194… → patched b8dfd275ade008a7fbebf32a01be0c4102673c689d4f9a2553cd78257f267271.
- 407_fx_c-radiov3-l_W.cmd: sha256 0dea9865ad7a10d58d32c2d62e7b87b186625321c03418fb42d0a7f24ffc8fca.
- 408_fx_c-radiov3-b_W.cmd: sha256 448b7617226c047434228a59d376248a5a386b26b1c16958098856f6eb9ef4c1.
- bb_train.read_verdict treats a missing file as MISSING (not FAIL). Only status FAIL stops an arm.

## A3. InkSight pseudo-label pipeline gate bug (15:22Z)
- The v1.0 jobs clusterA_pseudo_pool.sbatch and clusterA_pseudo_inksight.sbatch exited "verdict FAIL -> skipped": 24843885, array 24843886 (all 400 tasks, 6–7 s each), and build 24843887 (FAILED).
- Cause: the inline check `json.load(open(verdict_inksight_small_p.json))` raised FileNotFoundError, which gives python exit code 1, the same code used for FAIL. No InkSight verdict file exists.
- Fix: job files in code/fmdx_pseudo_jobs_v1.0.1/, outside the sha-pinned code dir (diff pseudo_v1.0_to_v1.0.1.diff). Only an existing file with status FAIL skips (exit code 10). A missing or unreadable file continues, and the check's rc is logged.
- Resubmitted: pool 24844987 → InkSight array 24844988 (0-399%100) → build 24844989 (unchanged v1 build sbatch).
- The v1.0 marker data/pseudo/SKIPPED_GATE_FAIL.txt was renamed to SKIPPED_GATE_FAIL.v1.0_missing_file_bug.txt.

## A4. C-RADIOv3-L and C-RADIOv3-B PASS_XS (lead, 15:2xZ)
- verdict_c-radiov3-l.json: status PASS_XS, priority_score 3051167544, sha256 1f18c1f6db471657a005de042d9a711d238fcd8970bb69d2800aa0122e946f43 (read 15:3xZ).
- verdict_c-radiov3-b.json: status PASS_XS, priority_score 3051143538, sha256 2d24e2d10de4a687988c4aee155fd2e46b20c16b8aff2e0bd52540807223b75e (read 15:3xZ). The lead quoted sha256 0fbb9ed9… earlier; the file was rewritten between the two reads.
- Licence (model card README front matter of both weight dirs exp/FMD/xs/weights/c-radiov3-{b,l}): `license: other`, `license_name: nvidia-open-model-license`, `license_link: https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf`.
- Repo nvidia/C-RADIOv3-B revision 44653a0482cf460bb4f12595fc3cc3dfecc403d1 (from the verdict).
- Ranking: controller v1.1 rank_key = A3.6 priority_score desc, so c-radiov3-l is first FM arm and c-radiov3-b second, after the controls NOFM, PEINIT (A4.3 b). The re-rank is automatic on every cycle from the verdict files. No code change.
- S40k REPA targets:
  - GPU tasks 24843687_30/_31 were cancelled; the GPU resubmission 24845031 had a gpuidle start estimate of 2026-11-04T00:00Z and was cancelled.
  - CPU-sharded extraction runs as 24845068 on cpuidle. Tasks 1-8: c-radiov3-l 8 shards; tasks 9-12: c-radiov3-b 4 shards. The script is code/_jobs/clusterA_targets_cpu.sbatch (sha256 98ba2614…), which is clusterA_targets.sbatch with a CPU header. The extractor selects cpu when no GPU is visible; the device is recorded in the target meta.
  - The main target array throttle was changed 3 → 1 (FMD GPU cap).
- CLUSTER-A warm copies fx_c-radiov3-l_W 24843728 and fx_c-radiov3-b_W 24843727 were changed from 1 to 2 GPUs (scontrol TresPerNode gres/gpu:2; arm.sh derives NG from CUDA_VISIBLE_DEVICES; A6 layout for <40 GB: 32×16/NG).
- CLUSTER-B copies: c-radiov3-l_W 3700360, c-radiov3-b_W 3700359 (held until the targets arrive on CLUSTER-B). VPS copies: cmds 407/408 (lexically before 440, so they run before dinov3-vitl16_W on the VPS). They are enqueued by waiter job 24845046 once the VPS stage .npy sha256 equals CLUSTER-A's DONE.json sha256.
- SCRATCH arms c-radiov3-l_S (CLUSTER-A 24843808, CLUSTER-B 3700411) and c-radiov3-b_S (CLUSTER-A 24843807, CLUSTER-B 3700410) remain queued. Their nice is set by the controller rank.

## A5. lightonocr-2-1b (lead, 15:3xZ)
- verdict_lightonocr-2-1b.json is FAIL (sha256 1ea17ad4… read 15:27Z; b74e3cf1d318f834e32b6c39de66d83ef9292e46e925b28715760e01c1351ee3 at the controller's cancel).
- The lead first asked to hold off cancellation pending an fp32 re-extraction, then confirmed the FAIL as valid (fp32 rows, array 24844928). Its arms were cancelled under A4.3(a):
  - CLUSTER-A controller: 24843771 fx_lightonocr-2-1b_W at 15:36:16Z and 24843845 _S at 15:37:03Z.
  - CLUSTER-B, manual (CLUSTER-B controller not yet running): 3700397 _W and 3700449 _S, logged in CLUSTER-B admin/cancellation_log_clusterB.md.

## A6. Gated FMs (lead relay)
- <author> accepted the HF licences for sd3-medium-diffusers, flux1-dev, flux1-schnell, sam3 and sd3.5.
- Their arms stay BLOCKED (held) until weights exist and their verdict is not BLOCKED.
- flux1-dev is under the FLUX.1 [dev] Non-Commercial License. Its checkpoints stay on CLUSTER-A/CLUSTER-B/VPS and are not redistributed.
