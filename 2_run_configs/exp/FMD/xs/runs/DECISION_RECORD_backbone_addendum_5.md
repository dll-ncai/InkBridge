# DECISION_RECORD_backbone_addendum_5 (FMDX warm monitor track)
Facts only. Written by the warm-monitor track (<agent>; monitoring with existing recipes). Reason for D5.1-D5.3: <author> decision 3 Nov ~23:50Z (FMDX sampling first on the 4090), relayed by the lead session.
## D5.1 Dispatcher
- One VPS runner cmd 411_fxm_dispatcher.cmd (sha256 cc39018b87b097a16cbfd8a72dd527fc58ba34f55a6a8533752f06b1c74f9f70) runs fmdx_dispatcher.ps1 (sha256 1637f76971a67653f8cdcffead46b4e50d98b1903ced63b968148920b5a84fd5) in the runner. It sorts before 451_order_oh_k1_rtlpe_vpstrain.cmd; 451 and every other track's cmd are untouched.
- Loop: lowest-named *.cmd in stage\exp\FMD\xs\runs\fmdx_monitor\dispatch\ready (names = priority.json order, backbone before FT) -> run -> done\ or failed\. Poll for new jobs every 600 s (EXIT/deadline checked every 30 s). Exit: UTC >= 2026-11-07T06:00:00Z (checked between jobs), or ready\ empty and file EXIT present (the monitor writes EXIT when every PASS_XS warm arm + NOFM + PEINIT is sampled or declared not-assessed).
- Log: dispatch\dispatcher_log.txt (start, each job start/end/exit/seconds, exit reason); mirrored into WARM_MONITOR_STATUS.md at each poll.
- Smoke test 2026-11-03T23:52Z on the VPS (root dispatch_smoke): job run exit 0 -> done, exit on EXIT with ready empty; second run exits on a past deadline.
- Queued 2026-11-03 ~23:55Z. Runner state then: running 440_fx_dinov3-vitl16_W; queue 411_fxm_dispatcher, 451_order_oh_k1_rtlpe_vpstrain.
## D5.2 Job content
Each ready job = vps_sample_cmd.py output (frozen bb_sample.py): backbone six non-Urdu test sets x seeds 0-2; FT test_ouhdl, val1000 (subset 1000) and the six sets, seeds 0-2 (FT six-set preds feed A4.4 composition on CLUSTER-A).
## D5.3 Supersedes
The per-arm runner cmds 411-439 plan of D4.1/D4.4: runner number 411 is the dispatcher; per-arm jobs go to dispatch\ready (labels 900+).
