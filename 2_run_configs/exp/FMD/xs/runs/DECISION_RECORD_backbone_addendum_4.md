# DECISION_RECORD_backbone_addendum_4 (FMDX warm monitor track)
Facts only. Written by the warm-monitor track (<agent>; <agent> chosen because monitoring/existing recipes, per <author>'s 30 Oct model policy). Source of each decision: lead session directive (<agent>), received 2026-11-03 between 18:2xZ and 2026-11-03T18:26:10Z.
Prereg sha256 b5af7ee97706d4a7c8d5c880c7612b769712e9d1a41de4808ee4158e95fd04c8 (A4.4/A4.5 govern the additions below).

## D4.1 (lead, 2026-11-03 ~18:3xZ) VPS cmd numbers
Do not use 410, 420, 430 (listed USED in CAPACITY_NOTE). Monitor sampling cmds use 411-419, 421-429, 431-439 (sort before 440). Code: fmdx_monitor_v1/vps_sample_cmd.py asserts this range.

## D4.2 (lead) scope additions to the warm-monitor brief
- A4.4 composed rows are REQUIRED for cell X: ScriptCNN router exp/D3_ROUTE/router/script_clf.pt (sha256 22bb8802...), rtl-threshold 0.5; p_rtl >= 0.5 -> the arm's Urdu-FT preds, else the arm's selected backbone preds; routes written per set. Backbone-only rows reported next to composed rows.
- A4.5 forgetting read: arm FT (no router) vs COMP-p01 seed-0 rows on the six non-Urdu sets.
- Therefore for every arm the FT model is sampled on ALL units of the six non-Urdu test sets, seeds 0-2, on the VPS 4090 (in addition to backbone on the six sets, FT on test_ouhdl and val1000). Per-arm 4090 budget ~1.7-2.5 h (lead's figure); 10 arms ~17-26 h.

## D4.3 (lead) 4090 reserved for sampling; VPS waiter cancelled
- Action by this track, 2026-11-03T18:25:41Z (CLUSTER-A clock): `scancel 24846288` (fmdx_vps_enq3, state before: PENDING, reason BeginTime 2026-11-03T19:30 CLUSTER-A-local = 18:30Z 3 Nov). rc=0; sacct: CANCELLED.
- State read right after (VPS runner, 2026-11-03 ~18:26Z): runner\queue empty; runner\running = 440_fx_dinov3-vitl16_W.cmd; race dirs: dinov3-vitl16_W owner=vps, NOFM_W owner=clusterA 24843707; no race\c-radio* dir exists; no 441-446 cmd in the queue.
- 440 (dinov3-vitl16_W) left running. The CLUSTER-A/CLUSTER-B copies of all C-RADIO warm arms remain queued. The VPS no longer races for C-RADIO WARM training.
- Not done by this track: pushing REPA targets to the VPS for 441-446 (was the waiter's job); any 441-446 cmd file.

## D4.4 (lead) FT lanes
FT + its in-job val1000 selection run on CLUSTER-A/CLUSTER-B GPUs (not the 4090). The 4090 sampling cmds (411-439) are queued as each arm becomes ready, in X/admin/priority.json order.
