# WARM_MONITOR_STATUS (facts only) - autopilot 2026-11-10T18:07:18Z
Track: FMDX warm monitor (replacement instance, <agent>; reason: <agent> API failing). Prereg: FMD_PREREG_2026-11-03.md incl. A9/A9.4 (sha in PREREG_FROZEN.flag).
Autopilot: CLUSTER-A cpuidle job (code/fmdx_monitor_v1/fxm_autopilot.py); sampling on CLUSTER-A only (lead 8 Nov); no CLUSTER-A<->CLUSTER-B relays.

| arm | bb ms | FT ms | preds bb/ft/cmp | rows bb/ft/cmp | since |
|---|---|---|---|---|---|
| NOFM_W | 3 | 1 | 18/24/18 | 18/24/18 |  |
| PEINIT_W | 4 | 2 | 18/24/18 | 18/24/18 |  |
| c-radiov3-b_W | 4 | 3 | 18/24/18 | 18/24/18 |  |
| c-radiov3-h_W | 4 | 4 | 18/24/18 | 18/24/18 |  |
| c-radiov4-so400m_W | 4 | 3 | 18/24/18 | 18/24/18 |  |
| cleandift-sd21_W | 4 | 3 | 18/24/18 | 18/24/18 |  |
| dinov3-vitb16_W | 4 | 2 | 18/24/18 | 18/24/18 | 2026-11-10T09:16:30Z |
| dinov3-vitl16-sat493m_W | 4 | 5 | 18/24/18 | 18/24/18 |  |
| pe-spatial-b16-512_W | 4 | 2 | 18/24/18 | 18/24/18 |  |
| vit-mae-large_W | 4 | 5 | 18/24/18 | 18/24/18 |  |
| webssl-mae1b-224_W | 4 | 3 | 12/24/0 | 0/24/0 |  |
| webssl-mae300m-224_W | 4 | 2 | 18/24/18 | 18/24/18 |  |

## Last table builds
- 2026-11-10T11:22:00Z job 25039395: NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb
- 2026-11-10T13:25:15Z job 25042310: NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-b_W:cmp,c-radiov3-b_W:ft,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,c-radiov4-so400m_W:cmp,c-radiov4-so400m_W:ft,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb,webssl-mae300m-224_W:cmp,webssl-mae300m-224_W:ft
- 2026-11-10T16:27:00Z job 25049213: NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-b_W:cmp,c-radiov3-b_W:ft,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,c-radiov4-so400m_W:cmp,c-radiov4-so400m_W:ft,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitb16_W:cmp,dinov3-vitb16_W:ft,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb,webssl-mae300m-224_W:cmp,webssl-mae300m-224_W:ft

## Running arms (fx_* on CLUSTER-A)
```

```

## Autopilot events (last 40)
```
- 2026-11-10T09:20:31Z [FMDX warm-monitor autopilot, <agent> track] A9 sampling race copies (3 earliest test-only) webssl-mae300m-224_W ft ms2 array 0-2: ['gpuidle:gpu:a30:1=25037191 (est 2026-11-12T08:18:31)', 'gpuidle:gpu:a4000:1=25037192 (est 2026-11-12T08:18:31)', 'gpuidle:gpu:v100:1=25037193 (est 2026-11-12T08:18:31)']
- 2026-11-10T09:20:40Z [FMDX warm-monitor autopilot, <agent> track] scoring webssl-mae300m-224_W bb: adapt=adapted 18 job 25037194
- 2026-11-10T09:20:40Z [FMDX warm-monitor autopilot, <agent> track] c-radiov3-b_W FT selected ms3 (val1000 E0); scancel issued for FT copies ['24970453', '24970454', '24970455'] (winner in ft/c-radiov3-b_W_FT_clusterA/job_history.txt)
- 2026-11-10T09:20:41Z [FMDX warm-monitor autopilot, <agent> track] A9 sampling race copies (3 earliest test-only) c-radiov3-b_W ft ms3 array 0-2: ['gpuidle:gpu:a30:1=25037203 (est 2026-11-12T08:30:40)', 'gpuidle:gpu:a4000:1=25037204 (est 2026-11-12T08:30:40)', 'gpuidle:gpu:v100:1=25037205 (est 2026-11-12T08:30:40)']
- 2026-11-10T09:20:57Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov3-b_W bb: adapt=adapted 18 job 25037303
- 2026-11-10T09:20:57Z [FMDX warm-monitor autopilot, <agent> track] c-radiov4-so400m_W FT selected ms3 (val1000 E0); scancel issued for FT copies ['24970473', '24970474', '24970478'] (winner in ft/c-radiov4-so400m_W_FT_clusterA/job_history.txt)
- 2026-11-10T09:21:08Z [FMDX warm-monitor autopilot, <agent> track] A9 sampling race copies (3 earliest test-only) c-radiov4-so400m_W ft ms3 array 0-2: ['gpuidle:gpu:a30:1=25037323 (est 2026-11-12T12:18:03)', 'gpuidle:gpu:v100:1=25037324 (est 2026-11-12T12:18:08)', 'gpuidle:gpu:a4000:1=25037325 (est 2026-11-12T18:13:45)']
- 2026-11-10T09:21:18Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov4-so400m_W bb: adapt=adapted 18 job 25037326
- 2026-11-10T09:21:18Z [FMDX warm-monitor autopilot, <agent> track] A9 race dinov3-vitb16_W bb: 3 seed locks taken; cancelled pending tasks ['25036933_0', '25036933_1', '25036933_2']
- 2026-11-10T10:01:25Z [FMDX warm-monitor autopilot, <agent> track] tables job 25038192 for NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb
- 2026-11-10T10:21:27Z [FMDX warm-monitor autopilot, <agent> track] A9 race webssl-mae300m-224_W ft: 3 seed locks taken; cancelled pending tasks ['25037191_0', '25037191_1', '25037191_2']
- 2026-11-10T10:31:33Z [FMDX warm-monitor autopilot, <agent> track] A9 race c-radiov3-b_W ft: 3 seed locks taken; cancelled pending tasks ['25037203_0', '25037203_1', '25037203_2']
- 2026-11-10T10:31:47Z [FMDX warm-monitor autopilot, <agent> track] scoring dinov3-vitb16_W bb: adapt=adapted 18 job 25038333
- 2026-11-10T11:11:57Z [FMDX warm-monitor autopilot, <agent> track] A9 race c-radiov4-so400m_W ft: 3 seed locks taken; cancelled pending tasks ['25037323_0', '25037323_1', '25037323_2']
- 2026-11-10T11:22:00Z [FMDX warm-monitor autopilot, <agent> track] tables job 25039395 for NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb
- 2026-11-10T12:02:11Z [FMDX warm-monitor autopilot, <agent> track] scoring webssl-mae300m-224_W ft: adapt=adapted 24 job 25040367
- 2026-11-10T12:02:53Z [FMDX warm-monitor autopilot, <agent> track] composed preds (A4.4) webssl-mae300m-224_W: german_onhw_all__webssl-mae300m-224_W_cmp_ms2__seed2 n_ft 3 of 5925
composed files 18
- 2026-11-10T12:02:58Z [FMDX warm-monitor autopilot, <agent> track] scoring webssl-mae300m-224_W cmp: adapt=adapted 18 job 25040380
- 2026-11-10T12:03:02Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov3-b_W ft: adapt=adapted 24 job 25040381
- 2026-11-10T12:03:48Z [FMDX warm-monitor autopilot, <agent> track] composed preds (A4.4) c-radiov3-b_W: german_onhw_all__c-radiov3-b_W_cmp_ms3__seed2 n_ft 3 of 5925
composed files 18
- 2026-11-10T12:03:55Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov3-b_W cmp: adapt=adapted 18 job 25040411
- 2026-11-10T12:54:19Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov4-so400m_W ft: adapt=adapted 24 job 25041141
- 2026-11-10T12:55:00Z [FMDX warm-monitor autopilot, <agent> track] composed preds (A4.4) c-radiov4-so400m_W: german_onhw_all__c-radiov4-so400m_W_cmp_ms3__seed2 n_ft 3 of 5925
composed files 18
- 2026-11-10T12:55:03Z [FMDX warm-monitor autopilot, <agent> track] scoring c-radiov4-so400m_W cmp: adapt=adapted 18 job 25041174
- 2026-11-10T13:05:09Z [FMDX warm-monitor autopilot, <agent> track] dinov3-vitb16_W FT selected ms2 (val1000 E0); scancel issued for FT copies ['25036922', '25036923', '25036924'] (winner in ft/dinov3-vitb16_W_FT_clusterA/job_history.txt)
- 2026-11-10T13:05:12Z [FMDX warm-monitor autopilot, <agent> track] A9 sampling race copies (3 earliest test-only) dinov3-vitb16_W ft ms2 array 0-2: ['b200idle:gpu:b200:1=25041706 (est 2026-11-11T00:39:21)', 'gpuidle:gpu:a4000:1=25041707 (est 2026-11-12T20:09:10)', 'gpuidle:gpu:v100:1=25041708 (est 2026-11-12T20:09:11)']
- 2026-11-10T13:15:13Z [FMDX warm-monitor autopilot, <agent> track] A9 race dinov3-vitb16_W ft: 3 seed locks taken; cancelled pending tasks ['25041706_0', '25041706_1', '25041706_2', '25041708_0', '25041708_1', '25041708_2']
- 2026-11-10T13:25:15Z [FMDX warm-monitor autopilot, <agent> track] tables job 25042310 for NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-b_W:cmp,c-radiov3-b_W:ft,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,c-radiov4-so400m_W:cmp,c-radiov4-so400m_W:ft,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb,webssl-mae300m-224_W:cmp,webssl-mae300m-224_W:ft
- 2026-11-10T14:17:18Z [FMDX warm-monitor, <agent>] webssl-mae1b-224_W bb: seed0 complete (owner 24969745_0); cancelled stale seed-0 copy 24969743_[0] (est 10 Dec) so the autopilot resubmits missing seeds 1-2
- 2026-11-10T14:25:26Z [FMDX warm-monitor autopilot, <agent> track] A9 sampling race copies (3 earliest test-only) webssl-mae1b-224_W bb ms4 array 1: ['b200idle:gpu:b200:1=25044497 (est 2026-11-11T00:39:21)', 'gpuidle:gpu:a4000:1=25044498 (est 2026-11-12T19:50:26)', 'gpuidle:gpu:v100:1=25044499 (est 2026-11-12T19:50:26)']
- 2026-11-10T14:25:26Z [FMDX warm-monitor autopilot, <agent> track] webssl-mae1b-224_W bb seed1: stale lock or nothing queued; resubmitted (repair 1)
- 2026-11-10T15:26:03Z [FMDX warm-monitor autopilot, <agent> track] scoring dinov3-vitb16_W ft: adapt=adapted 24 job 25047805
- 2026-11-10T15:26:44Z [FMDX warm-monitor autopilot, <agent> track] composed preds (A4.4) dinov3-vitb16_W: german_onhw_all__dinov3-vitb16_W_cmp_ms2__seed2 n_ft 3 of 5925
composed files 18
- 2026-11-10T15:26:47Z [FMDX warm-monitor autopilot, <agent> track] scoring dinov3-vitb16_W cmp: adapt=adapted 18 job 25047838
- 2026-11-10T16:27:00Z [FMDX warm-monitor autopilot, <agent> track] tables job 25049213 for NOFM_W:bb,NOFM_W:cmp,NOFM_W:ft,PEINIT_W:bb,PEINIT_W:cmp,PEINIT_W:ft,c-radiov3-b_W:bb,c-radiov3-b_W:cmp,c-radiov3-b_W:ft,c-radiov3-h_W:bb,c-radiov3-h_W:cmp,c-radiov3-h_W:ft,c-radiov4-so400m_W:bb,c-radiov4-so400m_W:cmp,c-radiov4-so400m_W:ft,cleandift-sd21_W:bb,cleandift-sd21_W:cmp,cleandift-sd21_W:ft,dinov3-vitb16_W:bb,dinov3-vitb16_W:cmp,dinov3-vitb16_W:ft,dinov3-vitl16-sat493m_W:bb,dinov3-vitl16-sat493m_W:cmp,dinov3-vitl16-sat493m_W:ft,pe-spatial-b16-512_W:bb,pe-spatial-b16-512_W:cmp,pe-spatial-b16-512_W:ft,vit-mae-large_W:bb,vit-mae-large_W:cmp,vit-mae-large_W:ft,webssl-mae1b-224_W:ft,webssl-mae300m-224_W:bb,webssl-mae300m-224_W:cmp,webssl-mae300m-224_W:ft
- 2026-11-10T17:13:29Z [warm-monitor, <agent>] end of run: HANDOFF_scratch_monitor.md + PROVENANCE_FMDX_warm.json + FMDX_not_assessed_A43f.csv written; final tables job 25049213

```