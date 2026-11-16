# D3 Stage A — routed ensemble vs joint multi-script model (cross-script table)

**Status: exploratory (single sampling device, 3 seeds); confirmatory re-run pending.** Val and test are read together per the 24 Oct 'results first' mode; no pre-registration.

Date: 2026-10-25. Scoring: uhwr_eval 0.3.1 (import path verified per row), native protocol primary, standardised secondary; per-unit mean over seeds 0-2 first, then writer-cluster percentile bootstrap (B=2000, numpy default_rng(0)), the table-builder definitions (code_7d405fd6). tau_cp = mean over multi-stroke units of tau with undefined tau = 0.

## Setup (facts)

- Sampling: all 75 prediction files sampled on one RTX 4090 (VPS, torch 2.9.1+cu130, fp32, DDIM-50, eta 0, batch 256, `d3a_sample.py` sha256 1a5e378f…), seeds 0, 1, 2; finished 2026-10-24T20:28Z; mirrored to CLUSTER-A `exp/D3_ROUTE/stageA/preds/`.
- Routes: *predicted* = ScriptCNN router (`script_clf.pt` sha256 22bb8802…, manifests `exp/D3_ROUTE/router/manifest_<set>.csv`); *oracle* = true script → specialist. Specialists: E0 ms8 (Urdu), X5 ms9 (IAM / CASIA-en), E27 ms47 (CASIA-zh), M5 joint ms9 (Tamil, German, 'other'). Comparator: M5 joint ms9 on every unit.
- Sets: OUHD-L v3 test 4,406 (32 writers); IAM m5_test_iam500 500 (63); CASIA zh 500 and en 500 (X4a samples, 60 writers each); HPL Tamil 500 (52); UNIPEN-de test 827 (7); OnHW 5,925 (2 writers). The 354-unit gold set was not sampled in Stage A (not in the preds set), so it has no row here.
- Scored units: 136,140 (75 files), 0 scoring errors. CLUSTER-A Slurm array 24619444 (26 tasks, epyc-256). Per-unit rows: `exp/D3_ROUTE/stageA/score_out/rows/`.
- Prediction identity (sha1 of per-unit arrays / per-unit metric rows): on every set the oracle-route file is identical to the home-specialist file, and pred-route units carry the same prediction as the specialist file they were routed to (deterministic per-unit sampling on the 4090). The pred−oracle contrast is therefore purely the mis-routed units.

## 1. Main table — native protocol, seed mean (3 seeds), 95% writer-cluster CI

| set | method | n | AIoU ↑ | LDTW ↓ | DTW ↓ | pen-up F1 v2 ↑ | tau ↑ (n def.) | tau_cp ↑ | direction ↑ | strict order ↑ |
|---|---|---:|---|---|---|---|---|---|---|---|
| OUHD-L Urdu test (4,406) | routed (predicted) | 4406 | 0.7982 [0.7946, 0.8015] | 0.1335 | 49.55 | 0.8412 | 0.8913 [0.8700, 0.9096] (2594) | 0.8592 | 0.9494 | 0.7500 |
| OUHD-L Urdu test (4,406) | routed (oracle) | 4406 | 0.7980 [0.7945, 0.8013] | 0.1306 | 49.23 | 0.8408 | 0.8917 [0.8704, 0.9097] (2592) | 0.8589 | 0.9508 | 0.7500 |
| OUHD-L Urdu test (4,406) | joint M5 ms9 | 4406 | 0.8207 [0.8176, 0.8241] | 0.1350 | 48.21 | 0.8850 | 0.8616 [0.8379, 0.8824] (2657) | 0.8507 | 0.9448 | 0.7661 |
| OUHD-L Urdu test (4,406) | specialist E0 ms8 | 4406 | 0.7980 [0.7945, 0.8013] | 0.1306 | 49.23 | 0.8408 | 0.8917 [0.8704, 0.9097] (2592) | 0.8589 | 0.9508 | 0.7500 |
| IAM English (500) | routed (predicted) | 500 | 0.7920 [0.7876, 0.7960] | 0.2178 | 29.35 | 0.9027 | 0.9434 [0.8961, 0.9790] (216) | 0.8938 | 0.9605 | 0.8867 |
| IAM English (500) | routed (oracle) | 500 | 0.7919 [0.7875, 0.7959] | 0.2184 | 29.43 | 0.9027 | 0.9434 [0.8961, 0.9790] (216) | 0.8938 | 0.9601 | 0.8867 |
| IAM English (500) | joint M5 ms9 | 500 | 0.7941 [0.7910, 0.7971] | 0.2383 | 32.07 | 0.9138 | 0.8965 [0.8483, 0.9418] (221) | 0.8690 | 0.9549 | 0.8733 |
| IAM English (500) | specialist X5 ms9 | 500 | 0.7919 [0.7875, 0.7959] | 0.2184 | 29.43 | 0.9027 | 0.9434 [0.8961, 0.9790] (216) | 0.8938 | 0.9601 | 0.8867 |
| CASIA zh (500) | routed (predicted) | 500 | 0.5512 [0.5465, 0.5559] | 0.7057 | 67.93 | 0.2742 | 0.9404 [0.9223, 0.9565] (462) | 0.8995 | 0.9377 | 0.0147 |
| CASIA zh (500) | routed (oracle) | 500 | 0.5477 [0.5430, 0.5523] | 0.7237 | 66.59 | 0.2665 | 0.9419 [0.9241, 0.9579] (463) | 0.9029 | 0.9403 | 0.0073 |
| CASIA zh (500) | joint M5 ms9 | 500 | 0.7295 [0.7241, 0.7344] | 0.2705 | 75.89 | 0.5520 | 0.9653 [0.9512, 0.9763] (476) | 0.9513 | 0.9601 | 0.3273 |
| CASIA zh (500) | specialist E27 ms47 | 500 | 0.5477 [0.5430, 0.5523] | 0.7237 | 66.59 | 0.2665 | 0.9419 [0.9241, 0.9579] (463) | 0.9029 | 0.9403 | 0.0073 |
| CASIA en (500) | routed (predicted) | 500 | 0.7634 [0.7571, 0.7695] | 0.6443 | 95.00 | 0.7435 | 0.7376 [0.6326, 0.8334] (152) | 0.6126 | 0.9248 | 0.7853 |
| CASIA en (500) | routed (oracle) | 500 | 0.7638 [0.7575, 0.7700] | 0.6390 | 95.47 | 0.7448 | 0.7407 [0.6355, 0.8367] (153) | 0.6193 | 0.9238 | 0.7867 |
| CASIA en (500) | joint M5 ms9 | 500 | 0.7988 [0.7960, 0.8016] | 0.4998 | 74.94 | 0.9298 | 0.9056 [0.8416, 0.9592] (179) | 0.8859 | 0.9820 | 0.9373 |
| CASIA en (500) | specialist X5 ms9 | 500 | 0.7638 [0.7575, 0.7700] | 0.6390 | 95.47 | 0.7448 | 0.7407 [0.6355, 0.8367] (153) | 0.6193 | 0.9238 | 0.7867 |
| HPL Tamil (500) | routed (predicted) | 500 | 0.8127 [0.8099, 0.8158] | 0.0766 | 16.32 | 0.9005 | 0.9809 [0.9595, 0.9963] (221) | 0.9508 | 0.9721 | 0.9400 |
| HPL Tamil (500) | routed (oracle) | 500 | 0.8132 [0.8103, 0.8162] | 0.0738 | 15.92 | 0.9024 | 0.9899 [0.9791, 0.9981] (221) | 0.9596 | 0.9721 | 0.9420 |
| HPL Tamil (500) | joint M5 ms9 | 500 | 0.8132 [0.8103, 0.8162] | 0.0738 | 15.92 | 0.9024 | 0.9899 [0.9791, 0.9981] (221) | 0.9596 | 0.9721 | 0.9420 |
| UNIPEN-de zero-shot (827) | routed (predicted) | 827 | 0.7680 [0.7643, 0.7731] | 0.3544 | 61.81 | 0.6935 | 0.7456 [0.6252, 0.8536] (161) | 0.5335 | 0.8703 | 0.7549 |
| UNIPEN-de zero-shot (827) | routed (oracle) | 827 | 0.7914 [0.7810, 0.8071] | 0.2598 | 48.81 | 0.7411 | 0.7442 [0.6275, 0.9707] (171) | 0.5656 | 0.9118 | 0.7465 |
| UNIPEN-de zero-shot (827) | joint M5 ms9 | 827 | 0.7914 [0.7810, 0.8071] | 0.2598 | 48.81 | 0.7411 | 0.7442 [0.6275, 0.9707] (171) | 0.5656 | 0.9118 | 0.7465 |
| OnHW German zero-shot (5,925) | routed (predicted) | 5925 | 0.7761 [0.7716, 0.7803] | 0.1869 | 37.83 | 0.3983 | 0.8539 [0.8309, 0.8724] (3533) | 0.6596 | 0.9733 | 0.3239 |
| OnHW German zero-shot (5,925) | routed (oracle) | 5925 | 0.7557 [0.7469, 0.7640] | 0.1798 | 37.26 | 0.3768 | 0.8993 [0.8797, 0.9169] (3716) | 0.7306 | 0.9764 | 0.3135 |
| OnHW German zero-shot (5,925) | joint M5 ms9 | 5925 | 0.7557 [0.7469, 0.7640] | 0.1798 | 37.26 | 0.3768 | 0.8993 [0.8797, 0.9169] (3716) | 0.7306 | 0.9764 | 0.3135 |

The specialist rows equal the oracle rows (identical predictions); they are kept so the home specialist is visible. For Tamil and German the oracle route is M5 joint itself.

## 2. Shape factor — routed (predicted) minus joint M5, paired per unit, native

| set | AIoU routed | AIoU joint | Δ AIoU [95% CI] | LDTW routed | LDTW joint | Δ LDTW | Δ pen-up F1 v2 | n pairs | clusters |
|---|---|---|---|---|---|---|---|---:|---|
| OUHD-L Urdu test (4,406) | 0.7982 | 0.8207 | -0.0226 [-0.0249, -0.0204] | 0.1335 | 0.1350 | -0.0015 [-0.0081, +0.0052] | -0.0438 [-0.0518, -0.0369] | 4406 | 32 writers |
| IAM English (500) | 0.7920 | 0.7941 | -0.0020 [-0.0044, +0.0001] | 0.2178 | 0.2383 | -0.0205 [-0.0353, -0.0061] | -0.0110 [-0.0243, +0.0030] | 500 | 63 writers |
| CASIA zh (500) | 0.5512 | 0.7295 | -0.1783 [-0.1857, -0.1703] | 0.7057 | 0.2705 | +0.4352 [+0.4055, +0.4700] | -0.2778 [-0.3070, -0.2485] | 500 | 60 writers |
| CASIA en (500) | 0.7634 | 0.7988 | -0.0355 [-0.0416, -0.0293] | 0.6443 | 0.4998 | +0.1444 [+0.1136, +0.1787] | -0.1863 [-0.2184, -0.1573] | 500 | 60 writers |
| HPL Tamil (500) | 0.8127 | 0.8132 | -0.0005 [-0.0014, +0.0001] | 0.0766 | 0.0738 | +0.0029 [+0.0000, +0.0077] | -0.0018 [-0.0042, +0.0000] | 500 | 52 writers |
| UNIPEN-de zero-shot (827) | 0.7680 | 0.7914 | -0.0234 [-0.0402, -0.0119] | 0.3544 | 0.2598 | +0.0947 [+0.0322, +0.1892] | -0.0476 [-0.1152, -0.0110] | 827 | 7 writers |
| OnHW German zero-shot (5,925) | 0.7761 | 0.7557 | +0.0204 [+0.0195, +0.0212] | 0.1869 | 0.1798 | +0.0071 [+0.0031, +0.0113] | +0.0215 [+0.0156, +0.0271] | 5925 | 2 writers (unit-cluster CI shown: 2 writers) |

## 3. Order factor — routed (predicted) minus joint M5, paired per unit, native

| set | tau routed | tau joint | Δ tau [95% CI] (n def.) | tau_cp routed | tau_cp joint | Δ tau_cp | Δ direction | Δ strict order | clusters |
|---|---|---|---|---|---|---|---|---|---|
| OUHD-L Urdu test (4,406) | 0.8913 | 0.8616 | +0.0218 [+0.0121, +0.0317] (2587) | 0.8592 | 0.8507 | +0.0084 [-0.0043, +0.0214] | +0.0047 [+0.0013, +0.0079] | -0.0160 [-0.0241, -0.0084] | 32 writers |
| IAM English (500) | 0.9434 | 0.8965 | +0.0385 [+0.0149, +0.0624] (216) | 0.8938 | 0.8690 | +0.0248 [-0.0002, +0.0519] | +0.0056 [-0.0042, +0.0152] | +0.0133 [-0.0013, +0.0284] | 62 writers |
| CASIA zh (500) | 0.9404 | 0.9653 | -0.0299 [-0.0425, -0.0177] (458) | 0.8995 | 0.9513 | -0.0518 [-0.0726, -0.0309] | -0.0216 [-0.0337, -0.0095] | -0.3127 [-0.3537, -0.2721] | 60 writers |
| CASIA en (500) | 0.7376 | 0.9056 | -0.1645 [-0.2762, -0.0529] (152) | 0.6126 | 0.8859 | -0.2732 [-0.3715, -0.1729] | -0.0576 [-0.0752, -0.0415] | -0.1520 [-0.1844, -0.1200] | 57 writers |
| HPL Tamil (500) | 0.9809 | 0.9899 | -0.0090 [-0.0280, +0.0000] (221) | 0.9508 | 0.9596 | -0.0088 [-0.0272, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0020 [-0.0063, +0.0000] | 51 writers |
| UNIPEN-de zero-shot (827) | 0.7456 | 0.7442 | -0.0015 [-0.3072, +0.1008] (146) | 0.5335 | 0.5656 | -0.0321 [-0.2460, +0.0354] | -0.0417 [-0.0950, -0.0075] | +0.0085 [-0.0330, +0.0256] | 5 writers |
| OnHW German zero-shot (5,925) | 0.8539 | 0.8993 | -0.0493 [-0.0639, -0.0353] (3117) | 0.6596 | 0.7306 | -0.0710 [-0.0875, -0.0546] | -0.0031 [-0.0061, -0.0001] | +0.0104 [+0.0043, +0.0160] | 2 writers (unit-cluster CI) |

## 4. Cost of mis-routing — routed (predicted) minus routed (oracle), native

| set | router acc. vs oracle | n mis-routed | Δ AIoU [95% CI] | Δ tau [95% CI] | Δ strict order | mis-routed units only: AIoU pred / oracle |
|---|---:|---:|---|---|---|---|
| OUHD-L Urdu test (4,406) | 0.9961 | 17 | +0.0001 [-0.0000, +0.0003] | -0.0003 [-0.0008, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.757 / 0.723 (n=17) |
| IAM English (500) | 0.9940 | 3 | +0.0001 [+0.0000, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.812 / 0.788 (n=3) |
| CASIA zh (500) | 0.9800 | 10 | +0.0035 [+0.0013, +0.0061] | -0.0014 [-0.0048, +0.0000] | +0.0073 [+0.0014, +0.0149] | 0.700 / 0.527 (n=10) |
| CASIA en (500) | 0.9920 | 4 | -0.0004 [-0.0020, +0.0009] | -0.0015 [-0.0048, +0.0000] | -0.0013 [-0.0063, +0.0031] | 0.569 / 0.624 (n=4) |
| HPL Tamil (500) | 0.9940 | 3 | -0.0005 [-0.0014, +0.0001] | -0.0090 [-0.0280, +0.0000] | -0.0020 [-0.0063, +0.0000] | 0.698 / 0.786 (n=3) |
| UNIPEN-de zero-shot (827) | 0.2660 | 607 | -0.0234 [-0.0402, -0.0119] | -0.0015 [-0.3072, +0.1008] | +0.0085 [-0.0330, +0.0256] | 0.755 / 0.787 (n=607) |
| OnHW German zero-shot (5,925) | 0.0010 | 5919 | +0.0204 [+0.0195, +0.0212] | -0.0493 [-0.0639, -0.0353] | +0.0104 [+0.0043, +0.0160] | 0.776 / 0.756 (n=5919) |

German has no specialist: the oracle route is M5 joint by convention, so 'router accuracy vs oracle' on German is the share routed to M5 joint (UNIPEN-de 0.266, OnHW 0.001); the router sent 59.0% of UNIPEN-de and 99.8% of OnHW units to X5 (IAM English). On OnHW this X5 routing gives AIoU +0.0204 and tau −0.0493 relative to joint; on UNIPEN-de the mixed routing (X5 59%, M5 27%, E0 13%, E27 2%) gives AIoU −0.0234.

## 5. Routing shares (predicted route, from the router manifests)

| set | n | E0 ms8 | E27 ms47 | X5 ms9 | M5 joint | script acc. | RTL/LTR family acc. | manifest = sampled route |
|---|---:|---:|---:|---:|---:|---|---|---|
| OUHD-L Urdu test (4,406) | 4406 | 4389 | 0 | 10 | 7 | 0.996 | 0.996 | 100% |
| IAM English (500) | 500 | 0 | 0 | 497 | 3 | 0.964 | 1.000 | 100% |
| CASIA zh (500) | 500 | 0 | 490 | 10 | 0 | 0.980 | 1.000 | 100% |
| CASIA en (500) | 500 | 0 | 3 | 496 | 1 | 0.960 | 1.000 | 100% |
| HPL Tamil (500) | 500 | 1 | 0 | 2 | 497 | 0.994 | 0.998 | 100% |
| UNIPEN-de zero-shot (827) | 827 | 106 | 13 | 488 | 220 | — | — | 100% |
| OnHW German zero-shot (5,925) | 5925 | 3 | 5 | 5911 | 6 | — | — | 100% |

## 6. German zero-shot rows vs RB4 (22-23 Oct, CLUSTER-A A30 MIG, DDIM-50 seed 0)

| set | row | Stage A seed 0 / 1 / 2 (4090) | Stage A seed mean | seed SD | RB4 seed 0 | Δ (Stage A seed 0 − RB4) |
|---|---|---|---|---|---|---|
| UNIPEN-de zero-shot (827) | joint M5 AIoU | 0.7908 / 0.7910 / 0.7924 | 0.7914 | 0.0008 | 0.7921 | -0.0013 |
| OnHW German zero-shot (5,925) | joint M5 AIoU | 0.7554 / 0.7559 / 0.7557 | 0.7557 | 0.0002 | 0.7554 | +0.0000 |
| OnHW German zero-shot (5,925) | X5 AIoU on the X5-routed units (n=5911) | seed 0: 0.7765 | — | — | 0.7765 (RB4 X5 on the same units) | -0.0000 |
| UNIPEN-de zero-shot (827) | X5 AIoU on the X5-routed units (n=488) | seed 0: 0.7818 | — | — | 0.7813 (RB4 X5 on the same units) | +0.0005 |

RB4 set-level values (UNIPEN-de joint 0.7921, X5 0.7768; OnHW joint 0.7554, X5 0.7765) are reproduced on the 4090 within 0.0013 AIoU; the largest set-level gap (UNIPEN-de joint, −0.0013) is 1.6× the across-seed SD (0.0008) on that set, the others are within 1 SD. At the unit level the same model and seed on different GPUs give different samples: identical per-unit AIoU for only 2.7% (UNIPEN-de) / 0.7% (OnHW) of units, mean |Δ AIoU| per unit 0.025 / 0.025 (file `device_check_vs_RB4_D3A.csv`). Stage A E0 ms8 on OUHD-L test: 0.7980 (4090) vs 0.7987 (CLUSTER-A, table builder) and M5 joint 0.8207 vs 0.8204.

## 7. Standardised protocol (1 px resampling of both inks), seed mean — AIoU and tau

| set | AIoU routed (pred) | AIoU oracle | AIoU joint | Δ AIoU routed−joint [CI] | tau routed | tau joint | Δ tau routed−joint [CI] |
|---|---|---|---|---|---|---|---|
| OUHD-L Urdu test (4,406) | 0.8189 | 0.8188 | 0.8420 | -0.0231 [-0.0256, -0.0209] | 0.8915 | 0.8628 | +0.0206 [+0.0113, +0.0304] |
| IAM English (500) | 0.8625 | 0.8624 | 0.8644 | -0.0018 [-0.0046, +0.0008] | 0.9345 | 0.8974 | +0.0282 [+0.0016, +0.0543] |
| CASIA zh (500) | 0.5918 | 0.5882 | 0.7812 | -0.1893 [-0.1974, -0.1806] | 0.9380 | 0.9657 | -0.0334 [-0.0503, -0.0190] |
| CASIA en (500) | 0.8375 | 0.8380 | 0.8812 | -0.0437 [-0.0508, -0.0366] | 0.7280 | 0.9056 | -0.1753 [-0.2822, -0.0679] |
| HPL Tamil (500) | 0.8625 | 0.8632 | 0.8632 | -0.0006 [-0.0015, +0.0000] | 0.9809 | 0.9899 | -0.0090 [-0.0280, +0.0000] |
| UNIPEN-de zero-shot (827) | 0.8210 | 0.8473 | 0.8473 | -0.0263 [-0.0462, -0.0127] | 0.7539 | 0.7491 | +0.0122 [-0.2559, +0.1009] |
| OnHW German zero-shot (5,925) | 0.8049 | 0.7843 | 0.7843 | +0.0206 [+0.0198, +0.0216] | 0.8535 | 0.8992 | -0.0485 [-0.0634, -0.0347] |

## 8. Confounds and caveats (stated, not resolved)

1. **CASIA zh resampling confound.** 46.2% of CASIA-zh units exceed L=320 points at 1 px (median 312 points; seqlen_check_M5.csv), so the E0-format 320-point sequence is coarser than 1 px for almost half the zh units for every model scored here; zh AIoU/LDTW are not comparable with the E27 X2 numbers (0.814 AIoU on E27's own pipeline, 14,400 units per split).
2. **E27 input pipeline mismatch.** E27 ms47 was trained on the Nagamatsu CASIA pipeline (casia_grid rendering); in Stage A it received the X4a/E0 64x64 `cond_rendered` images and its output was decoded with the E0 adapter. Its zh row (AIoU 0.5477, strict order 0.0073) is therefore E27 under a render/pipeline shift, and the zh 'specialist' route inherits that shift. M5 joint was trained on E0-pipeline zh units.
3. **CASIA-en 'specialist' is X5 (IAM lines/words model)** applied to isolated Latin letters; the en row is X5 under a unit-type shift.
4. **German oracle convention**: oracle = M5 joint (no German specialist), so routed-oracle = joint on German and the 'cost of mis-routing' on German is the effect of the router's X5/E0/E27 choices, not an error against a true specialist.
5. **OnHW has 2 writers**: writer-cluster CIs are degenerate there; unit-cluster CIs are shown for OnHW in sections 2-4 (both are in paired_D3A.csv).
6. **Single sampling device** (one RTX 4090, shared GPU); device effect at the unit level is documented in section 6. Single training seed per model.
7. **Seed noise**: the across-seed SD of set-mean AIoU is ≤ 0.0025 on every row (tau SD up to 0.04 on CASIA-en / UNIPEN-de, where few units have a defined tau: 152 and 146 pairs).
8. **Evaluator version of earlier rows.** `e0/code_4cd40f57` carries a vendored uhwr_eval 0.1.0. `e0_score.py` (RB4 rows) and `rescore_031.py` (table-builder 'rescored under 0.3.1' rows) put that directory first on sys.path, and their row files have no `penup_f1_v2` column, which 0.3.1 `score()` always emits; those rows were therefore produced by the vendored 0.1.0 (the 0.1.0-vs-0.3.1 identity gate compared 0.1.0 with itself). Per CHANGELOG 0.3.1 'returns every 0.2.0 key with 0.2.0 values', so the shared metrics should be unchanged, but that is asserted, not measured. Stage A rows import 0.3.1 from `shared/uhwr_eval-0.3.1` (path recorded per row, `penup_f1_v2` present). Measured (Slurm 24619470, `stageA/verify_031/`): the RB4 joint-ms9 UNIPEN-de preds re-scored under a verified 0.3.1 import give exactly the RB4 row values on all 16 shared metrics (1,654 rows = 827 units x 2 protocols, 0 differences, file `verify_031_vs_RB4_jointms9_unipen_de.csv`); the only consequence of the shadowing is the missing `penup_f1_v2` column in those rows.
9. The gold set (354) and the val splits were not part of Stage A sampling.

## 9. Files

- `results_D3A_crossscript.csv` — one row per (set, method, protocol): means, writer-cluster CIs, per-seed SD, n, n_writers, files + sha256.
- `results_D3A_per_seed.csv` — set means per seed.
- `paired_D3A.csv` — per-unit paired deltas (routed-pred − joint, routed-oracle − joint, routed-pred − routed-oracle, specialist − joint) with writer- and unit-cluster CIs, both protocols, 8 metrics.
- `routing_accuracy_D3A.csv`, `misrouted_units_D3A.csv`, `dedup_identity_D3A.csv`, `device_check_vs_RB4_D3A.csv`, `verify_031_vs_RB4_jointms9_unipen_de.csv`.
- `PROVENANCE_D3A.json` — preds sha256 (75), router sha, manifests sha, GT npz sha, checkpoint sha, evaluator, container, job ids, code sha.
- Per-unit rows (272,280 rows): CLUSTER-A `exp/D3_ROUTE/stageA/score_out/rows/` (tarball `d3a_rows_all.tar.gz` sha256 ab72b1b7…).