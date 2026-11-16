# RESULTS R1c (Urdu zero-shot, identity vs mirror_x; descriptive + K1 restated)

Generated 2026-11-12T19:49:44Z; OUHD-L test (4,406 units; TRACE: test253 lines), protocol as R1a. K1 per base: tau CI lo of the mirror_x arm >= 0.65 (restated, not a new claim for LUO).

## LUO_ms6 (test_ouhdl)

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| identity absolute (n=4406, writer-cluster) |
| 0.7117 [0.7070, 0.7164] | 0.5468 [0.5295, 0.5646] | 0.4799 [0.4426, 0.5177] | 0.4693 [0.4528, 0.4859] | 0.3581 [0.3398, 0.3785] |
| mirror_x absolute (n=4406, writer-cluster) |
| 0.7090 [0.7032, 0.7144] | 0.8687 [0.8589, 0.8785] | 0.6586 [0.6204, 0.6931] | 0.5232 [0.4997, 0.5470] | 0.3904 [0.3657, 0.4169] |
| **mirror_x - identity** (n=4406) |
| -0.0027 [-0.0062, +0.0007] | +0.3220 [+0.3012, +0.3417] | +0.1840 [+0.1237, +0.2404] | +0.0539 [+0.0352, +0.0725] | +0.0322 [+0.0181, +0.0479] |

K1 restated (mirror_x tau CI lo >= 0.65): False

## NOFM_W_bb3 (test_ouhdl)

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| identity absolute (n=4406, writer-cluster) |
| 0.8002 [0.7970, 0.8035] | 0.5767 [0.5584, 0.5947] | 0.3057 [0.2613, 0.3455] | 0.5150 [0.4969, 0.5339] | 0.7056 [0.6914, 0.7189] |
| mirror_x absolute (n=4406, writer-cluster) |
| 0.7954 [0.7914, 0.7994] | 0.8457 [0.8325, 0.8584] | 0.4653 [0.4194, 0.5125] | 0.5673 [0.5453, 0.5908] | 0.7313 [0.7174, 0.7454] |
| **mirror_x - identity** (n=4406) |
| -0.0048 [-0.0068, -0.0028] | +0.2682 [+0.2441, +0.2922] | +0.1778 [+0.1161, +0.2388] | +0.0524 [+0.0356, +0.0685] | +0.0257 [+0.0169, +0.0343] |

K1 restated (mirror_x tau CI lo >= 0.65): False

## PEINIT_W_bb4 (test_ouhdl)

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| identity NOT AVAILABLE |
| mirror_x NOT AVAILABLE |
| paired NOT AVAILABLE |

K1 restated (mirror_x tau CI lo >= 0.65): None

## TRACE_uhwr_f4 (test253)

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| identity absolute (n=253, writer-cluster) |
| 0.4914 [0.4854, 0.4978] | 0.4727 [0.4487, 0.4970] | -0.9881 [-0.9912, -0.9848] | 0.0000 [0.0000, 0.0000] | 0.1148 [0.1037, 0.1252] |
| mirror_x absolute (n=253, writer-cluster) |
| 0.4682 [0.4581, 0.4787] | 0.9029 [0.8879, 0.9179] | 0.9904 [0.9877, 0.9930] | 0.0000 [0.0000, 0.0000] | 0.1626 [0.1424, 0.1820] |
| **mirror_x - identity** (n=253) |
| -0.0233 [-0.0364, -0.0100] | +0.4302 [+0.4036, +0.4556] | +1.9785 [+1.9736, +1.9830] | +0.0000 [+0.0000, +0.0000] | +0.0478 [+0.0326, +0.0631] |

K1 restated (mirror_x tau CI lo >= 0.65): True

## TRACE_trace_render (test253)

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| identity absolute (n=253, writer-cluster) |
| 0.5172 [0.5108, 0.5239] | 0.4759 [0.4555, 0.4972] | -0.9865 [-0.9889, -0.9837] | 0.0000 [0.0000, 0.0000] | 0.1406 [0.1291, 0.1514] |
| mirror_x absolute (n=253, writer-cluster) |
| 0.4651 [0.4568, 0.4734] | 0.9183 [0.9059, 0.9306] | 0.9922 [0.9895, 0.9945] | 0.0000 [0.0000, 0.0000] | 0.1679 [0.1502, 0.1855] |
| **mirror_x - identity** (n=253) |
| -0.0522 [-0.0622, -0.0425] | +0.4424 [+0.4193, +0.4634] | +1.9786 [+1.9744, +1.9825] | +0.0000 [+0.0000, +0.0000] | +0.0273 [+0.0123, +0.0431] |

K1 restated (mirror_x tau CI lo >= 0.65): True

TRACE is deterministic (no sampling seed); n_seeds = 1.
