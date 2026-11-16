# M5 / X6 results: one model for many scripts (facts only)

Run dir CLUSTER-A $U/exp/M5 (runs/joint, runs/luo, runs/ft/p01..p100). Slurm: joint 24517794, leave-Urdu-out (LUO) 24517795, FT 24517796 (afterok:24517795); all b200idle, <node>, NVIDIA B200, COMPLETED 0:0, 0 restarts. Prep 24516200, smoke 24516245.

## Protocol
- Training (both arms): Nagamatsu model (repo 7c5cf623), E0 recipe: 100k steps, batch 128 x grad-accum 4 = 512 per step, lr 8e-5, ckpt every 10k, seed 20261016, 2x B200 (2-proc accelerate, NCCL_P2P_DISABLE=1). Joint = 100,000 units (20,000 each CASIA zh, CASIA en, HPL Tamil, IAM components, OUHD-L v3 ligatures); LUO = 80,000 (same minus OUHD-L). Data: data/m5_multiscript_v1.
- Checkpoint selection (E0 rule): lowest mean native LDTW, ties -> higher AIoU, DDIM-50 eta 0 seed 0. Joint: 1,000 val units (200 per script incl. OUHD-L val) -> ms9 (step 90k). LUO: 1,000 non-Urdu val units (250 x casia_zh, casia_en, hpl_tamil, iam; no Urdu data in training or selection) -> ms6 (step 60k).
- Fine-tunes: init = LUO ms6 (model + EMA weights; fresh Adam, step 0); 10k steps each, same recipe on 1 GPU (batch 128 x accum 4 = 512), ckpt every 1k; subsets of OUHD-L E0 train, nested, seed [20261016, 9]: p01 336 units, p10 3,362, p30 10,085, p100 33,615. Selection on E0 val1000 (e0_val.npz, subset seed 0): p01 ms4 (FT step 4k), p10 ms10, p30 ms10, p100 ms9.
- Evaluation: DDIM-50 eta 0 seed 0, EMA weights, e0_sample.py / e0_score.py (code 4cd40f57, unchanged), uhwr_eval native + standardised protocols, LPIPS off, 64x64 evaluation canvas. OUHD-L v3 test 4,406 units (253 lines, 32 writers) and gold 354 units (21 lines, 16 writers), input rendered (E0 training renderer, 2 px AA) and dataset (OUHD-L img/ crop, f=4 -> INTER_AREA 64x64). Non-Urdu (rendered input only): X4a sets casia_zh500 / casia_en500 (OLHWDB1.1 test, 60 writers), hpl_tamil500 (IWFHR06 test, 52 writers), iam500 (IAM testset_f, 63 writers, seed 2026).
- CIs: 95% writer-cluster bootstrap, B=2000, seed 0 (uhwr_baselines.aggregate.boot_ci, clusters = writer_id). Paired differences: per-unit M5 minus E0 on the same units, units where both values are defined, same bootstrap. kendall_tau and direction_acc are defined on a subset of units (n_valid in results_M5.csv).
- Comparison rows: E0 ms8 (Urdu-only, step 80k) final-eval rows for OUHD-L test/gold; E0 ms8 X4a zero-shot rows for casia_zh500 / casia_en500 / hpl_tamil500. No E0 row exists for iam500. E27 (CASIA-only) comparison: not run (no E27 checkpoint).
- Ligature-level results are given on all v3 test units and on the gold-line units separately (not pooled). v3 has an estimated wrong-unit rate of about 15-17 % (data card).
- Metrics (native protocol rows below): AIoU, LDTW (length-normalised DTW), DTW (px), Kendall tau (stroke order), direction_acc (per-stroke direction), strict_order_acc. Standardised-protocol rows and the other metrics (penup_f1, stroke counts, chamfer, hausdorff, frechet, rmse, loop orientation) are in results_M5.csv.

### OUHD-L v3 test (4,406 units, 32 writers), native protocol

| input | label | aiou | ldtw | dtw | kendall_tau | direction_acc | strict_order_acc | n_units | n_writers |
|---|---|---|---|---|---|---|---|---|---|
| dataset | e0_ms8 | 0.5667 [0.5592, 0.5741] | 0.5159 [0.4936, 0.5394] | 169.6405 [163.1162, 176.3222] | 0.7900 [0.7499, 0.8263] | 0.9039 [0.8923, 0.9143] | 0.3000 [0.2816, 0.3190] | 4406 | 32 |
| dataset | m5_joint_ms9 | 0.6291 [0.6219, 0.6363] | 0.8836 [0.8482, 0.9209] | 299.4954 [289.1271, 310.6767] | -0.0249 [-0.0622, 0.0114] | 0.6719 [0.6501, 0.6942] | 0.2197 [0.1995, 0.2421] | 4406 | 32 |
| dataset | m5_luo_ms6 | 0.5842 [0.5737, 0.5945] | 0.9476 [0.9054, 0.9900] | 317.3807 [306.6076, 328.5634] | -0.0068 [-0.0754, 0.0543] | 0.5959 [0.5718, 0.6206] | 0.1757 [0.1572, 0.1969] | 4406 | 32 |
| dataset | m5_ft_p01_ms4 | 0.5676 [0.5570, 0.5775] | 0.5139 [0.4845, 0.5448] | 159.6336 [152.0446, 167.8355] | 0.7561 [0.7082, 0.8020] | 0.9109 [0.8950, 0.9242] | 0.2962 [0.2720, 0.3194] | 4406 | 32 |
| dataset | m5_ft_p10_ms10 | 0.5543 [0.5411, 0.5661] | 0.5336 [0.5007, 0.5693] | 165.7324 [158.2227, 173.4699] | 0.7971 [0.7513, 0.8396] | 0.8992 [0.8856, 0.9116] | 0.2351 [0.2114, 0.2590] | 4406 | 32 |
| dataset | m5_ft_p30_ms10 | 0.5685 [0.5554, 0.5803] | 0.5320 [0.5053, 0.5611] | 170.6591 [163.4016, 178.5792] | 0.7817 [0.7355, 0.8221] | 0.8982 [0.8824, 0.9119] | 0.2322 [0.2071, 0.2578] | 4406 | 32 |
| dataset | m5_ft_p100_ms9 | 0.5639 [0.5506, 0.5762] | 0.5818 [0.5508, 0.6156] | 186.0754 [178.7014, 193.8624] | 0.7025 [0.6570, 0.7438] | 0.8577 [0.8430, 0.8707] | 0.2029 [0.1795, 0.2269] | 4406 | 32 |
| rendered | e0_ms8 | 0.7987 [0.7951, 0.8022] | 0.1320 [0.1217, 0.1434] | 49.1827 [45.3924, 53.1276] | 0.9018 [0.8797, 0.9187] | 0.9497 [0.9399, 0.9584] | 0.7458 [0.7241, 0.7681] | 4406 | 32 |
| rendered | m5_joint_ms9 | 0.8204 [0.8173, 0.8234] | 0.1321 [0.1191, 0.1466] | 47.7542 [44.1583, 51.7725] | 0.8695 [0.8471, 0.8874] | 0.9474 [0.9368, 0.9563] | 0.7644 [0.7434, 0.7856] | 4406 | 32 |
| rendered | m5_luo_ms6 | 0.7108 [0.7065, 0.7151] | 0.8625 [0.8276, 0.9000] | 305.6868 [296.6477, 315.1091] | 0.4832 [0.4394, 0.5284] | 0.5450 [0.5258, 0.5650] | 0.4673 [0.4499, 0.4851] | 4406 | 32 |
| rendered | m5_ft_p01_ms4 | 0.7522 [0.7484, 0.7562] | 0.2098 [0.1935, 0.2290] | 75.0069 [70.2255, 80.7249] | 0.8542 [0.8200, 0.8796] | 0.9370 [0.9254, 0.9471] | 0.6509 [0.6288, 0.6725] | 4406 | 32 |
| rendered | m5_ft_p10_ms10 | 0.7842 [0.7808, 0.7877] | 0.1494 [0.1357, 0.1653] | 55.1382 [50.8623, 60.0732] | 0.8963 [0.8720, 0.9175] | 0.9477 [0.9385, 0.9559] | 0.7217 [0.7014, 0.7418] | 4406 | 32 |
| rendered | m5_ft_p30_ms10 | 0.8015 [0.7983, 0.8048] | 0.1375 [0.1231, 0.1542] | 49.8829 [46.3091, 53.5814] | 0.8830 [0.8624, 0.9003] | 0.9499 [0.9392, 0.9586] | 0.7515 [0.7306, 0.7729] | 4406 | 32 |
| rendered | m5_ft_p100_ms9 | 0.8075 [0.8046, 0.8105] | 0.1351 [0.1210, 0.1511] | 48.2667 [44.3016, 52.5693] | 0.8732 [0.8530, 0.8911] | 0.9500 [0.9397, 0.9590] | 0.7628 [0.7408, 0.7854] | 4406 | 32 |

### OUHD-L gold units (354 units, 16 writers), native protocol

| input | label | aiou | ldtw | dtw | kendall_tau | direction_acc | strict_order_acc | n_units | n_writers |
|---|---|---|---|---|---|---|---|---|---|
| dataset | e0_ms8 | 0.5514 [0.5270, 0.5750] | 0.5947 [0.5147, 0.6923] | 196.9746 [176.1253, 216.1035] | 0.7804 [0.6286, 0.9004] | 0.8979 [0.8629, 0.9283] | 0.2655 [0.2122, 0.3221] | 354 | 16 |
| dataset | m5_joint_ms9 | 0.6057 [0.5790, 0.6302] | 0.7945 [0.7367, 0.8624] | 284.6178 [263.7618, 302.1638] | -0.0100 [-0.1481, 0.1386] | 0.6777 [0.6177, 0.7386] | 0.2119 [0.1586, 0.2702] | 354 | 16 |
| dataset | m5_luo_ms6 | 0.5592 [0.5272, 0.5877] | 0.8460 [0.7707, 0.9328] | 302.0212 [278.3594, 321.7507] | -0.1714 [-0.4098, 0.0769] | 0.6055 [0.5340, 0.6838] | 0.1667 [0.1268, 0.2165] | 354 | 16 |
| dataset | m5_ft_p01_ms4 | 0.5505 [0.5258, 0.5719] | 0.5714 [0.5033, 0.6374] | 175.9126 [155.8378, 194.2900] | 0.7009 [0.4779, 0.8605] | 0.9013 [0.8761, 0.9265] | 0.2684 [0.2231, 0.3162] | 354 | 16 |
| dataset | m5_ft_p10_ms10 | 0.5255 [0.4943, 0.5533] | 0.5850 [0.5061, 0.6707] | 184.9235 [166.6515, 201.2129] | 0.8021 [0.6666, 0.9250] | 0.8644 [0.8210, 0.9058] | 0.2175 [0.1778, 0.2582] | 354 | 16 |
| dataset | m5_ft_p30_ms10 | 0.5434 [0.5118, 0.5703] | 0.5900 [0.5174, 0.6725] | 197.1574 [174.9974, 216.5655] | 0.8706 [0.7471, 0.9728] | 0.8726 [0.8333, 0.9071] | 0.2260 [0.1734, 0.2848] | 354 | 16 |
| dataset | m5_ft_p100_ms9 | 0.5340 [0.4962, 0.5676] | 0.6573 [0.5861, 0.7404] | 217.5312 [197.6084, 235.0099] | 0.7639 [0.5895, 0.9565] | 0.8394 [0.7876, 0.8834] | 0.1864 [0.1448, 0.2328] | 354 | 16 |
| rendered | e0_ms8 | 0.8053 [0.7964, 0.8141] | 0.0908 [0.0703, 0.1134] | 35.8317 [28.2399, 44.5546] | 0.9079 [0.8515, 0.9555] | 0.9493 [0.9301, 0.9668] | 0.7938 [0.7464, 0.8394] | 354 | 16 |
| rendered | m5_joint_ms9 | 0.8215 [0.8150, 0.8278] | 0.1032 [0.0809, 0.1268] | 39.8609 [32.6061, 48.2828] | 0.9071 [0.8611, 0.9463] | 0.9516 [0.9345, 0.9686] | 0.7938 [0.7479, 0.8366] | 354 | 16 |
| rendered | m5_luo_ms6 | 0.7076 [0.6966, 0.7205] | 0.7147 [0.6535, 0.7833] | 266.1091 [249.0245, 286.8666] | 0.5735 [0.4535, 0.7006] | 0.5996 [0.5488, 0.6561] | 0.4944 [0.4248, 0.5545] | 354 | 16 |
| rendered | m5_ft_p01_ms4 | 0.7581 [0.7459, 0.7690] | 0.1760 [0.1350, 0.2289] | 63.9670 [51.9432, 74.8655] | 0.8589 [0.7670, 0.9343] | 0.9400 [0.9172, 0.9605] | 0.6836 [0.6419, 0.7304] | 354 | 16 |
| rendered | m5_ft_p10_ms10 | 0.7896 [0.7824, 0.7963] | 0.1244 [0.0996, 0.1509] | 46.4492 [38.3715, 55.2939] | 0.8814 [0.7961, 0.9474] | 0.9372 [0.9196, 0.9538] | 0.7232 [0.6806, 0.7639] | 354 | 16 |
| rendered | m5_ft_p30_ms10 | 0.8020 [0.7927, 0.8100] | 0.1071 [0.0772, 0.1411] | 36.7786 [30.2281, 44.1141] | 0.9256 [0.8679, 0.9697] | 0.9393 [0.9216, 0.9576] | 0.7712 [0.7256, 0.8140] | 354 | 16 |
| rendered | m5_ft_p100_ms9 | 0.8092 [0.8002, 0.8172] | 0.1036 [0.0777, 0.1345] | 38.0352 [30.3399, 46.9919] | 0.8858 [0.8263, 0.9362] | 0.9421 [0.9178, 0.9640] | 0.7655 [0.7065, 0.8145] | 354 | 16 |

### Non-Urdu test sets, rendered input, native protocol

| eval_set | label | aiou | ldtw | dtw | kendall_tau | direction_acc | strict_order_acc | n_units | n_writers |
|---|---|---|---|---|---|---|---|---|---|
| casia_en500 | e0_ms8_x4a | 0.6086 [0.5954, 0.6212] | 1.3216 [1.2561, 1.3853] | 183.6442 [175.1737, 192.4520] | -0.4138 [-0.6513, -0.1935] | 0.5993 [0.5514, 0.6469] | 0.4060 [0.3661, 0.4484] | 500 | 60 |
| casia_en500 | m5_joint_ms9 | 0.7991 [0.7961, 0.8018] | 0.4986 [0.4736, 0.5251] | 74.4985 [70.6436, 78.5461] | 0.9280 [0.8683, 0.9749] | 0.9814 [0.9712, 0.9908] | 0.9420 [0.9167, 0.9649] | 500 | 60 |
| casia_en500 | m5_luo_ms6 | 0.7916 [0.7884, 0.7946] | 0.5043 [0.4793, 0.5307] | 75.5418 [71.6245, 79.5601] | 0.8826 [0.7948, 0.9495] | 0.9794 [0.9698, 0.9888] | 0.9360 [0.9122, 0.9568] | 500 | 60 |
| casia_zh500 | e0_ms8_x4a | 0.4464 [0.4382, 0.4540] | 1.0251 [0.9968, 1.0544] | 265.1415 [259.6559, 270.5978] | -0.6889 [-0.8116, -0.5691] | 0.6231 [0.5673, 0.6755] | 0.0080 [0.0019, 0.0162] | 500 | 60 |
| casia_zh500 | m5_joint_ms9 | 0.7312 [0.7251, 0.7368] | 0.2704 [0.2609, 0.2808] | 75.9472 [73.3159, 79.0995] | 0.9601 [0.9428, 0.9744] | 0.9635 [0.9529, 0.9728] | 0.3420 [0.2985, 0.3875] | 500 | 60 |
| casia_zh500 | m5_luo_ms6 | 0.7103 [0.7049, 0.7157] | 0.2758 [0.2651, 0.2877] | 77.4571 [74.4379, 80.7306] | 0.9618 [0.9465, 0.9740] | 0.9611 [0.9474, 0.9720] | 0.3040 [0.2622, 0.3472] | 500 | 60 |
| hpl_tamil500 | e0_ms8_x4a | 0.5308 [0.5186, 0.5447] | 1.2294 [1.1600, 1.2940] | 288.5671 [276.5039, 301.0862] | 0.3721 [0.1860, 0.5467] | 0.5320 [0.4739, 0.5848] | 0.3580 [0.3223, 0.3954] | 500 | 52 |
| hpl_tamil500 | m5_joint_ms9 | 0.8142 [0.8109, 0.8174] | 0.0730 [0.0625, 0.0852] | 15.8542 [13.7097, 18.5823] | 0.9970 [0.9901, 1.0000] | 0.9716 [0.9613, 0.9814] | 0.9440 [0.9185, 0.9665] | 500 | 52 |
| hpl_tamil500 | m5_luo_ms6 | 0.7964 [0.7926, 0.8002] | 0.0831 [0.0665, 0.1060] | 17.9376 [15.0159, 21.8097] | 0.9844 [0.9474, 1.0000] | 0.9733 [0.9627, 0.9844] | 0.9280 [0.9018, 0.9503] | 500 | 52 |
| iam500 | m5_joint_ms9 | 0.7934 [0.7899, 0.7966] | 0.2331 [0.2051, 0.2631] | 32.0771 [28.3269, 36.3383] | 0.9147 [0.8581, 0.9611] | 0.9500 [0.9317, 0.9671] | 0.8800 [0.8426, 0.9153] | 500 | 63 |
| iam500 | m5_luo_ms6 | 0.7821 [0.7782, 0.7861] | 0.2283 [0.2035, 0.2554] | 31.2960 [27.9200, 34.9000] | 0.9258 [0.8788, 0.9674] | 0.9554 [0.9399, 0.9701] | 0.8760 [0.8400, 0.9074] | 500 | 63 |

### Paired differences M5 minus E0 ms8 (same units), native protocol

| eval_set | input | method | aiou | ldtw | kendall_tau | direction_acc |
|---|---|---|---|---|---|---|
| casia_en500 | rendered | m5_joint_ms9_ddim50 | +0.1905 [+0.1781, +0.2032] (n=500) | -0.8231 [-0.8849, -0.7610] (n=500) | +1.2759 [+1.0526, +1.5201] (n=58) | +0.3807 [+0.3325, +0.4300] (n=406) |
| casia_en500 | rendered | m5_luo_ms6_ddim50 | +0.1830 [+0.1698, +0.1962] (n=500) | -0.8173 [-0.8769, -0.7582] (n=500) | +1.2515 [+1.0357, +1.4966] (n=57) | +0.3787 [+0.3321, +0.4278] (n=406) |
| casia_zh500 | rendered | m5_joint_ms9_ddim50 | +0.2849 [+0.2758, +0.2944] (n=500) | -0.7547 [-0.7867, -0.7247] (n=500) | +1.6658 [+1.5483, +1.7890] (n=90) | +0.3371 [+0.2848, +0.3925] (n=280) |
| casia_zh500 | rendered | m5_luo_ms6_ddim50 | +0.2640 [+0.2548, +0.2734] (n=500) | -0.7493 [-0.7812, -0.7185] (n=500) | +1.6592 [+1.5310, +1.7899] (n=90) | +0.3339 [+0.2746, +0.3949] (n=280) |
| hpl_tamil500 | rendered | m5_joint_ms9_ddim50 | +0.2834 [+0.2684, +0.2964] (n=500) | -1.1564 [-1.2205, -1.0881] (n=500) | +0.6118 [+0.4502, +0.7871] (n=85) | +0.4342 [+0.3784, +0.4949] (n=375) |
| hpl_tamil500 | rendered | m5_luo_ms6_ddim50 | +0.2657 [+0.2503, +0.2787] (n=500) | -1.1463 [-1.2087, -1.0812] (n=500) | +0.6098 [+0.4444, +0.7901] (n=82) | +0.4364 [+0.3809, +0.4972] (n=375) |
| ouhdl_gold | dataset | m5_ft_p01_ms4_ddim50 | -0.0009 [-0.0112, +0.0091] (n=354) | -0.0233 [-0.0971, +0.0376] (n=354) | -0.0864 [-0.2222, +0.0498] (n=54) | -0.0033 [-0.0315, +0.0263] (n=205) |
| ouhdl_gold | dataset | m5_ft_p100_ms9_ddim50 | -0.0174 [-0.0341, -0.0031] (n=354) | +0.0626 [+0.0168, +0.1045] (n=354) | -0.0802 [-0.2667, +0.1450] (n=54) | -0.0317 [-0.0698, +0.0147] (n=179) |
| ouhdl_gold | dataset | m5_ft_p10_ms10_ddim50 | -0.0259 [-0.0379, -0.0145] (n=354) | -0.0097 [-0.0555, +0.0335] (n=354) | -0.1200 [-0.2721, +0.0000] (n=50) | -0.0290 [-0.0700, +0.0070] (n=184) |
| ouhdl_gold | dataset | m5_ft_p30_ms10_ddim50 | -0.0080 [-0.0222, +0.0059] (n=354) | -0.0047 [-0.0522, +0.0362] (n=354) | -0.0769 [-0.1449, +0.0000] (n=52) | -0.0061 [-0.0424, +0.0289] (n=190) |
| ouhdl_gold | dataset | m5_joint_ms9_ddim50 | +0.0543 [+0.0382, +0.0718] (n=354) | +0.1998 [+0.0912, +0.2919] (n=354) | -0.8974 [-1.0389, -0.7185] (n=65) | -0.1948 [-0.2747, -0.1132] (n=219) |
| ouhdl_gold | dataset | m5_luo_ms6_ddim50 | +0.0078 [-0.0070, +0.0224] (n=354) | +0.2513 [+0.1095, +0.3702] (n=354) | -1.1600 [-1.3529, -0.9524] (n=50) | -0.2863 [-0.3934, -0.1678] (n=179) |
| ouhdl_gold | rendered | m5_ft_p01_ms4_ddim50 | -0.0473 [-0.0559, -0.0404] (n=354) | +0.0851 [+0.0432, +0.1434] (n=354) | -0.0342 [-0.1003, +0.0225] (n=185) | -0.0093 [-0.0299, +0.0129] (n=352) |
| ouhdl_gold | rendered | m5_ft_p100_ms9_ddim50 | +0.0039 [-0.0017, +0.0086] (n=354) | +0.0128 [-0.0033, +0.0315] (n=354) | -0.0021 [-0.0393, +0.0387] (n=190) | -0.0074 [-0.0239, +0.0124] (n=353) |
| ouhdl_gold | rendered | m5_ft_p10_ms10_ddim50 | -0.0157 [-0.0220, -0.0100] (n=354) | +0.0336 [+0.0181, +0.0516] (n=354) | -0.0142 [-0.0629, +0.0283] (n=188) | -0.0121 [-0.0245, +0.0022] (n=352) |
| ouhdl_gold | rendered | m5_ft_p30_ms10_ddim50 | -0.0033 [-0.0095, +0.0019] (n=354) | +0.0163 [-0.0131, +0.0511] (n=354) | +0.0253 [+0.0063, +0.0492] (n=198) | -0.0102 [-0.0295, +0.0093] (n=353) |
| ouhdl_gold | rendered | m5_joint_ms9_ddim50 | +0.0162 [+0.0113, +0.0204] (n=354) | +0.0124 [-0.0034, +0.0294] (n=354) | -0.0062 [-0.0332, +0.0225] (n=193) | +0.0009 [-0.0157, +0.0193] (n=352) |
| ouhdl_gold | rendered | m5_luo_ms6_ddim50 | -0.0978 [-0.1074, -0.0884] (n=354) | +0.6239 [+0.5691, +0.6876] (n=354) | -0.3161 [-0.4087, -0.2182] (n=155) | -0.3475 [-0.3959, -0.2978] (n=342) |
| ouhdl_test | dataset | m5_ft_p01_ms4_ddim50 | +0.0009 [-0.0036, +0.0049] (n=4406) | -0.0020 [-0.0223, +0.0187] (n=4406) | -0.0643 [-0.1144, -0.0145] (n=690) | +0.0067 [-0.0053, +0.0184] (n=2816) |
| ouhdl_test | dataset | m5_ft_p100_ms9_ddim50 | -0.0028 [-0.0104, +0.0037] (n=4406) | +0.0659 [+0.0453, +0.0860] (n=4406) | -0.0805 [-0.1283, -0.0379] (n=687) | -0.0324 [-0.0477, -0.0187] (n=2552) |
| ouhdl_test | dataset | m5_ft_p10_ms10_ddim50 | -0.0124 [-0.0198, -0.0061] (n=4406) | +0.0177 [-0.0007, +0.0377] (n=4406) | +0.0026 [-0.0458, +0.0504] (n=642) | -0.0041 [-0.0174, +0.0093] (n=2622) |
| ouhdl_test | dataset | m5_ft_p30_ms10_ddim50 | +0.0018 [-0.0051, +0.0076] (n=4406) | +0.0161 [-0.0012, +0.0329] (n=4406) | -0.0190 [-0.0635, +0.0222] (n=700) | -0.0042 [-0.0164, +0.0078] (n=2648) |
| ouhdl_test | dataset | m5_joint_ms9_ddim50 | +0.0624 [+0.0576, +0.0670] (n=4406) | +0.3677 [+0.3309, +0.4020] (n=4406) | -0.8544 [-0.9247, -0.7850] (n=789) | -0.2189 [-0.2471, -0.1899] (n=2915) |
| ouhdl_test | dataset | m5_luo_ms6_ddim50 | +0.0175 [+0.0123, +0.0227] (n=4406) | +0.4317 [+0.3861, +0.4746] (n=4406) | -0.8990 [-0.9867, -0.8094] (n=670) | -0.3041 [-0.3343, -0.2729] (n=2580) |
| ouhdl_test | rendered | m5_ft_p01_ms4_ddim50 | -0.0465 [-0.0487, -0.0443] (n=4406) | +0.0777 [+0.0660, +0.0906] (n=4406) | -0.0352 [-0.0507, -0.0203] (n=2160) | -0.0130 [-0.0193, -0.0068] (n=4353) |
| ouhdl_test | rendered | m5_ft_p100_ms9_ddim50 | +0.0088 [+0.0067, +0.0112] (n=4406) | +0.0030 [-0.0055, +0.0122] (n=4406) | -0.0216 [-0.0335, -0.0096] (n=2432) | -0.0000 [-0.0047, +0.0047] (n=4388) |
| ouhdl_test | rendered | m5_ft_p10_ms10_ddim50 | -0.0145 [-0.0165, -0.0125] (n=4406) | +0.0174 [+0.0084, +0.0264] (n=4406) | -0.0017 [-0.0136, +0.0097] (n=2314) | -0.0019 [-0.0058, +0.0020] (n=4375) |
| ouhdl_test | rendered | m5_ft_p30_ms10_ddim50 | +0.0028 [+0.0008, +0.0049] (n=4406) | +0.0055 [-0.0032, +0.0153] (n=4406) | -0.0122 [-0.0240, -0.0008] (n=2413) | +0.0001 [-0.0047, +0.0052] (n=4385) |
| ouhdl_test | rendered | m5_joint_ms9_ddim50 | +0.0217 [+0.0197, +0.0238] (n=4406) | +0.0001 [-0.0066, +0.0078] (n=4406) | -0.0217 [-0.0333, -0.0105] (n=2445) | -0.0024 [-0.0063, +0.0015] (n=4389) |
| ouhdl_test | rendered | m5_luo_ms6_ddim50 | -0.0879 [-0.0923, -0.0828] (n=4406) | +0.7305 [+0.6988, +0.7637] (n=4406) | -0.4126 [-0.4595, -0.3663] (n=1905) | -0.4047 [-0.4256, -0.3827] (n=4243) |

## CASIA zh equal-sequence-length violation
- Check (data build, seqlen_check_M5.csv): at the E0 setting (64x64, 1 px per-stroke resampling, L=320), 9,241 of the 20,000 CASIA zh training units (46.2 %) and 117 of the 250 CASIA zh val units (46.8 %) have more than 320 points at 1 px. Other scripts: casia_en train 42 (0.21 %), hpl_tamil train 151 (0.76 %) / val 5, iam train 14 (0.07 %), ouhdl train 14 with step > 1 (0.07 %); their 1 px counts are estimates from the decoded sequence (19,986 of 20,000 counts exact), and 11 of the 14 estimates exceed 320, 0 in casia_en/iam/ouhdl val.
- Handling: the E0 adapter rule, unchanged: the per-stroke resampling step grows by x1.05 until the unit fits 320 points (no truncation). CASIA zh train step max 2.786 px; 90th percentile 1.407 px. Sequence length 320 and all model/training settings were kept; no unit was dropped.
- Test: casia_zh500 has 243 of 500 units with step > 1 (max 1.886); hpl_tamil500 4; ouhdl_test 3; ouhdl_gold 1; casia_en500 and iam500 0 (unit_steps_eval_M5.csv). Scoring compares predictions to the unresampled GT ink on the evaluation canvas (gt_points), for both protocols.
- Numbers it affects: the training targets of both arms (CASIA zh units at a coarser point spacing; both arms contain the same CASIA zh sample); the CASIA zh part of the pooled selection values of both arms (selection_M5.csv per-valset rows); all casia_zh500 rows; and indirectly every row of the joint and LUO models and of the fine-tunes (initialised from LUO). stepsplit_M5.csv reports every eval row split into step=1 and step>1 units.

### casia_zh500 split by resampling step (native protocol)

| method | step_group | aiou | direction_acc | kendall_tau | ldtw | n_units | n_writers |
|---|---|---|---|---|---|---|---|
| e0_ms8_x4a | step=1 | 0.4343 [0.4242, 0.4446] | 0.6496 [0.5813, 0.7167] | -0.6608 [-0.8170, -0.5030] | 1.1176 [1.0800, 1.1561] | 257 | 56 |
| e0_ms8_x4a | step>1 (n_points_1px>320) | 0.4591 [0.4511, 0.4664] | 0.5818 [0.4956, 0.6637] | -0.7374 [-0.9140, -0.5490] | 0.9273 [0.9023, 0.9534] | 243 | 58 |
| m5_joint_ms9_ddim50 | step=1 | 0.7506 [0.7436, 0.7575] | 0.9709 [0.9563, 0.9831] | 0.9735 [0.9513, 0.9880] | 0.2801 [0.2666, 0.2937] | 257 | 56 |
| m5_joint_ms9_ddim50 | step>1 (n_points_1px>320) | 0.7107 [0.7038, 0.7176] | 0.9556 [0.9412, 0.9680] | 0.9466 [0.9162, 0.9711] | 0.2601 [0.2471, 0.2731] | 243 | 58 |
| m5_luo_ms6_ddim50 | step=1 | 0.7291 [0.7224, 0.7358] | 0.9676 [0.9536, 0.9791] | 0.9591 [0.9375, 0.9770] | 0.2858 [0.2699, 0.3022] | 257 | 56 |
| m5_luo_ms6_ddim50 | step>1 (n_points_1px>320) | 0.6904 [0.6822, 0.6983] | 0.9540 [0.9368, 0.9683] | 0.9645 [0.9441, 0.9808] | 0.2652 [0.2524, 0.2790] | 243 | 58 |

## Files
- results_M5.csv: all rows (model x eval set x input x protocol x metric), mean + writer-cluster CI.
- paired_M5_vs_E0.csv, stepsplit_M5.csv, selection_M5.csv (every milestone, pooled and per val set, plus the E0 val1000 table), keytable_native_M5.csv.
- PROVENANCE_M5_results.json (inputs with sha256, code, Slurm, nodes/GPUs, selected checkpoints, incidents); PROVENANCE_M5.json (run-level, assembled by the FT job).
- m5_report_aggregate.py: the aggregation code (re-run reproduces the CSVs).
