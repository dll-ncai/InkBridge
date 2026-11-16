# RESULTS R1a (mirrored-LTR control; CONFIRMATORY)

Generated 2026-11-12T19:49:44Z; rows files: 112; scorer uhwr_eval 0.3.1 native; per-unit mean over sampling seeds; writer-cluster percentile bootstrap B=2000 seed 0, 95 %; paired on the unit intersection.
Columns: AIoU | direction | tau | strict | pen-up F1 v2 (point [lo, hi]).

## Base LUO ms6
### iam500

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=63) 
| -0.0296 [-0.0356, -0.0242] | -0.4839 [-0.5270, -0.4394] | -1.2363 [-1.3733, -1.1040] | -0.3160 [-0.3641, -0.2719] | -0.2053 [-0.2429, -0.1719] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=63) 
| +0.0296 [+0.0242, +0.0356] | +0.4838 [+0.4393, +0.5268] | +1.2363 [+1.1040, +1.3733] | +0.3160 [+0.2719, +0.3641] | +0.2053 [+0.1719, +0.2429] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True
### casia_en

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=60) 
| -0.0375 [-0.0422, -0.0327] | -0.3751 [-0.4109, -0.3406] | -0.8984 [-1.0500, -0.7479] | -0.3247 [-0.3596, -0.2908] | -0.2438 [-0.2683, -0.2179] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=60) 
| +0.0375 [+0.0327, +0.0421] | +0.3751 [+0.3406, +0.4109] | +0.8984 [+0.7479, +1.0500] | +0.3247 [+0.2908, +0.3596] | +0.2438 [+0.2179, +0.2683] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True
### hpl_tamil

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=52) 
| -0.1005 [-0.1096, -0.0906] | -0.6429 [-0.6862, -0.6028] | -1.0602 [-1.1948, -0.9154] | -0.5173 [-0.5562, -0.4768] | -0.4832 [-0.5260, -0.4386] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=52) 
| +0.1005 [+0.0906, +0.1096] | +0.6429 [+0.6028, +0.6862] | +1.0662 [+0.9170, +1.2089] | +0.5167 [+0.4756, +0.5556] | +0.4832 [+0.4386, +0.5260] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True

Rule R1a, base LUO ms6: COLLAPSE met on 3/3 sets -> True; REPAIR met on 3/3 sets -> True; complete=True.

## Base joint M5 ms9
### iam500

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=63) 
| -0.0227 [-0.0277, -0.0182] | -0.4773 [-0.5238, -0.4305] | -1.2008 [-1.3462, -1.0579] | -0.2953 [-0.3386, -0.2556] | -0.1700 [-0.2086, -0.1322] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=63) 
| +0.0226 [+0.0181, +0.0276] | +0.4773 [+0.4305, +0.5238] | +1.2018 [+1.0583, +1.3482] | +0.2953 [+0.2556, +0.3386] | +0.1700 [+0.1322, +0.2086] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True
### casia_en

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=60) 
| -0.0297 [-0.0351, -0.0247] | -0.3579 [-0.3954, -0.3220] | -0.9578 [-1.1068, -0.8027] | -0.3080 [-0.3477, -0.2692] | -0.2047 [-0.2381, -0.1732] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=60) 
| +0.0297 [+0.0247, +0.0351] | +0.3579 [+0.3219, +0.3954] | +0.9533 [+0.7993, +1.1063] | +0.3087 [+0.2697, +0.3484] | +0.2047 [+0.1732, +0.2381] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True
### hpl_tamil

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **identity(mirrored) - identity(original)** (n=500, writers=52) 
| -0.0842 [-0.0923, -0.0760] | -0.6006 [-0.6463, -0.5564] | -1.0946 [-1.2330, -0.9540] | -0.5200 [-0.5685, -0.4735] | -0.4822 [-0.5244, -0.4377] |
| **mirror_x(mirrored) - identity(mirrored)** (n=500, writers=52) 
| +0.0842 [+0.0760, +0.0923] | +0.6006 [+0.5564, +0.6463] | +1.0946 [+0.9540, +1.2330] | +0.5200 [+0.4735, +0.5685] | +0.4822 [+0.4377, +0.5244] |
COLLAPSE clause (direction CI hi < -0.10): True; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > -0.01035): True

Rule R1a, base joint M5 ms9: COLLAPSE met on 3/3 sets -> True; REPAIR met on 3/3 sets -> True; complete=True.

Note fixed in advance (prereg): mirror_x on a mirrored LTR set restores the original orientation, so REPAIR is partly a consistency check; COLLAPSE is the informative clause.
