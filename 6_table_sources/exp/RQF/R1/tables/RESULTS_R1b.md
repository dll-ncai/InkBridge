# RESULTS R1b (E0 ms8, identity vs mirror_x; CONFIRMATORY)

Generated 2026-11-12T19:49:44Z; same protocol as R1a. OnHW (2 writers): unit bootstrap; UNIPEN-de has 4 writers (writer-cluster CI as prereg; very few clusters).

| AIoU | direction | tau | strict | pen-up F1 v2 |
|---|---|---|---|---|
| **german_unipen_de_test** (n=827, writers=7, writer-cluster) 
| -0.0100 [-0.0246, +0.0161] | +0.3735 [+0.2968, +0.4891] | +0.7855 [+0.4608, +1.2333] | +0.1552 [+0.1313, +0.1948] | +0.0273 [-0.0008, +0.0805] |
REPAIR clause: False

| **test_iam500** (n=500, writers=63, writer-cluster) 
| +0.0125 [+0.0003, +0.0254] | +0.3165 [+0.2501, +0.3830] | +0.8620 [+0.6111, +1.1060] | +0.0747 [+0.0370, +0.1140] | +0.0759 [+0.0447, +0.1077] |
REPAIR clause: True

| **test_casia_en** (n=500, writers=60, writer-cluster) 
| -0.0162 [-0.0269, -0.0060] | +0.1105 [+0.0573, +0.1649] | +0.4280 [+0.1567, +0.7579] | +0.0907 [+0.0533, +0.1263] | +0.0244 [-0.0105, +0.0619] |
REPAIR clause: False

| **german_onhw_all** (n=5925, writers=2, unit-cluster) 
| +0.0038 [+0.0023, +0.0052] | +0.6258 [+0.6116, +0.6393] | +1.5647 [+1.5226, +1.6056] | +0.0177 [+0.0134, +0.0223] | +0.0004 [-0.0044, +0.0051] |
REPAIR clause: True

Rule R1b: REPAIR met on 2/4 sets (need >= 3) -> False; complete=True.
