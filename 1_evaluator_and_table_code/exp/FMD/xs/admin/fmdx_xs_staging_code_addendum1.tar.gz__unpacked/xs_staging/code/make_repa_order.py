#!/usr/bin/env python3
"""XS sizing: fixed, nested, script-stratified ordering of the 80,000 LUO training units for REPA-target subsets.
rule: for script s (sorted tag order casia_en, casia_zh, hpl_tamil, iam; i = 0..3) the file rows of that script (in file order) are permuted by np.random.default_rng([20261103, 7, i]).permutation(20000);
rank_in_script = position in that permutation.  A subset of size N = all rows with rank_in_script < N/4  (nested prefixes: 10k = 2,500 per script, 20k = 5,000, 40k = 10,000, 80k = all).
Output csv columns: row (index into m5_luo_train.npz), unit_id, script, sample_id (writer key), rank_in_script."""
import sys, csv, json, hashlib
import numpy as np
sys.path.insert(0, "$U/exp/FMD/code/fmd_harness_v1")
from fmd_harness import common as C
src = "$U/data/m5_multiscript_v1/m5_luo_train.npz"
Z = np.load(src, allow_pickle=False); sc = Z["scripts"]; uid = Z["unit_ids"]; sid = Z["sample_ids"]
rows = []; counts = {}
for i, s in enumerate(sorted(set(sc.tolist()))):
    idx = np.flatnonzero(sc == s); perm = np.random.default_rng([20261103, 7, i]).permutation(len(idx)); counts[s] = int(len(idx))
    for rank, j in enumerate(perm): rows.append((int(idx[j]), str(uid[idx[j]]), s, str(sid[idx[j]]), rank))
rows.sort(key=lambda r: (r[4], r[2]))     # interleaved by rank so any prefix of the file is stratified
out = "$U/exp/FMD/xs/data/luo_repa_order_v1.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["row", "unit_id", "script", "sample_id", "rank_in_script"]); w.writerows(rows)
info = dict(n_rows=len(rows), per_script=counts, writers_in_prefix={str(n): len({r[3] for r in rows if r[4] < n // 4}) for n in (10000, 20000, 40000, 80000)},
            unit_ids_sha256=C.sha256_bytes("\n".join(r[1] for r in rows).encode()))
C.write_meta(out, kind="xs_luo_repa_order", inputs={"m5_luo_train_npz": src}, params=dict(rule="see script docstring", seed_base=[20261103, 7], script_order=sorted(counts), nested_prefixes="rank_in_script < N/4"), extra=dict(result=info))
print(json.dumps(info))
