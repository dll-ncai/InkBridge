#!/usr/bin/env python3
"""XS sizing probe: can 224-px renders of the LUO training units (m5_luo_train.npz has no gt ink) be derived from the stored Nagamatsu seq,
and how fast?  decode(seq) -> ink on the 64-px canvas -> fmd_harness render64 vs stored cond_rendered; render224 speed.  500 units, 125 per script (seed 20261103)."""
import sys, json, time, hashlib
import numpy as np
sys.path.insert(0, "$U/exp/FMD/code/fmd_harness_v1"); sys.path.insert(0, "$U/e0/code_4cd40f57")
from fmd_harness import common as C, render as RN
import e0_adapter as A
_Z = np.load("$U/data/m5_multiscript_v1/m5_luo_train.npz", allow_pickle=False); Z = {k: _Z[k] for k in ("scripts", "seq", "seq_offsets", "cond_rendered", "step")}
sc = Z["scripts"]; rng = np.random.default_rng(20261103); out = {}; t_dec = t_64 = t_224 = 0.0; n = 0
for s in sorted(set(sc.tolist())):
    idx = np.sort(rng.choice(np.flatnonzero(sc == s), 125, replace=False)); same = 0; mx = 0; mean_d = []
    for i in idx:
        seq = Z["seq"][Z["seq_offsets"][i]:Z["seq_offsets"][i+1]]
        t0 = time.time(); ink = A.decode(seq); t_dec += time.time() - t0
        t0 = time.time(); r64 = RN.render64(ink); t_64 += time.time() - t0
        t0 = time.time(); r224 = RN.render224(ink); t_224 += time.time() - t0; n += 1
        d = np.abs(r64.astype(int) - Z["cond_rendered"][i].astype(int)); same += int(d.max() == 0); mx = max(mx, int(d.max())); mean_d.append(float(d.mean()))
    out[s] = dict(n=int(len(idx)), identical_64=same, max_abs_diff=mx, mean_abs_diff=round(float(np.mean(mean_d)), 4), step_gt1=int((Z["step"][idx] > 1.0).sum()))
out["speed_s_per_unit"] = dict(decode=t_dec / n, render64=t_64 / n, render224=t_224 / n, n=n)
out["note"] = "decode(seq) is the 1-px arc-length resampled ink (step grown until <=320 pts); stored cond_rendered was rendered from the un-resampled ink"
json.dump(out, open("$U/exp/FMD/xs/admin/luo_render_probe.json", "w"), indent=1); print(json.dumps(out, indent=1))
