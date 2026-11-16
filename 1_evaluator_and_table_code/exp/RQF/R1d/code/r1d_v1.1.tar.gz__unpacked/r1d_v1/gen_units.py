#!/usr/bin/env python3
"""R1d sampling units (same definitions as submit_r1d.sh race calls) -> units.json {unit: base, kind, time, args, files}."""
import json, sys
P = json.load(open(sys.argv[1])); RD = P["root"]; SIX = "test_casia_zh test_casia_en test_hpl_tamil test_iam500 german_unipen_de_test german_onhw_all".split()
U = {}
def add(u, b, t, kind, label, frames, sets, args):
    U[u] = dict(base=b, kind=kind, time=t, args=[kind] + args, files=[f"{s}__{label}_{fr}__seed{k}_preds.{x}" for s in sets for fr in frames.split(",") for k in range(3) for x in ("npz", "json")])
add("LUO6", "LUO_ms6", "08:00:00", "e0", "LUO_ms6", "mirror_x", SIX, ["LUO_ms6", "exp/M5/runs/luo", "6", f"{RD}/preds/LUO_ms6", "mirror_x"] + SIX)
add("M5u", "M5_ms9", "06:00:00", "e0", "M5_joint_ms9", "mirror_x", ["test_ouhdl"], ["M5_joint_ms9", "exp/M5/runs/joint", "9", f"{RD}/preds/M5_ms9", "mirror_x", "test_ouhdl"])
add("M56", "M5_ms9", "08:00:00", "e0", "M5_joint_ms9", "mirror_x", SIX, ["M5_joint_ms9", "exp/M5/runs/joint", "9", f"{RD}/preds/M5_ms9", "mirror_x"] + SIX)
for b in ("NOFM_W_bb", "PEINIT_W_bb"):
    e = P["bases"][b]; S = e["sets"]; k = b.split("_")[0]; tri = lambda s: f"{s}:{S[s]['render']}:{S[s]['gt']}"
    add(f"{k}u_id", b, "08:00:00", "bb", e["label"], "identity", ["test_ouhdl"], [e["label"], e["ckpt"], f"{RD}/preds/{b}", "identity", tri("test_ouhdl")])
    add(f"{k}u_mir", b, "08:00:00", "bb", e["label"], "mirror_x", ["test_ouhdl"], [e["label"], e["ckpt"], f"{RD}/preds/{b}", "mirror_x", tri("test_ouhdl")])
    add(f"{k}6", b, "12:00:00", "bb", e["label"], "mirror_x", SIX, [e["label"], e["ckpt"], f"{RD}/preds/{b}", "mirror_x"] + [tri(s) for s in SIX])
json.dump(U, open(sys.argv[2], "w"), indent=1); print(len(U), "units")
