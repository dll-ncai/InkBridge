#!/usr/bin/env python3
"""Build RQF/R1d/plan.json: per base x 7 test sets the identity / mirror_x 3-seed pools (existing files reused when present at build time,
else the path the R1d sampler will write) + sampling units. Checkpoint sha256 of every reused identity pool is checked against the base."""
import json, os, hashlib, time, sys
R = "$U"; Q = R + "/exp/RQF"; RD = Q + "/R1d"; X = R + "/exp/FMD/xs"; MON = X + "/runs/fmdx_monitor"
def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
GT = {"test_ouhdl": R + "/data/ouhdl_e0_v3/e0_test.npz", "test_casia_zh": R + "/exp/X4a/data/x4a_casia_zh.npz", "test_casia_en": R + "/exp/X4a/data/x4a_casia_en.npz",
      "test_hpl_tamil": R + "/exp/X4a/data/x4a_hpl_tamil.npz", "test_iam500": R + "/data/m5_multiscript_v1/m5_test_iam500.npz",
      "german_unipen_de_test": R + "/data/d2_v2/unipen_de/e0_test.npz", "german_onhw_all": R + "/data/d2_v2/onhw/e0_all.npz"}
SIX = [s for s in GT if s != "test_ouhdl"]
S3 = lambda f: [f.format(k=k) for k in range(3)]
B = {}; units = []; ck = {}
ck["LUO_ms6"] = R + "/exp/M5/runs/luo/checkpoints/checkpoint_000006.pt"; ck["M5_ms9"] = R + "/exp/M5/runs/joint/checkpoints/checkpoint_000009.pt"
ck["NOFM_W_bb"] = X + "/runs/NOFM_W_clusterA/bb_ms3.pt"; ck["PEINIT_W_bb"] = X + "/runs/PEINIT_W_clusterA/bb_ms4.pt"
cks = {b: sha(p) for b, p in ck.items()}
for b, run, ms in (("NOFM_W_bb", "NOFM_W_clusterA", 3), ("PEINIT_W_bb", "PEINIT_W_clusterA", 4)):
    best = open(f"{X}/runs/{run}/eval/best_milestone.txt").read().strip(); assert best == str(ms), (b, best)
# LUO ms6
L = {"test_ouhdl": dict(gt=GT["test_ouhdl"], id_pool=S3(R + "/exp/CAND/K1/preds/test_ouhdl__LUO_ms6_identity__seed{k}_preds.npz"),
                        mir_pool=S3(R + "/exp/CAND/K1/preds/test_ouhdl__LUO_ms6_mirror_x__seed{k}_preds.npz"))}
for s in SIX:
    L[s] = dict(gt=GT[s], id_pool=S3(MON + f"/comparators_a4000/LUO/preds/{s}__LUO_ms6_identity__seed{{k}}_preds.npz"),
                mir_pool=S3(RD + f"/preds/LUO_ms6/preds/{s}__LUO_ms6_mirror_x__seed{{k}}_preds.npz"))
B["LUO_ms6"] = dict(ckpt=ck["LUO_ms6"], ckpt_sha256=cks["LUO_ms6"], sets=L)
M = {s: dict(gt=GT[s], id_pool=S3(R + f"/exp/D3_ROUTE/stageA/preds/{s}__M5_joint_ms9__seed{{k}}_preds.npz"),
             mir_pool=S3(RD + f"/preds/M5_ms9/preds/{s}__M5_joint_ms9_mirror_x__seed{{k}}_preds.npz")) for s in GT}
B["M5_ms9"] = dict(ckpt=ck["M5_ms9"], ckpt_sha256=cks["M5_ms9"], sets=M, note="identity pool = D3 Stage A d3a_sample.py (RTX 4090); sidecars carry no ckpt sha")
for b, arm, ms in (("NOFM_W_bb", "NOFM_W", 3), ("PEINIT_W_bb", "PEINIT_W", 4)):
    lab = f"{arm}_bb_ms{ms}"; S = {}
    S["test_ouhdl"] = dict(gt=GT["test_ouhdl"], render=R + "/exp/FMD/data/fmd_render224_test.npz",
                           id_pool=S3(RD + f"/preds/{b}/preds/test_ouhdl__{lab}_identity__seed{{k}}_preds.npz"),
                           mir_pool=S3(RD + f"/preds/{b}/preds/test_ouhdl__{lab}_mirror_x__seed{{k}}_preds.npz"))
    for s in SIX:
        S[s] = dict(gt=GT[s], render=X + f"/data/monitor_v1/test_{s}_render224.npz", id_pool=S3(MON + f"/preds/{arm}/bb/{s}__{lab}__seed{{k}}_preds.npz"),
                    mir_pool=S3(RD + f"/preds/{b}/preds/{s}__{lab}_mirror_x__seed{{k}}_preds.npz"))
    B[b] = dict(ckpt=ck[b], ckpt_sha256=cks[b], label=lab, sets=S)
# reuse check: ckpt sha of existing identity pools
for b, e in B.items():
    for s, x in e["sets"].items():
        for f in x["id_pool"] + x["mir_pool"]:
            if os.path.exists(f) and os.path.exists(f[:-4] + ".json"):
                j = json.load(open(f[:-4] + ".json")); c = j.get("checkpoint_sha256") or j.get("ckpt_sha256")
                assert c in (None, e["ckpt_sha256"]), ("ckpt sha mismatch", b, f, c)
        x["reused"] = {k: all(os.path.exists(f) for f in x[k]) for k in ("id_pool", "mir_pool")}
P = dict(kind="rqf_r1d_plan", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), root=RD,
         agent="<agent>", lead_reason="Model <agent>: new inference code",
         prereg=dict(parent="PREREG_RQ_finalise_2026-11-07.md sha256 04766e973badf7ff469d468e25bab17cebabf38f2c9537d731a4d91ea960c624",
                     amendment="A2 sha256 bface42b... (A2.5 R1d)"),
         gt_sha256={s: sha(p) for s, p in GT.items()}, bases=B)
json.dump(P, open(RD + "/plan.json", "w"), indent=1)
print(json.dumps({b: {s: x["reused"] for s, x in e["sets"].items()} for b, e in B.items()}))
