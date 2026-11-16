"""Smoke gate for PENNET (run inside the container after the smoke runs). Writes S/smoke_gate.json; exit 1 on failure.
Gates:
  resume   : rA (200 steps, stop, resume to 300) vs rC (300 uninterrupted): both at step 300, same epoch/pos; parameters
             bitwise equal OR every logged loss at steps 210..300 within 1e-3 relative.
  sigterm  : rD received SIGTERM, saved ckpt_latest with tag signal_15 at step s > 0, resumed from s (run_config_resume_s.json)
             and completed s + 20 steps.
  finite   : all logged losses finite in rA, rC, rD.
  decode   : smoke decode + score rows complete (checked by the driver; passed in via --decode-ok).
Records s/step at batch 512 (rC, steps >= 20) and the 100k-step ETA.
"""
import argparse
import glob
import json
import math
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke-dir", required=True)
ap.add_argument("--decode-ok", default="unknown")
a = ap.parse_args()
S = a.smoke_dir
G = {}


def log(run):
    return [json.loads(l) for l in open(f"{S}/{run}/train_log.jsonl")]


A = torch.load(f"{S}/rA/ckpt_latest.pt", map_location="cpu", weights_only=False)
C = torch.load(f"{S}/rC/ckpt_latest.pt", map_location="cpu", weights_only=False)
maxdiff = max(float((A["model"][k].double() - C["model"][k].double()).abs().max()) for k in A["model"])
la = {r["step"]: r["loss"] for r in log("rA")}
lc = {r["step"]: r["loss"] for r in log("rC")}
common = [s for s in sorted(lc) if s >= 210 and s in la]
reld = max(abs(la[s] - lc[s]) / max(abs(lc[s]), 1e-12) for s in common) if common else float("nan")
resume_cfg = glob.glob(f"{S}/rA/run_config_resume_*.json")
G["resume"] = dict(stepA=A["step"], stepC=C["step"], posA=[A["epoch"], A["pos"]], posC=[C["epoch"], C["pos"]],
                   resumed_from=[json.load(open(f))["resumed_from_step"] for f in resume_cfg],
                   param_max_abs_diff=maxdiff, bitwise_equal=maxdiff == 0.0, loss_max_rel_diff_210_300=reld,
                   ok=bool(A["step"] == C["step"] == 300 and A["epoch"] == C["epoch"] and A["pos"] == C["pos"]
                           and resume_cfg and (maxdiff == 0.0 or reld <= 1e-3)))
ev = open(f"{S}/rD/ckpt_events.txt").read().strip().split("\n")
sig = [l for l in ev if "signal_15" in l]
s_sig = int(sig[0].split("step=")[1].split()[0]) if sig else -1
tc = json.load(open(f"{S}/rD/training_complete.json")) if glob.glob(f"{S}/rD/training_complete.json") else {}
G["sigterm"] = dict(events=ev, signal_step=s_sig, resume_config=bool(glob.glob(f"{S}/rD/run_config_resume_{s_sig}.json")),
                    final_step=tc.get("step"),
                    ok=bool(s_sig > 0 and glob.glob(f"{S}/rD/run_config_resume_{s_sig}.json") and tc.get("step") == s_sig + 20))
allv = [r["loss"] for run in ("rA", "rC", "rD") for r in log(run)]
G["finite"] = dict(n=len(allv), ok=all(math.isfinite(v) for v in allv))
rc = [r for r in log("rC") if r["step"] >= 20]
sps = sum(r["s_per_step"] for r in rc) / len(rc)
cfg = json.load(open(f"{S}/rC/run_config.json"))
G["timing"] = dict(device=cfg["device"], capability=cfg["capability"], n_params=cfg["n_params"], s_per_step_bs512=sps,
                   eta_100k_h=sps * 100000 / 3600, mean_T=sum(r["T"] for r in rc) / len(rc),
                   loss_step1=log("rC")[0]["loss"], loss_step300=lc.get(300))
G["decode"] = dict(ok=a.decode_ok == "1")
G["eta_100k_h"] = G["timing"]["eta_100k_h"]
G["all_ok"] = bool(all(G[k]["ok"] for k in ("resume", "sigterm", "finite", "decode")))
json.dump(G, open(f"{S}/smoke_gate.json", "w"), indent=1)
print(json.dumps(G))
sys.exit(0 if G["all_ok"] else 1)
