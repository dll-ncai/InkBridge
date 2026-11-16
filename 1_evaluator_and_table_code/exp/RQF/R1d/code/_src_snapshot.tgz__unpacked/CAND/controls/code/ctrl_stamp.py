"""Stamp CAND control sidecars with job/code provenance and write PROVENANCE_ctrl_<ARM>.json (stdlib only).
python ctrl_stamp.py --arm ARM --preds-dir DIR --run-dir RD --out PROV.json --job JOBID --restart N --ctrl-code DIR --frame-code DIR"""
import argparse, glob, hashlib, json, os, platform, time
def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None
ap = argparse.ArgumentParser()
for k in ("--arm", "--preds-dir", "--run-dir", "--out", "--job", "--restart", "--ctrl-code", "--frame-code"): ap.add_argument(k, required=True)
ap.add_argument("--label-glob", required=True, help="glob of sidecar json files to stamp, relative to --preds-dir")
ap.add_argument("--env-json", default="", help="json string of the arm env knobs")
a = ap.parse_args()
cc_sums = os.path.join(a.ctrl_code, "SHA256SUMS.txt"); fc_sums = os.path.join(a.frame_code, "SHA256SUMS.txt")
stamp = dict(slurm_job_id=a.job, slurm_restart=a.restart, ctrl_code_dir=a.ctrl_code, ctrl_code_SHA256SUMS_sha256=sha(cc_sums),
             ctrl_sample_sha256=sha(os.path.join(a.ctrl_code, "ctrl_sample.py")), frame_code_dir=a.frame_code,
             frame_code_tag=open(os.path.join(a.frame_code, "CODE_TAG.txt")).read().strip(), frame_code_SHA256SUMS_sha256=sha(fc_sums),
             stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node())
rows = []
for js in sorted(glob.glob(os.path.join(a.preds_dir, a.label_glob))):
    m = json.load(open(js)); npz = js[:-5] + ".npz"
    m.setdefault("cand_control", {}).update(stamp, arm=a.arm)
    m["preds_sha256"] = sha(npz)
    json.dump(m, open(js + ".tmp", "w"), indent=1); os.replace(js + ".tmp", js)
    rows.append(dict(file=os.path.basename(npz), sha256=m["preds_sha256"], set=m.get("set"), seed=m.get("seed"), n_units=m.get("n_units"),
                     device=m.get("device"), checkpoint_sha256=m.get("checkpoint_sha256"), npz_sha256=m.get("npz_sha256"), npz_rel=m.get("npz_rel"),
                     sampling_job=m.get("cand_control", {}).get("slurm_job_id")))
rd = a.run_dir; ev = os.path.join(rd, "eval")
ck = sorted(glob.glob(os.path.join(rd, "checkpoints", "checkpoint_*.pt")))
best = open(os.path.join(ev, "best_milestone.txt")).read().strip() if os.path.exists(os.path.join(ev, "best_milestone.txt")) else None
prov = dict(arm=a.arm, written_utc=stamp["stamped_utc"], env=json.loads(a.env_json) if a.env_json else None, job=stamp,
            run_config=jl(os.path.join(rd, "run_config.json")), replay_config=jl(os.path.join(rd, "replay_config.json")),
            training_complete=jl(os.path.join(rd, "training_complete.json")), selected_milestone=best,
            selection=jl(os.path.join(ev, "selection.json")),
            checkpoints={os.path.basename(p): sha(p) for p in ck} if ck else {},
            frame_provenance=jl(os.path.join(ev, f"PROVENANCE_frame_{a.arm}.json")) is not None,
            preds=rows)
json.dump(prov, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
print(json.dumps(dict(event="stamped", arm=a.arm, n_preds=len(rows), out=a.out)))
