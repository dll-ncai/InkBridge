"""Assemble exp/M5/PROVENANCE_M5.json from what is on disk (idempotent; re-run at the end of every M5 job).
Scans every run dir (a dir holding run_config.json) under exp/M5, the data set data/m5_multiscript_v1, the code
dirs, the env records and the per-job command copies.  Checkpoint sha256s are cached in <run>/ckpt_sha256/.
python m5_provenance.py --m5 $U/exp/M5 [--runroot ignored]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import time

DATA = "$U/data/m5_multiscript_v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    try:
        return json.load(open(p))
    except Exception:   # noqa: BLE001
        return None


def ckpt_sha(rd, f):
    cdir = os.path.join(rd, "ckpt_sha256"); os.makedirs(cdir, exist_ok=True)
    side = os.path.join(cdir, os.path.basename(f) + ".sha256")
    st = os.stat(f)
    if os.path.exists(side):
        d = jload(side)
        if d and d.get("mtime") == st.st_mtime and d.get("bytes") == st.st_size:
            return d["sha256"]
    h = sha(f)
    json.dump(dict(sha256=h, mtime=st.st_mtime, bytes=st.st_size), open(side, "w"))
    return h


def run_record(rd):
    r = dict(run_dir=rd)
    r["run_config"] = jload(os.path.join(rd, "run_config.json"))
    r["resume_configs"] = {os.path.basename(p): (jload(p) or {}).get("e0") for p in sorted(glob.glob(os.path.join(rd, "run_config_resume_*.json")))}
    r["init_from"] = jload(os.path.join(rd, "init_from.json"))
    r["training_complete"] = jload(os.path.join(rd, "training_complete.json"))
    jh = os.path.join(rd, "job_history.txt")
    r["job_history"] = open(jh).read().splitlines() if os.path.exists(jh) else []
    r["slurm_job_ids"] = sorted({m for l in r["job_history"] for m in re.findall(r"slurm=(\d+)", l)})
    r["checkpoints"] = {os.path.basename(f): dict(sha256=ckpt_sha(rd, f), bytes=os.path.getsize(f),
                                                  mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(f))))
                        for f in sorted(glob.glob(os.path.join(rd, "checkpoints", "checkpoint_*.pt")))}
    ev = os.path.join(rd, "eval")
    r["selection"] = jload(os.path.join(ev, "selection.json"))
    r["eval_outputs"] = {os.path.basename(f): dict(sha256=sha(f), bytes=os.path.getsize(f))
                         for f in sorted(glob.glob(os.path.join(ev, "*"))) if os.path.isfile(f) and ".part." not in f}
    sl = os.path.join(ev, "shard_log.tsv")
    r["sampling_log"] = [dict(zip(["file", "start_utc", "end_utc", "slurm", "restart", "host", "gpu", "sha256"], l.split("\t")))
                         for l in open(sl).read().splitlines()] if os.path.exists(sl) else []
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m5", required=True); ap.add_argument("--runroot")
    a = ap.parse_args()
    M = a.m5
    P = dict(experiment="M5 (X6)", title="One model for many scripts: joint and leave-Urdu-out Nagamatsu diffusion models (E0 recipe) "
             "on m5_multiscript_v1; zero-shot Urdu; data-efficiency fine-tunes 1/10/30/100 % of OUHD-L train",
             plan_refs=["experiments_to_run_2026-10-18.md v b474164e section X6 / M5",
                        "experiment_schedule_ijdar_2026-10-18.md v 4a36850e (DEC-x6: M5 on the B200 before LB5/LB6)"],
             assembled_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), m5_dir=M)
    P["data"] = dict(dir=DATA, files={os.path.basename(f): dict(sha256=sha(f), bytes=os.path.getsize(f))
                                      for f in sorted(glob.glob(os.path.join(DATA, "*"))) if os.path.isfile(f)})
    P["eval_inputs"] = {p: dict(sha256=sha(p)) for p in [
        "$U/data/ouhdl_e0_v3/e0_test.npz", "$U/data/ouhdl_e0_v3/e0_gold.npz",
        "$U/data/ouhdl_e0_v3/e0_val.npz", "$U/data/ouhdl_e0_v3/e0_train.npz",
        "$U/exp/X4a/data/x4a_casia_zh.npz", "$U/exp/X4a/data/x4a_casia_en.npz",
        "$U/exp/X4a/data/x4a_hpl_tamil.npz"] if os.path.exists(p)}
    P["code"] = {}
    for d in sorted(glob.glob(os.path.join(M, "code_*"))):
        if os.path.isdir(d):
            P["code"][os.path.basename(d)] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(d, "*"))) if os.path.isfile(f)}
    P["code_tarballs"] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(M, "code_tarballs", "*")))}
    P["e0_eval_code"] = dict(dir="$U/e0/code_4cd40f57", note="e0_sample.py / e0_score.py / e0_adapter.py / e0_dataset.py unchanged (final-eval tarball 4cd40f57)")
    P["nagamatsu_repo"] = dict(path="$U/repos/nagamatsu", commit=open(os.path.join(M, "env", "repo_commit.txt")).read().strip()
                               if os.path.exists(os.path.join(M, "env", "repo_commit.txt")) else None)
    P["environment"] = {os.path.basename(f): dict(sha256=sha(f), content=open(f).read()[:2000] if f.endswith(("sha256.txt", "commit.txt", "env_check.txt")) else None)
                        for f in sorted(glob.glob(os.path.join(M, "env", "*"))) if os.path.isfile(f)}
    P["container"] = dict(path="$U/images/torch27.sif", venv="$U/venvs/e0")
    P["job_commands"] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(M, "jobs", "*")))}
    P["prep"] = dict(job_history=open(os.path.join(M, "prep", "job_history.txt")).read().splitlines()
                     if os.path.exists(os.path.join(M, "prep", "job_history.txt")) else [])
    runs = {}
    for cfg in sorted(glob.glob(os.path.join(M, "**", "run_config.json"), recursive=True)):
        rd = os.path.dirname(cfg); runs[os.path.relpath(rd, M)] = run_record(rd)
    P["runs"] = runs
    P["outputs"] = {os.path.relpath(f, M): sha(f) for f in sorted(glob.glob(os.path.join(M, "**", "summary_M5_*.csv"), recursive=True))}
    tmp = os.path.join(M, ".PROVENANCE_M5.json.tmp%d" % os.getpid())
    json.dump(P, open(tmp, "w"), indent=1, default=str); os.replace(tmp, os.path.join(M, "PROVENANCE_M5.json"))
    print("PROVENANCE_M5.json", len(runs), "runs")


if __name__ == "__main__":
    main()
