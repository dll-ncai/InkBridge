"""Build PROVENANCE_PENNET[_smoke].json for exp/PENNET (stdlib only; run with any python3).
python3 pennet_provenance.py --exp E --code C --mode smoke|full [--extra key=path ...]
Collects: paper + official repo commit and file sha256s, code tarball/file sha256s, input data sha256s, uhwr_eval 0.3.1
release tarball sha256 and file sha256s, container image (sha256 file written by the smoke driver), env freeze sha256,
parameters (run_config*.json), seeds, job scripts, Slurm records (sacct for every id in slurm_ids.txt), GPU types
(job_history.txt, run_config*.json), checkpoint sha256s, output sha256s.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shas(pattern):
    return {os.path.relpath(p, E): dict(sha256=sha(p), bytes=os.path.getsize(p)) for p in sorted(glob.glob(pattern)) if os.path.isfile(p)}


def jl(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return f"unreadable: {e}"


ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--mode", required=True)
a = ap.parse_args()
E, C = a.exp, a.code
U = "$U"
ids = []
if os.path.exists(f"{E}/slurm_ids.txt"):
    ids = [l.split()[0] for l in open(f"{E}/slurm_ids.txt") if l.strip() and l.split()[0].isdigit()]
sacct = ""
if ids:
    try:
        sacct = subprocess.run(["sacct", "-j", ",".join(sorted(set(ids))), "-X", "-P", "--format",
                                "JobID,JobName,Partition,NodeList,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES,Restarts"],
                               capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        sacct = f"sacct failed: {e}"
repo = f"{U}/repos/PEN-Net"
try:
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    commit = None
P = dict(
    experiment_id="PENNET", mode=a.mode, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    task="PEN-Net re-implementation baseline (DEC-baselines; planned task T08) on OUHD-L v3 ligature units",
    paper=dict(citation="Z. Chen, D. Yang, J. Liang, X. Liu, Y. Wang, Z. Peng, S. Huang. Complex Handwriting Trajectory "
                        "Recovery: Evaluation Metrics and Algorithm. ACCV 2022, LNCS 13842, pp. 58-74 (CVF pp. 1060-1076). "
                        "doi:10.1007/978-3-031-26284-5_4; arXiv:2210.15879"),
    official_code=dict(url="https://github.com/ChenZhounan/PEN-Net", commit=commit, clone=repo,
                       files=({os.path.relpath(p, repo): sha(p) for p in sorted(glob.glob(f"{repo}/**/*", recursive=True))
                               if os.path.isfile(p) and "/.git/" not in p} if os.path.isdir(repo) else None),
                       status="not runnable as released: train.py builds model.Net_tr_2d (absent from model.py); config.py, "
                              "datasets/data_utils.py, soft_dtw_cuda.py, visualization.py, iam/hindi/japanese loaders are "
                              "missing; no weights; no licence file. Used as architecture reference only."),
    code=dict(dir=C, tarballs=shas(f"{E}/pennet_code_*.tar.gz"),
              files={os.path.relpath(p, C): sha(p) for p in sorted(glob.glob(f"{C}/**/*", recursive=True))
                     if os.path.isfile(p) and "__pycache__" not in p}),
    inputs=dict(data_dir=f"{U}/data/ouhdl_e0_v3",
                npz_sha256=open(f"{E}/data_sha256.txt").read().split("\n") if os.path.exists(f"{E}/data_sha256.txt") else None,
                uhwr_eval=dict(version="0.3.1", dir=f"{U}/shared/uhwr_eval-0.3.1",
                               release_tarball_sha256=sha(f"{U}/shared/uhwr_eval-0.3.1/release/uhwr_eval-0.3.1.tar.gz"),
                               files={os.path.relpath(p, f"{U}/shared/uhwr_eval-0.3.1"): sha(p)
                                      for p in sorted(glob.glob(f"{U}/shared/uhwr_eval-0.3.1/uhwr_eval/*.py"))}),
                uhwr_baselines_for_summary=f"{U}/e0/code_4cd40f57/uhwr_baselines (symlinked as {E}/lib/uhwr_baselines)",
                e0_scorer="e0_score.py + e0_adapter.py copied unchanged from $U/e0/code_4cd40f57 (cmp-checked by the smoke driver)"),
    environment=dict(container=f"{U}/images/torch27.sif",
                     container_sha256=open(f"{E}/sif_sha256.txt").read().strip() if os.path.exists(f"{E}/sif_sha256.txt") else None,
                     venv=f"{U}/venvs/e0", env_freeze=shas(f"{E}/env_freeze_*.txt"),
                     scoring_note="scoring runs from a /dev/shm copy of venvs/e0 with node-local NUMBA_CACHE_DIR "
                                  "(libllvmlite workaround of exp/M1b/code_v4_scorefix)"),
    seeds=dict(train_seed=20261016, data_order="epoch e: np.random.default_rng(seed + e).permutation(N), drop_last",
               decoding="greedy arg-max (deterministic, no sampling seed)", val_subset_seed=0, bootstrap_seed=0, bootstrap_B=2000),
    parameters={os.path.basename(p): jl(p) for p in sorted(glob.glob(f"{E}/run/run_config*.json"))},
    smoke=dict(results=jl(f"{E}/smoke/smoke_gate.json") if os.path.exists(f"{E}/smoke/smoke_gate.json") else None,
               tests=jl(f"{E}/smoke/tests.json") if os.path.exists(f"{E}/smoke/tests.json") else None),
    slurm=dict(ids_file=open(f"{E}/slurm_ids.txt").read() if os.path.exists(f"{E}/slurm_ids.txt") else None, sacct=sacct,
               job_history=open(f"{E}/run/job_history.txt").read() if os.path.exists(f"{E}/run/job_history.txt") else None),
    checkpoints=shas(f"{E}/run/milestones/ms_*.pt"),
    training_complete=jl(f"{E}/run/training_complete.json") if os.path.exists(f"{E}/run/training_complete.json") else None,
    selection=jl(f"{E}/eval/selection.json") if os.path.exists(f"{E}/eval/selection.json") else None,
    outputs={**shas(f"{E}/eval/*.csv"), **shas(f"{E}/eval/*.json"), **shas(f"{E}/eval/*.md"), **shas(f"{E}/eval/preds/*.npz")},
    decisions_file=f"{C}/README_PENNET.md",
)
if a.mode == "smoke":
    P["outputs"] = {**shas(f"{E}/smoke/*.json"), **shas(f"{E}/smoke/*.csv"), **shas(f"{E}/smoke/*.txt")}
    P["checkpoints"] = {}
out = f"{E}/PROVENANCE_PENNET{'_smoke' if a.mode == 'smoke' else ''}.json"
json.dump(P, open(out, "w"), indent=1)
print(out)
