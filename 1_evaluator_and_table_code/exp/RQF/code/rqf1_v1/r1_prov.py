"""RQF R1 PROVENANCE: aggregates sidecars of every preds file (device, host, ckpt/npz/preds sha256), reused K1 rows, code shas, job table."""
import argparse, glob, hashlib, json, os, time
ap = argparse.ArgumentParser(); ap.add_argument("--o", required=True); a = ap.parse_args(); O = a.o
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
side = []
for f in sorted(glob.glob(O + "/preds/*_preds.json")):
    m = json.load(open(f)); side.append({k: m.get(k) for k in ("set", "label", "checkpoint_label", "frame", "seed", "n_units", "device", "host", "utc", "ckpt_sha256", "checkpoint_sha256", "npz", "npz_sha256", "preds_file", "preds_sha256", "batch_size")})
k1 = sorted(glob.glob("$U/exp/CAND/K1/score/rows/test_ouhdl__LUO_ms6_*__seed*__rows031.csv")); k1 = [f for f in k1 if "_vt_" not in f]
code = {os.path.basename(f): sha(f) for f in sorted(glob.glob("$U/exp/RQF/code/rqf1_v1/*")) if os.path.isfile(f) and not f.endswith(".err")}
rows = {os.path.basename(f): sha(f) for f in sorted(glob.glob(O + "/score_out/rows/*__rows031.csv") + glob.glob(O + "/trace_score/rows/*__rows031.csv"))}
jobs = json.load(open(O + "/logs/R1_jobs.json")) if os.path.exists(O + "/logs/R1_jobs.json") else None
out = dict(kind="RQF R1 provenance", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prereg_sha256="04766e973badf7ff469d468e25bab17cebabf38f2c9537d731a4d91ea960c624",
  run_by="<agent> (sub-agent rqf-s1); model chosen by lead (<agent> lead) because the task only reruns existing sampling/scoring recipes with an existing frame transform",
  evaluator="uhwr_eval 0.3.1 (shared/uhwr_eval-0.3.1), lane_score.py rows031, native protocol", frame_code="exp/CAND/code/frame_code_a44e78b0 (CODE_TAG 5dd59e88)", rqf1_code_sha256=code,
  mirrored_sets_provenance=O + "/data/PROVENANCE_R1_mirrored_sets.json", mirrored_sets_provenance_sha256=sha(O + "/data/PROVENANCE_R1_mirrored_sets.json"),
  reused={"R1c LUO ms6 (E137 / CAND K1)": {os.path.basename(f): sha(f) for f in k1}, "R1c TRACE": "exp/TRACE/out/full/predictions_TRACE_ouhdl.pkl (deterministic; rescored with uhwr_eval 0.3.1, trace_rescore.py)"},
  sampling_files=side, rows_files_sha256=rows, jobs=jobs)
json.dump(out, open(O + "/PROVENANCE_R1.json", "w"), indent=1); print("wrote", len(side), "sidecars", len(rows), "rows files")
