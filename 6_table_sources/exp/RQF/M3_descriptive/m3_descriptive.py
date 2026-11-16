"""A1.3 (M3, descriptive). From existing per-unit rows (rows031 / InkSight rows CSV; protocol native):
share of paired units with |dAIoU| <= 0.01 and with direction disagreement >= 0.5 of strokes (|d direction_acc| >= 0.5, per-unit mean over seeds), and the joint share.
Pairs: P1 (LUO ms6 identity, LUO ms6 mirror_x; OUHD-L test, K1 rows, 3 seeds each); P2 (InkSight Small-p e0r rendered, E0 ms8; OUHD-L test, E0 3 seeds from D3 stageA);
P3 (E0 ms8 identity, LUO ms6; UNIPEN-de test, 3 seeds each). Units: intersection with defined AIoU and direction_acc in both arms. CI: writer-cluster percentile bootstrap B=2000 (seed 0).
Direction rule is the task's wording read as |d direction_acc| >= 0.5 (direction_acc = fraction of eligible strokes with correct direction); alt reading (arm B direction_acc lower by >= 0.5) also written."""
import csv, glob, hashlib, json, os, sys, time
import numpy as np
E = "$U/exp"
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def g(pat): return sorted(glob.glob(pat))
PAIRS = {
 "P1_LUO_identity_vs_mirror_x__ouhdl_test": (g(f"{E}/CAND/K1/score/rows/test_ouhdl__LUO_ms6_identity__seed*__rows031.csv"), g(f"{E}/CAND/K1/score/rows/test_ouhdl__LUO_ms6_mirror_x__seed*__rows031.csv")),
 "P2_InkSight_vs_E0__ouhdl_test": ([f"{E}/RB4_baselines/inksight_e0r/rows/inksight_e0r_rendered_rows.csv"], g(f"{E}/D3_ROUTE/stageA/score_out/rows/test_ouhdl__E0_ms8__seed*__rows031.csv")),
 "P3_E0_vs_LUO__unipen_de_test": (g(f"{E}/CAND/method_round/Frame/score_out/s2_e0_reverse/rows/german_unipen_de_test__E0_ms8_identity__seed*__rows031.csv"), g(f"{E}/D3_ROUTE/routeA/score_out/rows/german_unipen_de_test__LUO_ms6__seed*__rows031.csv")),
}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def f(x):
    try: return float(x)
    except: return np.nan
def load(files):
    acc = {}; wr = {}
    for p in files:
        for r in csv.DictReader(open(p)):
            if r["protocol"] != "native" or r.get("error", ""): continue
            acc.setdefault(r["unit_id"], []).append((f(r["aiou"]), f(r["direction_acc"]))); wr[r["unit_id"]] = r["writer_id"]
    out = {}
    for u, v in acc.items():
        a = np.array(v, float); out[u] = (np.nanmean(a[:, 0]) if np.isfinite(a[:, 0]).any() else np.nan, np.nanmean(a[:, 1]) if np.isfinite(a[:, 1]).any() else np.nan)
    return out, wr
def boot(flag, cl, B=2000):
    uc, inv = np.unique(cl, return_inverse=True); s = np.bincount(inv, weights=flag.astype(float)); n = np.bincount(inv).astype(float)
    idx = np.random.default_rng(0).integers(0, len(uc), size=(B, len(uc))); b = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
rows, prov = [], dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=sha(__file__), inputs={})
for name, (fa, fb) in PAIRS.items():
    assert fa and fb, name
    for p in fa + fb: prov["inputs"][p] = sha(p)
    A, wa = load(fa); B, wb = load(fb)
    us = sorted(u for u in A if u in B and np.isfinite(A[u][0]) and np.isfinite(B[u][0]) and np.isfinite(A[u][1]) and np.isfinite(B[u][1]))
    a = np.array([A[u] for u in us]); b = np.array([B[u] for u in us]); cl = np.array([wa[u] for u in us])
    d_aiou = b[:, 0] - a[:, 0]; d_dir = b[:, 1] - a[:, 1]
    flags = {"aiou_same(|dAIoU|<=0.01)": np.abs(d_aiou) <= 0.01, "dir_disagree(|d dir|>=0.5)": np.abs(d_dir) >= 0.5,
             "JOINT(aiou_same & dir_disagree)": (np.abs(d_aiou) <= 0.01) & (np.abs(d_dir) >= 0.5), "alt_JOINT(aiou_same & dir_B_lower>=0.5)": (np.abs(d_aiou) <= 0.01) & (d_dir <= -0.5),
             "alt_JOINT(aiou_same & dir_B_higher>=0.5)": (np.abs(d_aiou) <= 0.01) & (d_dir >= 0.5)}
    for k, v in flags.items():
        lo, hi = boot(v, cl)
        rows.append(dict(pair=name, quantity=k, n_paired_units=len(us), n_files_A=len(fa), n_files_B=len(fb), n_writers=len(set(cl)), count=int(v.sum()), share=float(v.mean()), ci_lo=lo, ci_hi=hi,
                         mean_aiou_A=float(a[:, 0].mean()), mean_aiou_B=float(b[:, 0].mean()), mean_dir_A=float(a[:, 1].mean()), mean_dir_B=float(b[:, 1].mean())))
w = csv.DictWriter(open(f"{OUT}/m3_descriptive.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(prov, open(f"{OUT}/m3_inputs_sha256.json", "w"), indent=1)
for r in rows: print(r["pair"][:8], r["quantity"][:40], r["n_paired_units"], r["count"], round(r["share"], 4), round(r["ci_lo"], 4), round(r["ci_hi"], 4), round(r["mean_aiou_A"], 3), round(r["mean_aiou_B"], 3), round(r["mean_dir_A"], 3), round(r["mean_dir_B"], 3))
