"""RQF R1 tables (prereg sec 1, sec 0 protocol): reads lane_score rows031 CSVs (uhwr_eval 0.3.1, native protocol), per-unit mean over
sampling seeds, paired differences on the unit intersection, percentile bootstrap B=2000 seed 0 over writer clusters (unit bootstrap when
<= 2 writers, i.e. OnHW), 95 %.  Applies R1a / R1b / R1c rules verbatim; writes CSVs + facts-only RESULTS_R1{a,b,c}.md.
python rqf1_tables.py --rows DIR [DIR..] --out OUTDIR"""
import argparse, glob, json, os, hashlib, time
import numpy as np, pandas as pd

B = 2000; METRICS = ["aiou", "direction_acc", "kendall_tau", "strict_order_acc", "penup_f1_v2"]
AIOU_M = 0.01035
R1A_BASES = {"LUO_ms6": "LUO ms6", "M5_ms9": "joint M5 ms9"}
R1A_SETS = ["iam500", "casia_en", "hpl_tamil"]
R1B_SETS = ["german_unipen_de_test", "test_iam500", "test_casia_en", "german_onhw_all"]

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

def load(dirs):
    fs = []
    for d in dirs: fs += sorted(glob.glob(os.path.join(d, "**", "*__rows031.csv"), recursive=True))
    recs = []
    for f in fs:
        df = pd.read_csv(f); df = df[(df.protocol == "native")]
        df = df[[c for c in ["set", "route_mode", "seed", "unit_id", "sample_id", "writer_id"] + METRICS if c in df.columns]]
        recs.append(df)
    D = pd.concat(recs, ignore_index=True)
    for m in METRICS:
        if m not in D.columns: D[m] = np.nan
    D["writer_id"] = D["writer_id"].astype(str)
    U = D.groupby(["set", "route_mode", "unit_id"], sort=False).agg(**{m: (m, "mean") for m in METRICS}, writer=("writer_id", "first"), n_seeds=("seed", "nunique")).reset_index()
    return U, fs

def boot(v, cl, unit_level=False):
    v = np.asarray(v, float); ok = ~np.isnan(v)
    if ok.sum() == 0: return np.nan, np.nan, np.nan, 0
    if unit_level: cl = np.arange(len(v))
    _, ci_ = np.unique(np.asarray(cl).astype(str), return_inverse=True); nc = ci_.max() + 1
    s = np.bincount(ci_[ok], weights=v[ok], minlength=nc); n = np.bincount(ci_[ok], minlength=nc).astype(float)
    idx = np.random.default_rng(0).integers(0, nc, (B, nc))
    den = n[idx].sum(1); m = np.where(den > 0, s[idx].sum(1) / np.maximum(den, 1), np.nan)
    return float(v[ok].mean()), float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5)), int(nc)

def ncl(U): return U.writer.nunique()

def absolute(U, st, rm):
    g = U[(U.set == st) & (U.route_mode == rm)]
    if g.empty: return None
    unit = ncl(g) <= 2
    r = dict(set=st, route_mode=rm, n_units=len(g), n_writers=ncl(g), n_seeds=int(g.n_seeds.max()), cluster="unit" if unit else "writer")
    for m in METRICS:
        mu, lo, hi, _ = boot(g[m].values, g.writer.values, unit); r[m] = mu; r[m + "_lo"] = lo; r[m + "_hi"] = hi
    return r

def paired(U, st_a, rm_a, st_b, rm_b, tag, base):
    a = U[(U.set == st_a) & (U.route_mode == rm_a)].set_index("unit_id"); b = U[(U.set == st_b) & (U.route_mode == rm_b)].set_index("unit_id")
    if a.empty or b.empty: return None
    com = a.index.intersection(b.index); w = a.loc[com, "writer"].values; unit = len(set(w)) <= 2
    r = dict(base=base, contrast=tag, set_a=st_a, arm_a=rm_a, set_b=st_b, arm_b=rm_b, n_units=len(com), n_writers=len(set(w)), cluster="unit" if unit else "writer")
    for m in METRICS:
        d = a.loc[com, m].values - b.loc[com, m].values
        mu, lo, hi, _ = boot(d, w, unit); r[m] = mu; r[m + "_lo"] = lo; r[m + "_hi"] = hi
    return r

def f(x): return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.4f}"
def fa(x): return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
def row_md(r, signed=True):
    return " | ".join(f"{(f if signed else fa)(r[m])} [{(f if signed else fa)(r[m+'_lo'])}, {(f if signed else fa)(r[m+'_hi'])}]" for m in METRICS)
HDR = "| " + " | ".join(["AIoU", "direction", "tau", "strict", "pen-up F1 v2"]) + " |"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); U, files = load(a.rows); utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    have = set(zip(U.set, U.route_mode))
    absr, pr, verdict = [], [], {}
    # ---------------- R1a
    md = [f"# RESULTS R1a (mirrored-LTR control; CONFIRMATORY)\n\nGenerated {utc}; rows files: {len(files)}; scorer uhwr_eval 0.3.1 native; per-unit mean over sampling seeds; writer-cluster percentile bootstrap B={B} seed 0, 95 %; paired on the unit intersection.\nColumns: AIoU | direction | tau | strict | pen-up F1 v2 (point [lo, hi]).\n"]
    for base, bn in R1A_BASES.items():
        col, rep = {}, {}
        md.append(f"\n## Base {bn}\n")
        for s in R1A_SETS:
            o, m_ = f"test_{s}" if s != "iam500" else "test_iam500", None
            o = "test_iam500" if s == "iam500" else f"test_{s}"; mset = o + "_mir"
            idn_o, idn_m, mx_m = f"{base}_identity", f"{base}_identity", f"{base}_mirror_x"
            for st, rm in ((o, idn_o), (mset, idn_m), (mset, mx_m)):
                r = absolute(U, st, rm)
                if r: absr.append(dict(r, base=base))
            c1 = paired(U, mset, idn_m, o, idn_o, "identity(mirrored) - identity(original)", base)
            c2 = paired(U, mset, mx_m, mset, idn_m, "mirror_x(mirrored) - identity(mirrored)", base)
            for r in (c1, c2):
                if r: pr.append(r)
            col[s] = None if c1 is None else bool(c1["direction_acc_hi"] < -0.10)
            rep[s] = None if c2 is None else bool(c2["direction_acc_lo"] > 0 and c2["aiou_lo"] > -AIOU_M)
            md.append(f"### {s}\n\n{HDR}\n|---|---|---|---|---|\n")
            for r in (c1, c2):
                md.append((f"| **{r['contrast']}** (n={r['n_units']}, writers={r['n_writers']}) " if r else "| NOT AVAILABLE ") + "\n" + ("| " + row_md(r) + " |\n" if r else ""))
            md.append(f"COLLAPSE clause (direction CI hi < -0.10): {col[s]}; REPAIR clause (direction CI lo > 0 AND AIoU CI lo > {-AIOU_M}): {rep[s]}\n")
        nC = sum(1 for v in col.values() if v); nR = sum(1 for v in rep.values() if v); avail = all(v is not None for v in col.values()) and all(v is not None for v in rep.values())
        verdict[base] = dict(collapse_sets=col, repair_sets=rep, collapse_n=nC, repair_n=nR, complete=avail, COLLAPSE=bool(nC >= 2) if avail else None, REPAIR=bool(nR >= 2) if avail else None)
        md.append(f"\nRule R1a, base {bn}: COLLAPSE met on {nC}/3 sets -> {verdict[base]['COLLAPSE']}; REPAIR met on {nR}/3 sets -> {verdict[base]['REPAIR']}; complete={avail}.\n")
    md.append("\nNote fixed in advance (prereg): mirror_x on a mirrored LTR set restores the original orientation, so REPAIR is partly a consistency check; COLLAPSE is the informative clause.\n")
    open(os.path.join(a.out, "RESULTS_R1a.md"), "w").write("".join(md))
    # ---------------- R1b
    md = [f"# RESULTS R1b (E0 ms8, identity vs mirror_x; CONFIRMATORY)\n\nGenerated {utc}; same protocol as R1a. OnHW (2 writers): unit bootstrap; UNIPEN-de has 4 writers (writer-cluster CI as prereg; very few clusters).\n\n{HDR}\n|---|---|---|---|---|\n"]
    rep = {}; base = "E0_ms8"
    for s in R1B_SETS:
        for rm in (f"{base}_identity", f"{base}_mirror_x"):
            r = absolute(U, s, rm)
            if r: absr.append(dict(r, base=base))
        c = paired(U, s, f"{base}_mirror_x", s, f"{base}_identity", "mirror_x - identity", base)
        if c: pr.append(c)
        rep[s] = None if c is None else bool(c["direction_acc_lo"] > 0 and c["aiou_lo"] > -AIOU_M)
        md.append(f"| **{s}** (n={c['n_units']}, writers={c['n_writers']}, {c['cluster']}-cluster) " if c else f"| {s} NOT AVAILABLE ")
        md.append("\n" + ("| " + row_md(c) + " |\n" if c else ""))
        md.append(f"REPAIR clause: {rep[s]}\n\n")
    nR = sum(1 for v in rep.values() if v); avail = all(v is not None for v in rep.values())
    verdict["R1b"] = dict(repair_sets=rep, repair_n=nR, complete=avail, REPAIR=bool(nR >= 3) if avail else None)
    md.append(f"Rule R1b: REPAIR met on {nR}/4 sets (need >= 3) -> {verdict['R1b']['REPAIR']}; complete={avail}.\n")
    open(os.path.join(a.out, "RESULTS_R1b.md"), "w").write("".join(md))
    # ---------------- R1c
    md = [f"# RESULTS R1c (Urdu zero-shot, identity vs mirror_x; descriptive + K1 restated)\n\nGenerated {utc}; OUHD-L test (4,406 units; TRACE: test253 lines), protocol as R1a. K1 per base: tau CI lo of the mirror_x arm >= 0.65 (restated, not a new claim for LUO).\n"]
    c_bases = [("LUO_ms6", "test_ouhdl"), ("NOFM_W_bb3", "test_ouhdl"), ("PEINIT_W_bb4", "test_ouhdl"), ("TRACE_uhwr_f4", "test253"), ("TRACE_trace_render", "test253")]
    verdict["R1c"] = {}
    for base, st in c_bases:
        arms = {fr: absolute(U, st, f"{base}_{fr}") for fr in ("identity", "mirror_x")}
        for r in arms.values():
            if r: absr.append(dict(r, base=base))
        c = paired(U, st, f"{base}_mirror_x", st, f"{base}_identity", "mirror_x - identity", base)
        if c: pr.append(c)
        md.append(f"\n## {base} ({st})\n\n{HDR}\n|---|---|---|---|---|\n")
        for fr, r in arms.items(): md.append(f"| {fr} absolute (n={r['n_units']}, {r['cluster']}-cluster) |\n| {row_md(r, False)} |\n" if r else f"| {fr} NOT AVAILABLE |\n")
        md.append((f"| **mirror_x - identity** (n={c['n_units']}) |\n| {row_md(c)} |\n" if c else "| paired NOT AVAILABLE |\n"))
        k1 = None if not arms["mirror_x"] else bool(arms["mirror_x"]["kendall_tau_lo"] >= 0.65)
        verdict["R1c"][base] = dict(K1_tau_lo_ge_0_65=k1, tau_lo=None if not arms["mirror_x"] else arms["mirror_x"]["kendall_tau_lo"])
        md.append(f"\nK1 restated (mirror_x tau CI lo >= 0.65): {k1}\n")
    md.append("\nTRACE is deterministic (no sampling seed); n_seeds = 1.\n")
    open(os.path.join(a.out, "RESULTS_R1c.md"), "w").write("".join(md))
    pd.DataFrame(absr).to_csv(os.path.join(a.out, "R1_absolute.csv"), index=False); pd.DataFrame(pr).to_csv(os.path.join(a.out, "R1_paired.csv"), index=False)
    json.dump(dict(utc=utc, verdicts=verdict, n_rows_files=len(files), rows_files_sha256_of_list=hashlib.sha256("\n".join(files).encode()).hexdigest(), B=B, margins=dict(aiou=AIOU_M)), open(os.path.join(a.out, "R1_verdicts.json"), "w"), indent=1)
    print(json.dumps(verdict)[:1500])

if __name__ == "__main__": main()
