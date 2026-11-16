"""CAND K1 readout (exp/CAND/K1): per-frame means + writer-cluster CIs, paired per-unit seed-averaged deltas vs the identity
frame, stroke-class split of direction / loop-orientation accuracy, and the K1 kill rule of prereg_candidates_2026-10-28.md
sec. 5, evaluated mechanically (numbers + rule outcome only).

Aggregation = routeA_tables.py (exp/D3_ROUTE/routeA/code): native protocol rows (uhwr_eval 0.3.1, lane_score.py rows031);
per-unit mean over seeds first (NaN skipped); point = mean over units with a defined value; tau_cp = tau with undefined
tau := 0 on units with n_gt_strokes >= 2; CI = writer-cluster percentile bootstrap, multinomial writer counts, ratio of
resampled sums, B = 2000, numpy default_rng(20261028) (prereg sec. 6 bootstrap seed).  Paired deltas: inner join on unit_id,
both sides defined.  Stroke-class split: stroke-level pooled accuracy = sum(ok)/sum(eligible) over all seeds, writer-cluster
bootstrap with the same counts matrix for both frames of a pair (paired difference of ratios).
python k1_summary_cand.py --rows R --strokes S --out O [--boot 2000]
"""
import argparse, glob, json, os, re, hashlib, time
import numpy as np, pandas as pd

METRICS = ["aiou", "ldtw", "dtw", "kendall_tau", "tau_cp", "direction_acc", "strict_order_acc", "penup_f1_v2", "loop_orient_acc", "stroke_count_abs_err"]
SEED = 20261028
REF = dict(aiou=0.7117, kendall_tau=0.4799, direction_acc=0.5468)   # LUO ms6 zero-shot ouhdl_test (prereg sec. 2)
KILL = dict(direction_min=0.70, tau_min=0.65, aiou_drop_max=0.02)
FRAMES = ["identity", "mirror_x", "vt"]
CLASSES = ["horizontal", "vertical", "chirality", "lt3pts", "all_ge3pts", "all"]

def boot_counts(n, B, seed=SEED):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)

def stat(v, C, wix):
    v = np.asarray(v, float); ok = ~np.isnan(v); n = int(ok.sum())
    if n == 0: return np.nan, np.nan, np.nan, 0
    m = float(v[ok].mean())
    if C is None: return m, np.nan, np.nan, n
    S = np.bincount(wix[ok], weights=v[ok], minlength=C.shape[1]); N = np.bincount(wix[ok], minlength=C.shape[1]).astype(float)
    num, den = C @ S, C @ N; bm = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(bm, [2.5, 97.5]); return m, float(lo), float(hi), n

def frame_of(rm):
    m = re.search(r"_(identity|mirror_x|vt)$", rm); return m.group(1) if m else None

def per_unit(g, cluster):
    cols = [c for c in METRICS if c in g.columns and c != "tau_cp"] + ["n_gt_strokes"]
    agg = g.groupby("unit_id", sort=True)[cols].mean(numeric_only=True)
    agg["_cl"] = g.groupby("unit_id", sort=True)[cluster].first().astype(str)
    agg["n_seeds"] = g.groupby("unit_id", sort=True)["seed"].nunique()
    t = agg["kendall_tau"].to_numpy(float)
    agg["tau_cp"] = np.where((agg["n_gt_strokes"] >= 2).to_numpy(), np.nan_to_num(t, nan=0.0), np.nan)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--strokes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.rows, "*__rows031.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False).assign(rows_file=os.path.basename(f)) for f in files], ignore_index=True)
    df["unit_id"] = df["unit_id"].astype(str); df["error"] = df["error"].fillna("")
    df["frame"] = df["route_mode"].map(frame_of)
    nat = df[df.protocol == "native"].copy()
    inputs = nat.groupby(["set", "frame", "seed", "preds_file", "preds_sha256"]).agg(n_units=("unit_id", "size"), n_error=("error", lambda e: int((e != "").sum()))).reset_index()
    inputs.to_csv(os.path.join(a.out, "k1_inputs.csv"), index=False)
    per_frame, paired, seedtab = [], [], []
    aggs = {}
    for clname in ("writer_id", "sample_id"):
        for (s, fr), g in nat.groupby(["set", "frame"]):
            agg = per_unit(g, clname); aggs[(clname, s, fr)] = agg
            uw, wix = np.unique(agg["_cl"].to_numpy(), return_inverse=True)
            C = boot_counts(len(uw), a.boot) if len(uw) > 1 else None
            for met in METRICS:
                if met not in agg.columns: continue
                m, lo, hi, n = stat(agg[met].to_numpy(float), C, wix)
                if met == "tau_cp":
                    sm = g.assign(_t=np.where(g["n_gt_strokes"] >= 2, g["kendall_tau"].fillna(0.0), np.nan)).groupby("seed")["_t"].mean()
                else:
                    sm = g.groupby("seed")[met].mean()
                per_frame.append(dict(set=s, frame=fr, cluster=clname, metric=met, mean=m, ci_lo=lo, ci_hi=hi, n_units=len(agg), n_defined=n,
                                      n_clusters=len(uw), n_seeds=int(agg["n_seeds"].max()), seed_means=";".join(f"{k}:{v:.4f}" for k, v in sm.items()),
                                      seed_sd=float(sm.std(ddof=1)) if len(sm) > 1 else np.nan))
        for s in sorted(nat.set.unique()):
            if (clname, s, "identity") not in aggs: continue
            b = aggs[(clname, s, "identity")]
            for fr in ("mirror_x", "vt"):
                if (clname, s, fr) not in aggs: continue
                A_ = aggs[(clname, s, fr)]
                for met in METRICS:
                    if met not in A_.columns: continue
                    j = A_[[met, "_cl"]].join(b[[met]], how="inner", rsuffix="_ref").dropna(subset=[met, met + "_ref"])
                    d = (j[met] - j[met + "_ref"]).to_numpy(float)
                    uw, wix = np.unique(j["_cl"].to_numpy(), return_inverse=True)
                    m, lo, hi, n = stat(d, boot_counts(len(uw), a.boot) if len(uw) > 1 else None, wix)
                    _, ulo, uhi, _ = stat(d, boot_counts(len(d), a.boot), np.arange(len(d)))
                    paired.append(dict(set=s, frame=fr, comparator="identity", cluster=clname, metric=met, delta=m, ci_lo=lo, ci_hi=hi, n_pairs=n,
                                       n_clusters=len(uw), ci_lo_unit=ulo, ci_hi_unit=uhi, ci_excludes_0=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                                       frac_pos=float((d > 0).mean()), frac_neg=float((d < 0).mean())))
    PF = pd.DataFrame(per_frame); PD = pd.DataFrame(paired)
    PF.to_csv(os.path.join(a.out, "k1_per_frame.csv"), index=False); PD.to_csv(os.path.join(a.out, "k1_paired_vs_identity.csv"), index=False)
    # ---- stroke-class split
    sfiles = sorted(glob.glob(os.path.join(a.strokes, "*__strokes.csv.gz")))
    st = pd.concat([pd.read_csv(f, dtype={"unit_id": str, "writer_id": str, "sample_id": str}) for f in sfiles], ignore_index=True)
    # consistency vs rows031: per (file, unit) direction_acc / loop_orient_acc / n_dir_eligible
    chk = st.groupby(["preds_file", "unit_id"]).agg(n_de=("dir_elig", "sum"), n_dk=("dir_ok", "sum"), n_le=("loop_elig", "sum"), n_lk=("loop_ok", "sum")).reset_index()
    rr = nat[["preds_file", "unit_id", "direction_acc", "n_dir_eligible", "loop_orient_acc", "n_loop_eligible"]]
    cj = chk.merge(rr, on=["preds_file", "unit_id"], how="inner")
    de, dk, le_, lk = (cj[c].to_numpy(float) for c in ("n_de", "n_dk", "n_le", "n_lk"))
    dacc = np.where(de > 0, dk / np.maximum(de, 1), np.nan); lacc = np.where(le_ > 0, lk / np.maximum(le_, 1), np.nan)
    consistency = dict(n_units_joined=int(len(cj)), n_units_strokes=int(len(chk)),
                       n_dir_eligible_mismatch=int((de != cj.n_dir_eligible.to_numpy(float)).sum()), n_loop_eligible_mismatch=int((le_ != cj.n_loop_eligible.to_numpy(float)).sum()),
                       max_abs_diff_direction_acc=float(np.nanmax(np.abs(dacc - cj.direction_acc.to_numpy(float)))) if len(cj) else None,
                       n_direction_nan_mismatch=int((np.isnan(dacc) != cj.direction_acc.isna().to_numpy()).sum()),
                       max_abs_diff_loop_orient_acc=float(np.nanmax(np.abs(lacc - cj.loop_orient_acc.to_numpy(float)))) if len(cj) else None)
    sc_rows, sc_pair, shares = [], [], []
    for s, gs in st.groupby("set"):
        u0 = gs[(gs.preds_file == gs.preds_file.iloc[0])]
        vc = u0.cls.value_counts(); ge3 = u0[u0.cls != "lt3pts"]
        shares.append(dict(set=s, preds_file_used=u0.preds_file.iloc[0], n_gt_strokes=len(u0), n_ge3pts=len(ge3),
                           **{f"n_{c}": int(vc.get(c, 0)) for c in ["horizontal", "vertical", "chirality", "lt3pts"]},
                           **{f"share_ge3_{c}": float((ge3.cls == c).mean()) for c in ["horizontal", "vertical", "chirality"]}))
        uw, wix_all = np.unique(gs["writer_id"].to_numpy(), return_inverse=True); gs = gs.assign(_w=wix_all)
        C = boot_counts(len(uw), a.boot) if len(uw) > 1 else None
        sums = {}
        for fr, gf in gs.groupby("frame"):
            for c in CLASSES:
                gc = gf if c == "all" else (gf[gf.cls != "lt3pts"] if c == "all_ge3pts" else gf[gf.cls == c])
                rec = dict(set=s, frame=fr, cls=c, n_strokes=len(gc), n_seeds=int(gf.seed.nunique()))
                for nm, el, ok in (("direction", "dir_elig", "dir_ok"), ("loop_orient", "loop_elig", "loop_ok"), ("match", None, "matched")):
                    e = np.ones(len(gc)) if el is None else gc[el].to_numpy(float); k = gc[ok].to_numpy(float)
                    Se = np.bincount(gc["_w"], weights=e, minlength=len(uw)); Sk = np.bincount(gc["_w"], weights=k, minlength=len(uw))
                    sums[(fr, c, nm)] = (Sk, Se)
                    m = Sk.sum() / Se.sum() if Se.sum() > 0 else np.nan
                    if C is not None and Se.sum() > 0:
                        den = C @ Se; bm = (C @ Sk)[den > 0] / den[den > 0]; lo, hi = np.percentile(bm, [2.5, 97.5])
                    else: lo = hi = np.nan
                    rec.update({f"{nm}_acc": m, f"{nm}_lo": lo, f"{nm}_hi": hi, f"{nm}_n_eligible_all_seeds": int(Se.sum())})
                rec["n_writers"] = len(uw); sc_rows.append(rec)
        for fr, ref in (("mirror_x", "identity"), ("vt", "identity"), ("mirror_x", "vt")):
            for c in CLASSES:
                for nm in ("direction", "loop_orient"):
                    if (fr, c, nm) not in sums or (ref, c, nm) not in sums: continue
                    Ka, Ea = sums[(fr, c, nm)]; Kb, Eb = sums[(ref, c, nm)]
                    if Ea.sum() == 0 or Eb.sum() == 0: continue
                    d = Ka.sum() / Ea.sum() - Kb.sum() / Eb.sum()
                    if C is not None:
                        da, db = C @ Ea, C @ Eb; okb = (da > 0) & (db > 0)
                        bd = (C @ Ka)[okb] / da[okb] - (C @ Kb)[okb] / db[okb]; lo, hi = np.percentile(bd, [2.5, 97.5])
                    else: lo = hi = np.nan
                    sc_pair.append(dict(set=s, frame=fr, comparator=ref, cls=c, metric=nm + "_acc", delta=d, ci_lo=lo, ci_hi=hi,
                                        ci_excludes_0=bool(np.isfinite(lo) and (lo > 0 or hi < 0)), n_writers=len(uw)))
    SC = pd.DataFrame(sc_rows); SP = pd.DataFrame(sc_pair); SH = pd.DataFrame(shares)
    SC.to_csv(os.path.join(a.out, "k1_strokeclass.csv"), index=False); SP.to_csv(os.path.join(a.out, "k1_strokeclass_paired.csv"), index=False)
    SH.to_csv(os.path.join(a.out, "k1_strokeclass_shares.csv"), index=False)
    # ---- K1 rule on ouhdl_test (writer cluster)
    T = PF[(PF.set == "test_ouhdl") & (PF.cluster == "writer_id")].set_index(["frame", "metric"])
    P_ = PD[(PD.set == "test_ouhdl") & (PD.cluster == "writer_id")].set_index(["frame", "metric"])
    k1 = {}
    for fr in ("mirror_x", "vt"):
        if (fr, "aiou") not in T.index: continue
        g = lambda met, col: float(T.loc[(fr, met), col])
        dm, dl = g("direction_acc", "mean"), g("direction_acc", "ci_lo"); tm, tl = g("kendall_tau", "mean"), g("kendall_tau", "ci_lo")
        am = g("aiou", "mean")
        c = dict(direction_mean=dm, direction_ci=[dl, g("direction_acc", "ci_hi")], tau_mean=tm, tau_ci=[tl, g("kendall_tau", "ci_hi")],
                 aiou_mean=am, aiou_ci=[g("aiou", "ci_lo"), g("aiou", "ci_hi")],
                 direction_ge_070=dm >= KILL["direction_min"], direction_ci_excludes_ref=dl > REF["direction_acc"],
                 tau_ge_065=tm >= KILL["tau_min"], tau_ci_excludes_ref=tl > REF["kendall_tau"],
                 aiou_within_002=am >= REF["aiou"] - KILL["aiou_drop_max"],
                 paired_direction_ci_excludes_0=bool(P_.loc[(fr, "direction_acc"), "ci_excludes_0"]) and float(P_.loc[(fr, "direction_acc"), "delta"]) > 0,
                 paired_tau_ci_excludes_0=bool(P_.loc[(fr, "kendall_tau"), "ci_excludes_0"]) and float(P_.loc[(fr, "kendall_tau"), "delta"]) > 0)
        dir_ok = c["direction_ge_070"] and c["direction_ci_excludes_ref"]; tau_ok = c["tau_ge_065"] and c["tau_ci_excludes_ref"]
        c["pass_reading_AND"] = bool(dir_ok and tau_ok and c["aiou_within_002"])      # all three conditions (frame_k1_summary.py reading)
        c["pass_reading_OR"] = bool((dir_ok or tau_ok) and c["aiou_within_002"])      # literal 'does not reach direction>=0.70 or tau>=0.65'
        k1[fr] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in c.items()}
    def select(reading):
        ps = [f for f in k1 if k1[f][reading]]
        if not ps: return None
        ps.sort(key=lambda f: -k1[f]["tau_mean"])
        if len(ps) == 2 and abs(k1[ps[0]]["tau_mean"] - k1[ps[1]]["tau_mean"]) < 0.01:
            ps.sort(key=lambda f: (-k1[f]["direction_mean"], -k1[f]["aiou_mean"]))
        return ps[0]
    idn = {met: float(T.loc[("identity", met), "mean"]) for met in ("aiou", "kendall_tau", "direction_acc")} if ("identity", "aiou") in T.index else {}
    out = dict(rule=dict(text="K1 kill if best frame does not reach direction >= 0.70 or tau >= 0.65 (identity 0.547/0.480), or AIoU drops > 0.02 below 0.7117 (prereg sec. 5); CI must exclude identity values 0.5468/0.4799 (track brief)",
                         refs=REF, thresholds=KILL, set="test_ouhdl", cluster="writer_id", boot=a.boot, seed=SEED),
               identity_this_run=idn, per_frame=k1, winner_reading_AND=select("pass_reading_AND"), winner_reading_OR=select("pass_reading_OR"),
               stroke_consistency_vs_rows031=consistency, n_rows_files=len(files), n_stroke_files=len(sfiles), seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(a.out, "k1_verdict.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

if __name__ == "__main__":
    main()
