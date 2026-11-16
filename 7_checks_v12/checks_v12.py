#!/usr/bin/env python3
"""InkBridge v12 review checks on EXISTING cluster outputs (read-only inputs; no training, no sampling).
python checks_v12.py OUT [--smoke] [--procs N]
A randctl : selector experiment (29 specs of exp/RQF/R3/read_all3 + the ORDER sealed FT1% Urdu spec). Per cell and metric (the 8 metrics of
            loop_tables.MET): selected (rep3) minus mean of its three candidates (pool3) and random pick (rnd3) minus the same mean, paired per unit,
            writer-cluster bootstrap of loop_tables.boot_idx (B=2000, rng 20261029, percentile), as in sealed_read.py. Integrity: same units, same model
            stem, same GT file, same evaluator, sampling sidecars, and whether each selected row equals one of the three candidate rows.
B inkinv  : uhwr_eval 0.3.1 score() on ground-truth inks: GT vs GT with strokes reversed / reordered; step diagnostics (points, integer pixels, masks).
C n4358   : LUO_ms6 identity vs mirror_x on OUHD-L test (exp/CAND/K1/score/rows): units with direction defined in both arms.
D ft1abs  : plain FT1% row for tab:abs with the functions of exp/RQF/code/rqf_paired_ext.py (copied verbatim: boot_counts, boot, load, peritem).
E screen  : status counts of the foundation-model screen verdict files."""
import sys, os, json, glob, hashlib, collections, tarfile, time, traceback, io
import numpy as np, pandas as pd
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; B = 100 if SMOKE else 2000
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
U = "$U"; E = U + "/exp/"; Q = E + "RQF"; E0C = U + "/e0/code_4cd40f57"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, Q + "/code/rq3_v1")
from loop_tables import load_cell, unit_table, boot_idx, MET
import loop_tables as LT
_sha = {}
def sha(p):
    if p not in _sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
        _sha[p] = h.hexdigest()
    return _sha[p]
PROV = dict(script="checks_v12.py", script_sha256=sha(os.path.abspath(__file__)), B=B, smoke=SMOKE, boot_seed_randctl=20261029, boot_seed_ft1abs=0,
            host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), utc_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            code={p: sha(p) for p in [LT.__file__, Q + "/code/rq3_v1/sealed_read.py", Q + "/code/rq3_v1/rq3_summarize.py", Q + "/code/rqf_paired_ext.py",
                                      E + "ORDER/code/loop_tables.py"] if os.path.exists(p)}, inputs={}, status={})
FACTS = {}
def inp(p): PROV["inputs"][p] = sha(p); return p
def part(name, fn):
    t = time.time()
    try: fn(); PROV["status"][name] = "ok %.0fs" % (time.time() - t)
    except Exception: PROV["status"][name] = "FAILED " + traceback.format_exc()[-1800:]
    print(name, PROV["status"][name], flush=True)
_rows = {}
def rows(f):
    if f not in _rows:
        d = pd.read_csv(inp(f), low_memory=False); d = d[d.protocol == "native"].copy()
        d["tau_cp"] = np.where(d.n_gt_strokes >= 2, d.kendall_tau.fillna(0.0), np.nan); d["unit_id"] = d.unit_id.astype(str)
        _rows[f] = d.set_index("unit_id")
    return _rows[f]
# ---------------------------------------------------------------- A
def contrast(um, uc, m):
    j = um[[m, "writer_id"]].join(uc[[m]], rsuffix="_c", how="inner").dropna()      # same join as sealed_read.pboot
    d = (j[m] - j[m + "_c"]).to_numpy(); w = j.writer_id.to_numpy()
    bs = np.array([d[i].mean() for i in boot_idx(w, B)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); blo, bhi = np.percentile(bs, [100 * 0.05 / 16, 100 - 100 * 0.05 / 16])
    return dict(n=len(j), n_writers=len(np.unique(w)), mean=j[m].mean(), mean3=j[m + "_c"].mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi,
                bonf8_lo=blo, bonf8_hi=bhi, p_boot=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
def flat(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flat(v, pre + str(k) + "."))
    else: out[pre[:-1]] = o if not isinstance(o, list) else json.dumps(o)[:200]
    return out
SKIP = ("seed", "sha", "time", "utc", "file", "path", "host", "job", "npz", "elapsed", "sec", "date", "out", "dir", "preds", "label", "n_units", "rate", "pid")
def settings(p):
    if not os.path.exists(p): return None
    return {k: v for k, v in flat(json.load(open(inp(p)))).items() if not any(s in k.lower() for s in SKIP)}
CMP = ["aiou", "ldtw", "dtw", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "n_pred_pts", "n_pred_strokes", "chamfer"]
def randctl():
    plan = json.load(open(inp(Q + "/R3/plan.json"))); PU = {}
    for u in plan["units"]:
        fb = "COMP_p01" if u["base"].startswith("COMP_p01") else u["base"]; PU[(fb, u["set"], str(u["train_idx"]))] = u
    specs = [(os.path.basename(s)[5:-5].split("__", 1), s, Q + "/R3/read_all3") for s in sorted(glob.glob(Q + "/R3/read_all3/spec_*.json"))]
    specs.append((["COMP_p01", "test_ouhdl"], E + "ORDER/sealed/COMP_rep3/read/spec_test_ouhdl.json", E + "ORDER/sealed/COMP_rep3/read"))
    if SMOKE: specs = specs[:2] + specs[-1:]
    summ = pd.read_csv(inp(Q + "/R3/R3_summary_3seed_cells.csv")).set_index(["base", "set"])
    l08 = pd.read_csv(inp(E + "RV10/tables/facts_v11_v2/L08_control_per_cell.csv")).set_index(["base", "set"])
    R, CELLS, example = [], [], None
    for (base, st), sp, rd in specs:
        S = json.load(open(inp(sp))); lab = os.path.basename(sp)[5:-5]; src = "ORDER_sealed" if "ORDER" in sp else "R3_read_all3"
        Pf, Rf, Nf = (S["cells"][k]["files"] for k in ("COMP_pool3", "COMP_rep3", "COMP_rnd3"))
        U3 = {k: unit_table(load_cell(S["cells"][k])) for k in ("COMP_pool3", "COMP_rep3", "COMP_rnd3")}
        by = collections.defaultdict(dict)
        for k in sorted(Pf): by[k[:-1]][int(k[-1])] = k
        ig = dict(keys_equal=bool(set(Pf) == set(Rf) == set(Nf)), n_train=len(by), n_keys=len(Pf)); acc = collections.Counter(); strict = {m: [] for m in MET}; sdiff = set(); sfound = 0
        for t, ks in by.items():
            pool = [rows(Pf[ks[j]]) for j in range(3)]; rep = rows(Rf[ks[0]]); rnd = rows(Nf[ks[0]])
            acc["rep_file_same_for_3_keys"] += int(len({Rf[k] for k in ks.values()}) == 1)
            b0 = os.path.basename(Pf[ks[0]]); acc["same_model_stem"] += int(os.path.basename(Rf[ks[0]]) == b0.replace("__seed0__rows031.csv", "_rep3__seed0__rows031.csv")
                                                                         and all(os.path.basename(Pf[ks[j]]) == b0.replace("__seed0__", "__seed%d__" % j) for j in range(3)))
            acc["same_unit_set"] += int(all(set(rep.index) == set(p.index) for p in pool) and set(rnd.index) == set(rep.index) and rep.index.is_unique)
            allr = pool + [rep, rnd]
            acc["same_gt_sha"] += int(len({x for d in allr for x in d.gt_sha256.astype(str).unique()}) == 1)
            acc["same_evaluator"] += int(len({x for d in allr for x in d.evaluator.astype(str).unique()}) == 1)
            acc["same_specialist"] += int(len({x for d in allr for x in d.specialist.astype(str).unique()}) == 1)
            a = rep[CMP].to_numpy(float); eq = np.stack([np.all(np.isclose(a, p.loc[rep.index, CMP].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True), axis=1) for p in pool], 1)
            nm = eq.sum(1); acc["sel_match_none"] += int((nm == 0).sum()); acc["sel_match_one"] += int((nm == 1).sum()); acc["sel_match_multi"] += int((nm > 1).sum()); acc["sel_rows"] += len(nm)
            for j in range(3): acc["sel_is_seed%d_unique" % j] += int((eq[:, j] & (nm == 1)).sum())
            for m in MET:
                pm = np.stack([p.loc[rep.index, m].to_numpy(float) for p in pool], 1).mean(1); strict[m].append(rep[m].to_numpy(float) - pm)
            u = PU.get((base, st, t)) if src == "R3_read_all3" else None
            if u is not None:
                ss = [settings(p[:-4] + ".json") for p in u["pool"]]
                if all(s is not None for s in ss):
                    sfound += 1; sdiff |= {k for k in set().union(*ss) if len({json.dumps(s.get(k), default=str) for s in ss}) > 1}
                    if example is None: example = dict(cell=lab, train=t, settings_seed0={k: ss[0][k] for k in list(ss[0])[:60]})
        ig.update({k: int(v) for k, v in acc.items()}); ig["sidecars_found_train"] = sfound; ig["sampling_sidecar_keys_differing_across_3_candidates"] = ";".join(sorted(sdiff))
        stored = pd.read_csv(inp(f"{rd}/paired_{lab}.csv")); cj = json.load(open(inp(f"{rd}/cells_{lab}.json")))["cells"]
        sm = summ.loc[(base, st)] if (base, st) in summ.index and src == "R3_read_all3" else None; l8 = l08.loc[(base, st)] if (base, st) in l08.index and src == "R3_read_all3" else None
        cell = dict(source=src, base=base, set=st, urdu=bool(st == "test_ouhdl"), spec=sp, order_win_stored=cj["COMP_rep3__vs__COMP_pool3"]["order_win"],
                    part1_metrics=",".join(cj["COMP_rep3__vs__COMP_pool3"]["part1_metrics"]), p_holm_stored=None if sm is None else float(sm.p_holm),
                    flag_all8_ci_contain0_stored=None if sm is None else bool(sm.control_all_ci_contain0), order_win_and_holm_and_flag_stored=None if sm is None else bool(sm.order_win_and_holm), **ig)
        if l8 is not None:
            for k in l8.index:
                if k.startswith("randomization") or k in ("min_randomization_p", "stored_rule_fail"): cell["L08_" + k] = l8[k]
        nex = nexb = rex = rexb = 0
        for m in MET:
            a = contrast(U3["COMP_rep3"], U3["COMP_pool3"], m); r = contrast(U3["COMP_rnd3"], U3["COMP_pool3"], m)
            sd = np.nanmean(np.stack(strict[m], 0), 0) if False else np.stack(strict[m], 0).mean(0)
            row = dict(check="randctl", source=src, base=base, set=st, metric=m, n_units=a["n"], n_writers=a["n_writers"], mean_selected=a["mean"], mean_of_3_candidates=a["mean3"])
            for pre, x in (("sel_minus_mean3", a), ("rnd_minus_mean3", r)):
                row.update({f"{pre}_delta": x["delta"], f"{pre}_ci_lo": x["ci_lo"], f"{pre}_ci_hi": x["ci_hi"], f"{pre}_bonf8_lo": x["bonf8_lo"], f"{pre}_bonf8_hi": x["bonf8_hi"], f"{pre}_p_boot": x["p_boot"],
                            f"{pre}_ci_excl0": bool(not (x["ci_lo"] <= 0 <= x["ci_hi"])), f"{pre}_bonf8_excl0": bool(not (x["bonf8_lo"] <= 0 <= x["bonf8_hi"]))})
            row.update(sel_minus_mean3_strict_delta=float(np.nanmean(sd)), sel_minus_mean3_strict_n=int(np.isfinite(sd).sum()))
            for pre, mth in (("sel_minus_mean3", "COMP_rep3"), ("rnd_minus_mean3", "COMP_rnd3")):
                q = stored[(stored.method == mth) & (stored.comparator == "COMP_pool3") & (stored.metric == m)]
                if len(q): row.update({f"{pre}_stored_delta": q.delta.iloc[0], f"{pre}_stored_ci_lo": q.ci_lo.iloc[0], f"{pre}_stored_ci_hi": q.ci_hi.iloc[0], f"{pre}_stored_n": int(q.n.iloc[0])})
            R.append(row); nex += row["sel_minus_mean3_ci_excl0"]; nexb += row["sel_minus_mean3_bonf8_excl0"]; rex += row["rnd_minus_mean3_ci_excl0"]; rexb += row["rnd_minus_mean3_bonf8_excl0"]
        cell.update(sel_n_metrics_ci_excl0=int(nex), sel_n_metrics_bonf8_excl0=int(nexb), rnd_n_metrics_ci_excl0=int(rex), rnd_n_metrics_bonf8_excl0=int(rexb),
                    flag_all8_ci_contain0_recomputed=bool(rex == 0), flag_all8_bonf8_ci_contain0_recomputed=bool(rexb == 0))
        CELLS.append(cell); print("A", src, lab, "n", R[-1]["n_units"], "sel_none", ig.get("sel_match_none"), flush=True)
    D = pd.DataFrame(R); C = pd.DataFrame(CELLS)
    D = D.merge(C[["source", "base", "set", "urdu", "order_win_stored", "p_holm_stored", "flag_all8_ci_contain0_stored", "flag_all8_ci_contain0_recomputed", "flag_all8_bonf8_ci_contain0_recomputed"]
                  + [c for c in C.columns if c.startswith("L08_randomization")]], on=["source", "base", "set"])
    D.to_csv(OUT + "/checks_v12_results.csv", index=False); C.to_csv(OUT + "/checks_v12_randctl_cells.csv", index=False)
    N = C[(~C.urdu) & (C.source == "R3_read_all3")].copy(); lab = lambda d: sorted((d.base + "|" + d.set).tolist())
    rc = [c for c in C.columns if c.startswith("L08_randomization")]; f = {}
    N["gain"] = N.order_win_stored.astype(bool); N["gain_holm"] = N.gain & (N.p_holm_stored < 0.05)
    f["n_cells_non_urdu"] = len(N); f["n_gain_rule(order_win)"] = int(N.gain.sum()); f["n_gain_rule_and_family_holm"] = int(N.gain_holm.sum())
    f["n_flag_pass_stored"] = int(N.flag_all8_ci_contain0_stored.astype(bool).sum()); f["n_flag_pass_recomputed"] = int(N.flag_all8_ci_contain0_recomputed.sum())
    f["n_flag_pass_bonf8_recomputed"] = int(N.flag_all8_bonf8_ci_contain0_recomputed.sum())
    f["n_full_with_flag_stored(order_win&holm&flag)"] = int(N.order_win_and_holm_and_flag_stored.astype(bool).sum())
    f["n_full_with_flag_bonf8"] = int((N.gain_holm & N.flag_all8_bonf8_ci_contain0_recomputed).sum())
    for cname in rc:
        v = N[cname].astype(str).str.lower().eq("true"); f[f"n_{cname}"] = int(v.sum()); f[f"n_full_with_{cname}"] = int((N.gain_holm & v).sum())
        f[f"cells_gain_holm_but_fail_{cname}"] = lab(N[N.gain_holm & ~v])
    f["cells_gain_holm"] = lab(N[N.gain_holm]); f["cells_full_with_flag"] = lab(N[N.order_win_and_holm_and_flag_stored.astype(bool)])
    f["cells_gain_holm_but_flag_fail"] = lab(N[N.gain_holm & ~N.flag_all8_ci_contain0_stored.astype(bool)])
    f["cells_flag_fail_stored"] = lab(N[~N.flag_all8_ci_contain0_stored.astype(bool)]); f["cells_flag_fail_bonf8"] = lab(N[~N.flag_all8_bonf8_ci_contain0_recomputed])
    f["rnd_minus_mean3_n_cellmetrics"] = int(len(D[(~D.urdu) & (D.source == "R3_read_all3")])); dn = D[(~D.urdu) & (D.source == "R3_read_all3")]
    f["rnd_minus_mean3_n_cellmetrics_ci_excl0"] = int(dn.rnd_minus_mean3_ci_excl0.sum()); f["rnd_minus_mean3_n_cellmetrics_bonf8_excl0"] = int(dn.rnd_minus_mean3_bonf8_excl0.sum())
    f["rnd_minus_mean3_max_abs_delta"] = float(dn.rnd_minus_mean3_delta.abs().max())
    f["sel_minus_mean3_per_metric_non_urdu"] = {m: dict(n_cells=int(len(g)), ci_above0=int((g.sel_minus_mean3_ci_lo > 0).sum()), ci_below0=int((g.sel_minus_mean3_ci_hi < 0).sum()),
                                                      bonf8_above0=int((g.sel_minus_mean3_bonf8_lo > 0).sum()), bonf8_below0=int((g.sel_minus_mean3_bonf8_hi < 0).sum()),
                                                      min_delta=float(g.sel_minus_mean3_delta.min()), max_delta=float(g.sel_minus_mean3_delta.max())) for m, g in dn.groupby("metric")}
    du = D[D.urdu]
    f["sel_minus_mean3_per_metric_urdu"] = {m: dict(n_cells=int(len(g)), ci_above0=int((g.sel_minus_mean3_ci_lo > 0).sum()), ci_below0=int((g.sel_minus_mean3_ci_hi < 0).sum()),
                                                  bonf8_above0=int((g.sel_minus_mean3_bonf8_lo > 0).sum()), bonf8_below0=int((g.sel_minus_mean3_bonf8_hi < 0).sum()),
                                                  min_delta=float(g.sel_minus_mean3_delta.min()), max_delta=float(g.sel_minus_mean3_delta.max())) for m, g in du.groupby("metric")}
    f["urdu_cells"] = C[C.urdu][["source", "base", "order_win_stored", "rnd_n_metrics_ci_excl0", "rnd_n_metrics_bonf8_excl0"]].to_dict("records")
    for pre in ("sel_minus_mean3", "rnd_minus_mean3"):
        if f"{pre}_stored_delta" in D: f[f"{pre}_max_abs_diff_vs_stored"] = dict(delta=float((D[f"{pre}_delta"] - D[f"{pre}_stored_delta"]).abs().max()), ci_lo=float((D[f"{pre}_ci_lo"] - D[f"{pre}_stored_ci_lo"]).abs().max()),
                                                                               ci_hi=float((D[f"{pre}_ci_hi"] - D[f"{pre}_stored_ci_hi"]).abs().max()), n_equal=bool((D["n_units"] == D[f"{pre}_stored_n"]).all()) if pre == "sel_minus_mean3" else None)
    f["strict_pairing_max_abs_diff_delta"] = float((D.sel_minus_mean3_delta - D.sel_minus_mean3_strict_delta).abs().max())
    ic = ["keys_equal", "rep_file_same_for_3_keys", "same_model_stem", "same_unit_set", "same_gt_sha", "same_evaluator", "same_specialist"]
    f["integrity_all_cells"] = {k: (bool(C[k].all()) if k == "keys_equal" else bool((C[k] == C.n_train).all())) for k in ic}
    f["integrity_totals"] = {k: int(C[k].sum()) for k in ["n_train", "sel_rows", "sel_match_none", "sel_match_one", "sel_match_multi", "sidecars_found_train"]}
    f["sampling_sidecar_keys_differing"] = sorted(set(";".join(C.sampling_sidecar_keys_differing_across_3_candidates.fillna("")).split(";")) - {""}); f["sidecar_example"] = example
    FACTS["randctl"] = f
# ---------------------------------------------------------------- B
def _ink_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.append(E0C)
_G = {}
def _ink_work(t):
    st, path, i = t
    import cv2, e0_adapter as A, uhwr_eval
    from uhwr_eval import score, to_nagamatsu, clean_ink
    import uhwr_eval._nagamatsu_metrics as NM
    if path not in _G:
        z = np.load(path, allow_pickle=False); _G.clear(); _G[path] = {k: z[k] for k in ["gt_points", "gt_stroke_id", "gt_offsets", "unit_ids"]}
    Z = _G[path]; gt = [np.asarray(s, np.float64) for s in A.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], i)]
    rng = np.random.default_rng(20261115 + i); perm = rng.permutation(len(gt))
    V = {"identity": gt, "reverse_points": [s[::-1].copy() for s in gt], "reorder_reversed": gt[::-1], "reorder_random": [gt[k] for k in perm],
         "reverse_points_and_order": [s[::-1].copy() for s in gt[::-1]]}
    sc = lambda a, b: score(a, b, (64, 64), protocol="native", with_lpips=False, with_frechet=False)
    def geom(ink):
        T = to_nagamatsu(clean_ink(ink), (64, 64)); px = ((T[:, :2] + 1.0) / 2.0 * np.array([63, 63], np.float32))
        return T, np.unique(np.round(T[:, :2].astype(np.float64), 9), axis=0), np.unique(px.astype(np.int32), axis=0), NM._render_trajectory(T, (64, 64), 1), NM._render_trajectory(T, (64, 64), 2)
    T0, p0, i0, m10, m20 = geom(gt); asym1 = asym2 = 0
    for s in clean_ink(gt):
        if len(s) < 2: continue
        Ts = to_nagamatsu([s], (64, 64)); Tr = to_nagamatsu([s[::-1].copy()], (64, 64))
        asym1 += int((NM._render_trajectory(Ts, (64, 64), 1) != NM._render_trajectory(Tr, (64, 64), 1)).any()); asym2 += int((NM._render_trajectory(Ts, (64, 64), 2) != NM._render_trajectory(Tr, (64, 64), 2)).any())
    out = []; base = None
    for name, v in V.items():
        r = sc(gt, v); r2 = sc(v, gt); T, p, ii, m1, m2 = geom(v)
        if name == "identity": base = r
        out.append(dict(set=st, unit_id=str(Z["unit_ids"][i]), variant=name, n_strokes=len(gt), n_pts=int(sum(map(len, gt))), n_strokes_lt2pts=int(sum(len(s) < 2 for s in gt)),
                        aiou_self=base["aiou"], aiou_variant_as_pred=r["aiou"], aiou_variant_as_ref=r2["aiou"], ldtw=r["ldtw"], direction_acc=r["direction_acc"], strict_order_acc=r["strict_order_acc"],
                        kendall_tau=r["kendall_tau"], penup_f1_v2=r["penup_f1_v2"],
                        points_equal=bool(p.shape == p0.shape and np.array_equal(p, p0)), int_pixels_equal=bool(ii.shape == i0.shape and np.array_equal(ii, i0)),
                        mask_t1_px_diff=int((m1 != m10).sum()), mask_t2_px_diff=int((m2 != m20).sum()), strokes_asym_t1=asym1, strokes_asym_t2=asym2, n_draw_strokes=int(sum(len(s) >= 2 for s in gt))))
    return out
def inkinv():
    from multiprocessing import get_context
    import inspect, uhwr_eval, cv2, uhwr_eval._nagamatsu_metrics as NM
    sets = {}
    for sp in sorted(glob.glob(Q + "/R3/read_all3/spec_M5__*.json")):
        S = json.load(open(sp)); sets[S["set"]] = str(rows(next(iter(S["cells"]["COMP_pool3"]["files"].values()))).gt_npz.iloc[0])
    tasks = []; samp = {}
    for st, p in sorted(sets.items()):
        n = len(np.load(inp(p), allow_pickle=False)["unit_ids"]); k = min(20 if SMOKE else 500, n)
        idx = sorted(np.random.default_rng(20261115).choice(n, k, replace=False).tolist()); samp[st] = dict(gt_npz=p, n_units_in_file=n, n_sampled=k); tasks += [(st, p, i) for i in idx]
    with get_context("spawn").Pool(PROCS, initializer=_ink_init) as pool: res = pool.map(_ink_work, tasks, chunksize=8)
    D = pd.DataFrame([r for rr in res for r in rr]); D.to_csv(OUT + "/checks_v12_inkinv_units.csv", index=False)
    D["d_pred"] = D.aiou_variant_as_pred - D.aiou_self; D["d_ref"] = D.aiou_variant_as_ref - D.aiou_self
    ag = lambda g: pd.Series(dict(n_units=len(g), n_multi_stroke=int((g.n_strokes >= 2).sum()), mean_aiou_self=g.aiou_self.mean(), mean_aiou_variant_as_pred=g.aiou_variant_as_pred.mean(),
        n_aiou_changed_as_pred=int((g.d_pred.abs() > 1e-12).sum()), max_abs_aiou_change_as_pred=g.d_pred.abs().max(), mean_aiou_change_as_pred=g.d_pred.mean(),
        n_aiou_changed_as_ref=int((g.d_ref.abs() > 1e-12).sum()), max_abs_aiou_change_as_ref=g.d_ref.abs().max(), mean_aiou_change_as_ref=g.d_ref.mean(),
        n_points_not_equal=int((~g.points_equal).sum()), n_int_pixels_not_equal=int((~g.int_pixels_equal).sum()), n_mask_t1_diff=int((g.mask_t1_px_diff > 0).sum()), n_mask_t2_diff=int((g.mask_t2_px_diff > 0).sum()),
        max_mask_t1_px_diff=int(g.mask_t1_px_diff.max()), max_mask_t2_px_diff=int(g.mask_t2_px_diff.max()), n_units_with_asym_stroke_t1=int((g.strokes_asym_t1 > 0).sum()), n_units_with_asym_stroke_t2=int((g.strokes_asym_t2 > 0).sum()),
        mean_direction_acc=g.direction_acc.mean(), mean_strict_order_acc=g.strict_order_acc.mean(), mean_kendall_tau=g.kendall_tau.mean(), mean_penup_f1_v2=g.penup_f1_v2.mean(), n_units_with_stroke_lt2pts=int((g.n_strokes_lt2pts > 0).sum())))
    A1 = D.groupby(["set", "variant"]).apply(ag).reset_index(); A2 = D.groupby("variant").apply(ag).reset_index(); A2.insert(0, "set", "ALL_7_SETS")
    A = pd.concat([A1, A2], ignore_index=True); A.insert(0, "check", "inkinv"); A.to_csv(OUT + "/checks_v12_inkinv.csv", index=False)
    # which mask differences coincide with AIoU changes
    x = D[D.variant != "identity"]; ch = x.d_pred.abs() > 1e-12
    FACTS["inkinv"] = dict(evaluator=f"uhwr_eval {uhwr_eval.__version__}", evaluator_file=uhwr_eval.__file__, cv2=cv2.__version__, sample=samp, seed=20261115,
        all_sets={r["variant"]: {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items() if k not in ("set", "variant")} for r in A2.to_dict("records")},
        changed_as_pred_total=int(ch.sum()), changed_as_pred_with_mask_t1_diff=int((ch & (x.mask_t1_px_diff > 0)).sum()), changed_as_pred_with_int_pixels_equal=int((ch & x.int_pixels_equal).sum()),
        changed_as_pred_with_points_equal=int((ch & x.points_equal).sum()), unchanged_with_mask_t1_diff=int((~ch & (x.mask_t1_px_diff > 0)).sum()),
        render_source=inspect.getsource(NM._render_trajectory))
    for f_ in (uhwr_eval.__file__, NM.__file__): inp(f_)
# ---------------------------------------------------------------- C
def n4358():
    fs = {a: sorted(glob.glob(E + f"CAND/K1/score/rows/test_ouhdl__LUO_ms6_{a}__seed[0-2]__rows031.csv")) for a in ("identity", "mirror_x")}
    T = {}
    for a, ff in fs.items():
        d = pd.concat([rows(f).reset_index() for f in ff]); g = d.groupby("unit_id")
        T[a] = pd.DataFrame(dict(aiou=g.aiou.mean(), direction_acc=g.direction_acc.mean(), n_rows=g.size(), n_dir_nan_rows=g.direction_acc.apply(lambda s: int(s.isna().sum())),
                                 max_dir_eligible=g.n_dir_eligible.max(), n_gt_strokes=g.n_gt_strokes.first(), writer=g.writer_id.first(), max_pred_strokes=g.n_pred_strokes.max(), any_empty=g.empty_pred.max()))
    I, M = T["identity"], T["mirror_x"]; J = I.join(M, lsuffix="_i", rsuffix="_m", how="outer")
    di, dm = J.direction_acc_i.notna(), J.direction_acc_m.notna(); both = di & dm; X = J[~both]; K = J[both]
    dd = (K.direction_acc_i - K.direction_acc_m).abs(); da = (K.aiou_i - K.aiou_m).abs(); allda = (J.aiou_i - J.aiou_m).abs()
    FACTS["n4358"] = dict(files={a: [os.path.basename(f) for f in ff] for a, ff in fs.items()}, n_units_identity=len(I), n_units_mirror=len(M), n_units_union=len(J), n_units_in_both_arms=int((J.n_rows_i.notna() & J.n_rows_m.notna()).sum()),
        n_aiou_defined_both=int((J.aiou_i.notna() & J.aiou_m.notna()).sum()), n_direction_defined_identity=int(di.sum()), n_direction_defined_mirror=int(dm.sum()), n_direction_defined_both=int(both.sum()),
        n_excluded=int((~both).sum()), excluded_undefined_identity_only=int((~di & dm).sum()), excluded_undefined_mirror_only=int((di & ~dm).sum()), excluded_undefined_both=int((~di & ~dm).sum()),
        excluded_rows_all3_seeds_nan_identity=int((X.n_dir_nan_rows_i == X.n_rows_i).sum()), excluded_rows_all3_seeds_nan_mirror=int((X.n_dir_nan_rows_m == X.n_rows_m).sum()),
        excluded_max_dir_eligible_identity_counts={str(k): int(v) for k, v in X.max_dir_eligible_i.fillna(-1).value_counts().items()}, excluded_max_dir_eligible_mirror_counts={str(k): int(v) for k, v in X.max_dir_eligible_m.fillna(-1).value_counts().items()},
        excluded_n_gt_strokes_counts={str(k): int(v) for k, v in X.n_gt_strokes_i.value_counts().sort_index().items()}, excluded_any_empty_pred=int(((X.any_empty_i > 0) | (X.any_empty_m > 0)).sum()), excluded_n_writers=int(X.writer_i.nunique()),
        included_units_with_some_seed_direction_nan=int(((K.n_dir_nan_rows_i > 0) | (K.n_dir_nan_rows_m > 0)).sum()),
        share_dir_disagree_ge_0p5=float((dd >= 0.5).mean()), n_dir_disagree=int((dd >= 0.5).sum()), share_aiou_same_le_0p01=float((da <= 0.01).mean()), n_aiou_same=int((da <= 0.01).sum()),
        share_aiou_same_lt_0p01=float((da < 0.01).mean()), share_joint=float(((dd >= 0.5) & (da <= 0.01)).mean()), n_joint=int(((dd >= 0.5) & (da <= 0.01)).sum()),
        mean_aiou_identity_4358=float(K.aiou_i.mean()), mean_aiou_mirror_4358=float(K.aiou_m.mean()), mean_aiou_identity_all=float(J.aiou_i.mean()), mean_aiou_mirror_all=float(J.aiou_m.mean()),
        mean_direction_identity=float(K.direction_acc_i.mean()), mean_direction_mirror=float(K.direction_acc_m.mean()), share_aiou_same_le_0p01_on_all_units=float((allda <= 0.01).mean()), n_all_units_aiou=int(allda.notna().sum()),
        M3_artifact_values=dict(source="M3_m3_descriptive.csv artifact version 240fde67", n=4358, aiou_same=0.15511702615878845, dir_disagree=0.42748967416245987, joint=0.06333180357962367))
# ---------------------------------------------------------------- D  (functions copied verbatim from exp/RQF/code/rqf_paired_ext.py)
def boot_counts(n,B,seed=0):
    return np.random.default_rng(seed).multinomial(n,np.full(n,1.0/n),size=B).astype(float)
def boot(v,wix,nw,C):
    ok=~np.isnan(v); m=float(v[ok].mean()) if ok.any() else np.nan
    if not ok.any() or C is None: return m,np.nan,np.nan,int(ok.sum())
    S=np.bincount(wix[ok],weights=v[ok],minlength=nw); N=np.bincount(wix[ok],minlength=nw).astype(float)
    num,den=C@S,C@N; bm=num[den>0]/den[den>0]; lo,hi=np.percentile(bm,[2.5,97.5]); return m,float(lo),float(hi),int(ok.sum())
def load(files, flt=None):
    dfs=[]
    for f in files:
        d=pd.read_csv(f,low_memory=False); d["_file"]=os.path.basename(f)
        if "seed" not in d.columns: d["seed"]=0
        dfs.append(d)
    d=pd.concat(dfs,ignore_index=True)
    if flt is not None: d=d[flt(d)]
    return d
def peritem(d,key="unit_id"):
    d=d[d["protocol"]=="native"].copy()
    d[key]=d[key].astype(str)
    cols=[c for c in ["aiou","penup_f1_v2","strict_order_acc","direction_acc","kendall_tau","ldtw","n_gt_strokes"] if c in d.columns]
    g=d.groupby(key,sort=True)
    a=g[cols].mean(numeric_only=True); a["writer"]=g["writer_id"].first().astype(str)
    a["tau_cp"]=np.where(a["n_gt_strokes"]>=2,np.nan_to_num(a["kendall_tau"].to_numpy(float),nan=0.0),np.nan)
    a["n_rows"]=g.size(); return a
METD = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "tau_cp", "ldtw"]
def ft1abs():
    g = lambda p: sorted(glob.glob(E + p, recursive=True)); M = {}; info = {}
    def reg(name, files, device="", note=""):
        for f in files: inp(f)
        M[name] = peritem(load(files)); info[name] = dict(device=device, note=note, n_files=len(files), files=[os.path.relpath(f, E) for f in files], n_units=len(M[name]), rows_per_unit_median=float(M[name].n_rows.median()))
    reg("lineHT_d4", g("CAND/method_round/ConfirmV2/score_out/4090/**/test_ouhdl__lineHT_d4*__rows031.csv"), "RTX 4090")
    reg("LineT_d4", g("CAND/method_round/ConfirmV2/score_out/4090/**/test_ouhdl__lineT_d4*__rows031.csv"), "RTX 4090")
    reg("COMP_rep3", g("ORDER/sealed/COMP_rep3/score_out/**/test_ouhdl__*_rep3__seed0__rows031.csv"), "RTX 4090 (verifier on CPU)")
    reg("NOFM_W_FT", g("FMD/xs/runs/fmdx_monitor/score_out/**/test_ouhdl__NOFM_W_ft_ms1__seed*__rows031.csv"), "RTX A4000")
    reg("PEINIT_W_FT", g("FMD/xs/runs/fmdx_monitor/score_out/**/test_ouhdl__PEINIT_W_ft_ms2__seed*__rows031.csv"), "RTX A4000")
    reg("E0scratch_p01", g("D2_MOE/controls/score_out/rows/test_ouhdl__E0scratch_p01_ms10__seed*__rows031.csv"), "cluster GPU")
    reg("PEN-Net", [E + "PENNET/results/test_rendered_ms10_rows.csv"]); reg("InkSight_e0r", [E + "RB4_baselines/inksight_e0r/rows/inksight_e0r_rendered_rows.csv"]); reg("InkSight_g1f4", [E + "RB4_baselines/inksight_e0r/rows/inksight_g1f4_rendered_rows.csv"])
    ORIG = list(M); common = sorted(set.intersection(*[set(M[k].index) for k in ORIG]))
    S = json.load(open(inp(E + "ORDER/sealed/COMP_rep3/read/spec_test_ouhdl.json"))); pool = sorted(set(S["cells"]["COMP_pool3"]["files"].values()))
    gl = [f for f in g("ORDER/sealed/COMP_rep3/score_out/**/test_ouhdl__comp_p01_identity_s*_identity__seed[0-2]__rows031.csv")]
    reg("FT1pct_plain", pool, S["cells"]["COMP_pool3"].get("device", ""), "COMP_pool3 files of the ORDER sealed spec: the 3 training seeds x 3 sampling seeds that COMP_rep3 selects from; per-unit mean")
    info["FT1pct_plain"]["glob_equals_spec_pool"] = bool(sorted(os.path.realpath(x) for x in gl) == sorted(os.path.realpath(x) for x in pool))
    a4 = [f for f in g("FMD/xs/runs/fmdx_monitor/score_out_cmp/**/test_ouhdl__comp_p01*__seed*__rows031.csv") if "_rep3" not in f and "_rnd3" not in f]
    if a4: reg("FT1pct_plain_A4000cmp", a4, "RTX A4000 (per v10 table-source record)", "comparator rows of tab:warm (fmdx_monitor/score_out_cmp); different device from COMP_rep3")
    absr = []
    for k in list(M):
        miss = len(set(common) - set(M[k].index)); A = M[k].loc[[u for u in common if u in M[k].index]]; uw, wix = np.unique(A["writer"].to_numpy(), return_inverse=True); C = boot_counts(len(uw), B)
        r = dict(check="ft1abs", method=k, n_units=len(A), n_writers=len(uw), n_common_units_missing=miss, device=info[k]["device"], n_files=info[k]["n_files"])
        for met in METD:
            m = boot(A[met].to_numpy(float), wix, len(uw), C); r[met] = m[0]; r[met + "_lo"] = m[1]; r[met + "_hi"] = m[2]; r[met + "_ndef"] = m[3]
        absr.append(r)
    D = pd.DataFrame(absr); st = pd.read_csv(inp(Q + "/tables/paired_ext/absolute_common_units.csv")).set_index("method"); num = [c for c in st.columns if c in D.columns and c not in ("method",)]
    dif = (D[D.method.isin(ORIG)].set_index("method")[num].astype(float) - st.loc[ORIG, num].astype(float)).abs().max().max()
    D["in_stored_absolute_common_units_csv"] = D.method.isin(ORIG); D.to_csv(OUT + "/checks_v12_ft1abs.csv", index=False)
    FACTS["ft1abs"] = dict(n_common=len(common), max_abs_diff_recomputed_vs_stored_9_methods=float(dif), methods=info, rows={r["method"]: {k: r[k] for k in r if k not in ("check", "method")} for r in absr if r["method"].startswith("FT1pct") or r["method"] == "COMP_rep3"})
# ---------------------------------------------------------------- E
def screen():
    key = lambda J: str(J.get("status")); vs = sorted(glob.glob(E + "FMD/xs/verdicts/verdict_*.json")); now = collections.Counter(); tri = collections.Counter()
    for v in vs:
        J = json.load(open(inp(v))); now[key(J)] += 1; tri[f"urdu_pass={J.get('urdu_pass')}|xs_pass={J.get('xs_pass')}|status={J.get('status')}"] += 1
    f = dict(verdict_dir="exp/FMD/xs/verdicts", n_verdict_files_now=len(vs), status_counts_now=dict(now), urdu_xs_status_counts_now=dict(tri))
    tg = E + "FMD/xs/read_v1/snap_final_20261107T0047Z.tgz"
    if os.path.exists(tg):
        sn = collections.Counter(); n = 0
        with tarfile.open(inp(tg)) as tf:
            for mem in tf.getmembers():
                if os.path.basename(mem.name).startswith("verdict_") and mem.name.endswith(".json"): sn[key(json.load(tf.extractfile(mem)))] += 1; n += 1
        f.update(snapshot=os.path.relpath(tg, E), n_verdict_files_snapshot=n, status_counts_snapshot=dict(sn))
    for gr in ("v1_1_fmdx_gate_read.csv", "v1_fmdx_gate_read.csv"):
        p = E + "FMD/xs/read_v1/" + gr
        if os.path.exists(p):
            d = pd.read_csv(inp(p)); f[gr] = dict(n_rows=len(d), columns=list(d.columns)[:40], counts={c: {str(k): int(v) for k, v in d[c].value_counts(dropna=False).items()} for c in d.columns if d[c].nunique(dropna=False) <= 8})
    FACTS["screen"] = f
if __name__ == "__main__":   # guard: Pool("spawn") re-imports this module in every worker (run 1, Slurm 25171088, lacked it and was cancelled)
    part("C_n4358", n4358); part("E_screen", screen); part("D_ft1abs", ft1abs); part("A_randctl", randctl); part("B_inkinv", inkinv)
    PROV["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); PROV["n_inputs"] = len(PROV["inputs"])
    json.dump(FACTS, open(OUT + "/checks_v12_facts.json", "w"), indent=1, default=str)
    PROV["outputs"] = {f: sha(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.startswith("checks_v12_")}
    json.dump(PROV, open(OUT + "/PROVENANCE_checks_v12.json", "w"), indent=1, default=str)
    print("STATUS", json.dumps(PROV["status"])[:3000]); sys.exit(0 if all(v.startswith("ok") for v in PROV["status"].values()) else 4)
