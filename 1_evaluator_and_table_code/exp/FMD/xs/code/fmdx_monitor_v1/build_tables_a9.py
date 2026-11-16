#!/usr/bin/env python3
"""FMDX warm monitor (A9 version: + device_a/device_b/DEVICE_MIXED per contrast row, device per abs row; prereg A9 sha256 7a71c715...): absolute + paired contrast tables, NO verdicts, NO promotion labels (the lead applies prereg A4.5).
Inputs: per-unit rows from lane_score.py (uhwr_eval 0.3.1; columns unit_id, writer_id, seed, protocol, aiou, penup_f1_v2, strict_order_acc,
direction_acc, kendall_tau, ldtw, ...), file name <set>__<tag>__seed<k>__rows031.csv; tag = <ARM>_<bb|ft>_ms<m>.
Stats (prereg Section 4 / A4.5): per-unit value = mean over sampling seeds 0-2 of the native-protocol rows; contrast = paired per-unit difference
(guided - comparator) over units present in both; 95 % percentile interval of B = 2,000 writer-cluster bootstrap resamples,
numpy.random.default_rng(20261103), algorithm identical to loop_tables.boot_idx / fmd_harness.stats.boot_idx (writers drawn with replacement, all their
units enter). german_onhw_all (2 writers): unit-cluster CI (Route A convention).
"""
import argparse, glob, json, os, re, sys, hashlib
import numpy as np, pandas as pd
METRICS = ["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "ldtw"]
ALIAS = {"kendall_tau": "tau_cp"}
SIX = ["test_casia_zh", "test_casia_en", "test_hpl_tamil", "test_iam500", "german_unipen_de_test", "german_onhw_all"]
LUO_DIR = "$U/exp/D3_ROUTE/routeA/score_out/rows"; LUO_TAG = "LUO_ms6"   # 4090 reference defaults; --luo-dir/--comp-dir/--luo-tag override (A8: primary comparators = A4000 re-sampling)
COMP_DIR = "$U/exp/CAND/method_round/COMP/score_out/4090/rows"
COMP_TAGS = {"20261034": "comp_p01_identity_s20261034_vps_ms3_identity", "20261035": "comp_p01_identity_s20261035_ms3_identity", "20261036": "comp_p01_identity_s20261036_ms4_identity"}
B = 2000; SEED = 20261103

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
USED = {}
def load(paths):
    dfs = []
    for p in paths:
        d = pd.read_csv(p, usecols=lambda c: c in ["unit_id", "writer_id", "seed", "protocol", "error"] + METRICS)
        dfs.append(d); USED[p] = None
    if not dfs: return None
    d = pd.concat(dfs); d = d[d.protocol == "native"]
    return d
def unit_table(d):
    """mean over sampling seeds per unit; also returns n_seeds"""
    g = d.groupby("unit_id")
    t = g[METRICS].mean(); t["writer_id"] = g.writer_id.first(); t["n_seeds"] = g.seed.nunique()
    return t
def rows_for(sset, tag, seeds=(0, 1, 2), d=None):
    d = d or ""
    return [p for s in seeds for p in glob.glob(f"{d}/{sset}__{tag}__seed{s}__rows031.csv")]
def boot_idx(w):
    w = np.asarray(w); uw = np.unique(w); rng = np.random.default_rng(SEED)
    pos = {k: np.where(w == k)[0] for k in uw}
    for _ in range(B):
        pick = rng.choice(uw, len(uw), replace=True)
        yield np.concatenate([pos[k] for k in pick])

PRED_GLOBS = ["$U/exp/FMD/xs/runs/fmdx_monitor/preds/*/*", "$U/exp/FMD/xs/runs/fmdx_monitor/comparators_a4000/*/preds"]
_DEV = {}
def dev_of(sset, tag, seeds=(0, 1, 2)):
    """device type(s) of the prediction sidecars behind (set, tag); composed sidecars give device_bb + device_ft"""
    k = (sset, tag, tuple(seeds))
    if k not in _DEV:
        ds = set()
        for g in PRED_GLOBS:
            for s in seeds:
                for p in glob.glob(f"{g}/{sset}__{tag}__seed{s}_preds.json"):
                    try: j = json.load(open(p))
                    except Exception: continue
                    for f in ("device", "device_bb", "device_ft"):
                        if j.get(f): ds.add(str(j[f]).replace("NVIDIA ", ""))
        _DEV[k] = "|".join(sorted(ds)) if ds else "unknown"
    return _DEV[k]
def dev_union(*xs):
    ds = set()
    for x in xs: ds |= set(x.split("|"))
    return "|".join(sorted(ds))
_cache = {}
def paired(a, b, sset):
    """per-metric pairing: a unit enters a metric's contrast when both arms have a value for it (tau/strict are undefined for 1-stroke units)"""
    j = a[METRICS + ["writer_id"]].join(b[METRICS], rsuffix="_c", how="inner")
    if len(j) == 0: return None
    cl = np.arange(len(j)) if sset == "german_onhw_all" else j.writer_id.to_numpy()
    D = np.column_stack([(j[m] - j[m + "_c"]).to_numpy() for m in METRICS])
    V = ~np.isnan(D); S = np.where(V, D, 0.0)
    key = (sset, hashlib.md5(cl.tobytes()).hexdigest())
    if key not in _cache:
        _cache.clear(); _cache[key] = [i for i in boot_idx(cl)]
    bs = np.array([S[i].sum(0) / np.maximum(V[i].sum(0), 1) for i in _cache[key]])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    res = {}
    for k, m in enumerate(METRICS):
        n = int(V[:, k].sum())
        res[m] = dict(delta=float(S[:, k].sum() / n) if n else float("nan"), ci_lo=float(lo[k]) if n else float("nan"), ci_hi=float(hi[k]) if n else float("nan"),
                      n=n, ncl=int(len(np.unique(cl[V[:, k]]))))
    return res

def pooled_comp(sset):
    parts = []
    for sd, tag in COMP_TAGS.items():
        ps = rows_for(sset, tag, d=COMP_DIR)
        d = load(ps)
        if d is not None and len(d): parts.append(unit_table(d)[METRICS])
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--arm-meta", required=True)
    ap.add_argument("--read-label", default="")
    ap.add_argument("--luo-dir"); ap.add_argument("--comp-dir"); ap.add_argument("--luo-tag"); ap.add_argument("--comp-device-note", default="")
    global LUO_DIR, COMP_DIR, LUO_TAG
    a = ap.parse_args()
    LUO_DIR = a.luo_dir or LUO_DIR; COMP_DIR = a.comp_dir or COMP_DIR; LUO_TAG = a.luo_tag or LUO_TAG
    meta = json.load(open(a.arm_meta))     # {arm: {fm, status, regime, controls: bool}}
    # discover arm tags present
    avail = {}
    for f in glob.glob(a.rows_dir + "/*__rows031.csv"):
        m = re.match(r"(.+?)__((.+?)_(bb|ft|cmp)_ms(\d+))__seed(\d+)__rows031\.csv$", os.path.basename(f))
        if m: avail.setdefault((m.group(3), m.group(4)), {}).setdefault((m.group(1), m.group(2)), []).append(int(m.group(6)))
    absr, conr = [], []
    UT = {}   # (arm, kind, set) -> unit table
    def get(arm, kind, sset):
        k = (arm, kind, sset)
        if k not in UT:
            tags = [t for (s, t) in avail.get((arm, kind), {}) if s == sset]
            if not tags: UT[k] = None
            else:
                tag = tags[0]; d = load(rows_for(sset, tag, d=a.rows_dir))
                t = unit_table(d); t.attrs["tag"] = tag; t.attrs["milestone"] = int(re.search(r"_ms(\d+)$", tag).group(1)); t.attrs["dev"] = dev_of(sset, tag); UT[k] = t
        return UT[k]
    luo = {}
    for s in SIX:
        d = load(rows_for(s, LUO_TAG, d=LUO_DIR)); luo[s] = unit_table(d) if d is not None else None
        if luo[s] is not None: luo[s].attrs["dev"] = dev_of(s, LUO_TAG)
    comp = {}
    for s in ("test_ouhdl", "val1000_ouhdl"):
        parts = pooled_comp(s)
        comp[s] = (sum(parts) / len(parts), len(parts)) if parts else (None, 0)
        if parts:  # writer ids from first COMP file
            d = load(rows_for(s, list(COMP_TAGS.values())[1], d=COMP_DIR)); comp[s][0]["writer_id"] = unit_table(d)["writer_id"]; comp[s][0].attrs["dev"] = dev_union(*[dev_of(s, t_) for t_ in COMP_TAGS.values()])
    def add_abs(arm, kind, sset, t):
        r = dict(arm=arm, kind=kind, set=sset, milestone=t.attrs.get("milestone"), n_units=len(t), n_seeds_min=int(t.n_seeds.min()), n_seeds_max=int(t.n_seeds.max()),
                 fm=meta.get(arm, {}).get("fm"), fm_status_at_read=meta.get(arm, {}).get("status"), device=t.attrs.get("dev", "unknown"), label="EXPLORATORY (single training seed)")
        for m in METRICS: r[m] = float(t[m].mean())
        absr.append(r)
    def add_con(cell, arm, kind, sset, cname, ta, tb, note=""):
        if ta is None or tb is None: return
        p = paired(ta, tb, sset)
        if p is None: return
        res = p
        da, db = ta.attrs.get("dev", "unknown"), tb.attrs.get("dev", "unknown"); mixed = (da != db) or ("|" in da) or ("|" in db) or ("unknown" in (da + db))
        for m in METRICS:
            conr.append(dict(cell=cell, arm=arm, kind=kind, set=sset, comparator=cname, metric=m, metric_alias=ALIAS.get(m, m), delta=res[m]["delta"], ci_lo=res[m]["ci_lo"], ci_hi=res[m]["ci_hi"],
                             n_units=res[m]['n'], n_clusters=res[m]['ncl'], cluster=("unit" if sset == "german_onhw_all" else "writer"), arm_milestone=ta.attrs.get("milestone", ""),
                             fm=meta.get(arm, {}).get("fm"), fm_status_at_read=meta.get(arm, {}).get("status"), note=(note + ("; " if note else "") + "DEVICE_MIXED: DDIM sampling is not bitwise reproducible across GPU types (prereg A9.2)") if mixed else note, device_a=da, device_b=db, DEVICE_MIXED=bool(mixed), label="EXPLORATORY (single training seed)"))
    arms = sorted({k[0] for k in avail})
    CTRLS = {"W": ["NOFM_W", "PEINIT_W"], "S": ["NOFM_S", "PEINIT_S"]}; CTRL = CTRLS["W"] + CTRLS["S"]   # controls matched by regime suffix (_W warm, _S scratch)
    comp0 = {}
    for s_ in SIX:   # COMP-p01 sampling-seed-0 rows on the six sets (per training seed) and their per-unit pool
        parts = []
        for sd, tag in COMP_TAGS.items():
            d = load(rows_for(s_, tag, seeds=(0,), d=COMP_DIR))
            if d is not None and len(d):
                tt_ = unit_table(d); tt_.attrs["dev"] = dev_of(s_, tag, seeds=(0,)); parts.append((sd, tt_))
        if parts:
            pool = sum(t[METRICS] for _, t in parts) / len(parts); pool["writer_id"] = parts[0][1]["writer_id"]; pool.attrs["dev"] = dev_union(*[t.attrs["dev"] for _, t in parts])
            comp0[s_] = (pool, parts)
    for arm in arms:
        for s in SIX:
            tb, tf, tc = get(arm, "bb", s), get(arm, "ft", s), get(arm, "cmp", s)
            for kind, t in (("bb", tb), ("ft", tf), ("cmp", tc)):
                if t is not None: add_abs(arm, kind, s, t)
            if luo[s] is not None:
                add_con("X_backbone_only_vs_LUOms6", arm, "bb", s, "LUO_ms6", tb, luo[s])
                add_con("X_composed_vs_LUOms6", arm, "cmp", s, "LUO_ms6", tc, luo[s])
            if s in comp0:
                add_con("X_FTnoRouter_vs_COMPp01_seed0_pooled", arm, "ft", s, f"pooled_COMP_p01_samplingseed0_x{len(comp0[s][1])}", tf, comp0[s][0], note="forgetting read; comparator has sampling seed 0 only")
                for sd, tt in comp0[s][1]: add_con("X_FTnoRouter_vs_singleCOMP_seed0", arm, "ft", s, f"COMP_{sd}_samplingseed0", tf, tt)
            if arm not in CTRL:
                for c in CTRLS.get(arm[-1], []):
                    if c != arm:
                        add_con("F_backbone_vs_control", arm, "bb", s, c, tb, get(c, "bb", s))
                        add_con("F_composed_vs_control", arm, "cmp", s, c, tc, get(c, "cmp", s))
                        add_con("F_FTnoRouter_vs_control", arm, "ft", s, c, tf, get(c, "ft", s))
        for s in ("test_ouhdl", "val1000_ouhdl"):
            t = get(arm, "ft", s)
            if t is None: continue
            add_abs(arm, "ft", s, t)
            pc, npc = comp[s]
            if pc is not None: add_con("U_FT_vs_pooledCOMPp01", arm, "ft", s, f"pooled_COMP_p01_seeds_{npc}", t, pc, note=f"pooled over {npc} COMP seeds")
            for sd, tag in COMP_TAGS.items():
                d = load(rows_for(s, tag, d=COMP_DIR))
                if d is not None and len(d):
                    tt_ = unit_table(d); tt_.attrs["dev"] = dev_of(s, tag); add_con("U_FT_vs_singleCOMP", arm, "ft", s, f"COMP_{sd}", t, tt_)
            if arm not in CTRL:
                for c in CTRLS.get(arm[-1], []):
                    if c != arm: add_con("F_FT_vs_control", arm, "ft", s, c, t, get(c, "ft", s))
    pd.DataFrame(absr).to_csv(a.out + "_absolute.csv", index=False)
    cdf = pd.DataFrame(conr); cdf.to_csv(a.out + "_contrasts.csv", index=False)
    # markdown (facts only)
    L = [f"# FMDX warm contrasts (no verdicts, no promotion labels)", f"Read label: {a.read_label}", "",
         "Per-unit = mean over sampling seeds 0-2 (native protocol, uhwr_eval 0.3.1); paired difference arm - comparator; 95 % percentile CI of B=2000 writer-cluster bootstrap, default_rng(20261103); german_onhw_all unit-cluster. "
         "Comparator rows: " + (a.comp_device_note or "default") + ". Every row: EXPLORATORY (single training seed). Arms read: " + str(len(arms)) + " (" + ", ".join(arms) + "). No multiplicity correction.", ""]
    if len(cdf):
        for (cell, sset), g in cdf.groupby(["cell", "set"], sort=False):
            L.append(f"## {cell} / {sset}")
            piv = g[g.metric.isin(["aiou", "penup_f1_v2", "strict_order_acc", "direction_acc", "kendall_tau", "ldtw"])]
            L.append("| arm | comparator | device_a / device_b | n | " + " | ".join(f"{ALIAS.get(m, m)} d [lo, hi]" for m in METRICS) + " |")
            L.append("|---|---|---|---|" + "---|" * len(METRICS))
            for (arm, comp_), h in piv.groupby(["arm", "comparator"], sort=False):
                hh = h.set_index("metric")
                L.append(f"| {arm} | {comp_} | {h.device_a.iloc[0]} / {h.device_b.iloc[0]}{' DEVICE_MIXED' if h.DEVICE_MIXED.iloc[0] else ''} | {int(h.n_units.min())}-{int(h.n_units.max())} | " + " | ".join(f"{hh.loc[m, 'delta']:+.4f} [{hh.loc[m, 'ci_lo']:+.4f}, {hh.loc[m, 'ci_hi']:+.4f}]" for m in METRICS) + " |")
            L.append("")
    open(a.out + "_contrasts.md", "w").write("\n".join(L))
    inputs = {p: sha(p) for p in sorted(USED)}
    json.dump(dict(inputs_sha256=inputs, n_inputs=len(inputs), arms=arms, code_sha256=sha(__file__), B=B, seed=SEED, n_contrast_rows=len(cdf), n_abs_rows=len(absr)), open(a.out + "_inputs.json", "w"), indent=1)
    print("arms", arms, "abs rows", len(absr), "contrast rows", len(cdf))
if __name__ == "__main__":
    main()
