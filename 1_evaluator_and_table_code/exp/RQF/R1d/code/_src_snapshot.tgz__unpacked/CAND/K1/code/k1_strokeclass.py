"""CAND K1 per-stroke direction / loop-orientation scorer with a stroke-sense class per GT stroke.

Per preds file (Route A layout, frame_sample.py sidecar) and unit: GT ink = e0_adapter.unpack_ink(gt npz), prediction =
e0_adapter.decode(preds[k]) (preds = ORIGINAL frame, inverse applied); both cleaned with uhwr_eval.clean_ink (native
protocol, no resampling); strokes matched with uhwr_eval.metrics.match_strokes (Hungarian on mask IoU, thickness 5,
iou_min 0.5) - the same calls uhwr_eval 0.3.1 order_metrics() makes, so per-unit sums reproduce direction_acc /
loop_orient_acc of the rows031 files (checked by k1_summary_cand.py).

Stroke class (rule from the stroke-sense analysis, candidates document sec. 2 / PROVENANCE_stroke_sense_canonical_frames.json):
per GT stroke with >= 3 distinct points: total signed turning angle T (image coords, y down; sum of wrapped heading
changes between consecutive non-zero segments), net dx/L, net dy/L (L = polyline length); dominant component =
argmax(|dx/L|, |dy/L|, |T/2pi|) -> 'horizontal' | 'vertical' | 'chirality'.  Strokes with < 3 distinct points -> 'lt3pts'.
The class is a property of the GT stroke (original frame) and is identical for every frame.

Out: <out>/strokes/<stem>__strokes.csv.gz, one row per GT stroke: set, frame, seed, unit_id, writer_id, sample_id, gt_idx,
n_pts, n_distinct, L, dx_n, dy_n, turn_n, cls, matched, dir_elig, dir_ok, loop_elig, loop_ok.
python k1_strokeclass.py --preds-dir P --out O --e0code E0C --procs 16 --only GLOB
"""
import argparse, glob, json, math, os, sys, time, hashlib
from multiprocessing import get_context
import numpy as np, pandas as pd

GT = {}

def _init(e0code):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import uhwr_eval
    assert uhwr_eval.__version__ == "0.3.1", (uhwr_eval.__version__, uhwr_eval.__file__)
    sys.path.append(e0code)

def _gt(path):
    if path not in GT:
        GT.clear(); z = np.load(path, allow_pickle=False)
        GT[path] = {k: z[k] for k in ["gt_points", "gt_stroke_id", "gt_offsets"]}
    return GT[path]

def stroke_sense(s):
    s = np.asarray(s, np.float64)
    nd = len(np.unique(np.round(s, 9), axis=0)) if len(s) else 0
    seg = np.diff(s, axis=0); sl = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
    L = float(sl.sum())
    if nd < 3 or L <= 0:
        return nd, L, np.nan, np.nan, np.nan, "lt3pts"
    dx, dy = (s[-1] - s[0]) / L
    nz = seg[sl > 0]; ang = np.arctan2(nz[:, 1], nz[:, 0])
    dang = np.diff(ang); dang = (dang + np.pi) % (2 * np.pi) - np.pi
    T = float(dang.sum()) / (2 * np.pi)
    cls = ("horizontal", "vertical", "chirality")[int(np.argmax([abs(dx), abs(dy), abs(T)]))]
    return nd, L, float(dx), float(dy), T, cls

def _work(t):
    gt_path, i, arr = t
    import e0_adapter as A
    from uhwr_eval import clean_ink
    from uhwr_eval.metrics import match_strokes, signed_area
    from uhwr_eval.strokes import stroke_length
    Z = _gt(gt_path)
    gt = clean_ink(A.unpack_ink(Z["gt_points"], Z["gt_stroke_id"], Z["gt_offsets"], i))
    try:
        pr = clean_ink(A.decode(arr))
    except Exception:
        pr = []
    m = {gi: pj for gi, pj, _ in match_strokes(gt, pr, (64, 64), 5, 0.5)} if pr else {}
    rows = []
    for gi, g in enumerate(gt):
        nd, L, dx, dy, T, cls = stroke_sense(g)
        rec = dict(gt_idx=gi, n_pts=len(g), n_distinct=nd, L=L, dx_n=dx, dy_n=dy, turn_n=T, cls=cls, matched=int(gi in m),
                   dir_elig=0, dir_ok=0, loop_elig=0, loop_ok=0)
        if gi in m:
            p = pr[m[gi]]
            if np.linalg.norm(g[-1] - g[0]) >= 2.0:
                rec["dir_elig"] = 1; rec["dir_ok"] = int(np.linalg.norm(p[0] - g[0]) < np.linalg.norm(p[0] - g[-1]))
            Ar = signed_area(g); Lg = stroke_length(g)
            if abs(Ar) >= 4.0 and Lg > 0 and 4 * math.pi * abs(Ar) / Lg ** 2 >= 0.1:
                rec["loop_elig"] = 1; rec["loop_ok"] = int(np.sign(signed_area(p)) == np.sign(Ar))
        rows.append(rec)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--e0code", required=True)
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--only", default="*_preds.npz"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-root", default="$U")
    a = ap.parse_args()
    import uhwr_eval; assert uhwr_eval.__version__ == "0.3.1"
    os.makedirs(os.path.join(a.out, "strokes"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.preds_dir, a.only)))
    pool = get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.e0code,))
    for f in files:
        t0 = time.time(); stem = os.path.basename(f).replace("_preds.npz", "")
        outp = os.path.join(a.out, "strokes", stem + "__strokes.csv.gz")
        if os.path.exists(outp): print("exists, skip", stem, flush=True); continue
        side = json.load(open(f.replace("_preds.npz", "_preds.json")))
        gt_path = side.get("npz_clusterA") or os.path.join(a.gt_root, side["npz_rel"])
        z = np.load(f, allow_pickle=False); P = z["preds"]; pu = [str(u) for u in z["unit_ids"]]
        G = np.load(gt_path, allow_pickle=False); gu = [str(u) for u in G["unit_ids"]]; pos = {u: i for i, u in enumerate(gu)}
        gw = G["writer_ids"]; gs = G["sample_ids"]
        idx = [pos[u] for u in pu]
        if a.limit: idx = idx[: a.limit]
        res = pool.imap(_work, [(gt_path, i, P[k]) for k, i in enumerate(idx)], chunksize=8)
        rows = []
        for k, rr in enumerate(res):
            i = idx[k]
            for r in rr:
                rows.append(dict(set=side["set"], frame=side["frame"], seed=int(side["seed"]), unit_id=pu[k], writer_id=str(gw[i]), sample_id=str(gs[i]), **r))
        df = pd.DataFrame(rows); df["preds_file"] = os.path.basename(f); df["preds_sha256"] = side.get("preds_sha256", "")
        df.to_csv(outp, index=False, compression="gzip")
        print(dict(file=os.path.basename(f), n_units=len(idx), n_strokes=len(df), seconds=round(time.time() - t0, 1)), flush=True)
    pool.close(); pool.join()

if __name__ == "__main__":
    main()
