"""M1a: Urdu script rules applied after the fact to recovered trajectories (no sampler change).

python -m uhwr_exp.m1a_run --bundle bundle --pred out/pred --out out/m1a [--workers N] [--limit K]

Rules (eval-canvas coordinates, x to the right; Urdu is written right to left):
  flip_all    every stroke whose chord |dx| >= 2 line-px and whose start lies LEFT of its end is
              reversed ("strokes start at their right end")
  flip_body   the same, but only for body strokes (marks are left alone; GT check: small marks
              are often written left to right)
  order_ab    body strokes re-ordered right to left by rightmost ink; each mark is placed right
              after the body whose x-extent contains its x-centre (else the body with the
              nearest x-centre), several marks of one body right to left
  order_ml    body strokes right to left, then all marks right to left ("marks last")
  combinations flip_body+order_ab, flip_all+order_ab, flip_body+order_ml
Marks are predicted strokes whose bbox sides are both < 0.18 x 56 line-px (the same size rule as
the geometric baseline / oracle units).  Thresholds are in LINE-canvas pixels and are converted
for ligature units with the unit's scale.  The PF2 delayed-stroke slot is not available yet
(PF2 not run), so order_ab / order_ml stand in for "delayed strokes go into their slot".
Applied to: geo_rtl, geo_rtl_local, momna_s123_part (ligature level, reference inputs),
geo_rtl, geo_rtl_local, momna_s123, momna_s123_subpaths (line level, reference inputs), and to
the GT itself (rule ceiling).  Inputs: reference condition and dataset img/.
Metrics: uhwr_eval 0.2.0 native protocol, without LPIPS (rules only permute/reverse strokes, so
masks, AIoU, Chamfer, Hausdorff and LPIPS cannot change).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UHWR_TORCH_THREADS", "1")

import numpy as np  # noqa: E402

RULES = ["none", "flip_all", "flip_body", "order_ab", "order_ml", "flip_body+order_ab",
         "flip_all+order_ab", "flip_body+order_ml"]
MARK = 0.18 * 56
G = {}


def apply_rule(ink, rule, k):
    """k = eval px per line px (1 for lines, unit scale for ligatures)."""
    ink = [np.asarray(s, float) for s in ink if len(s)]
    if rule == "none" or not ink:
        return ink
    w = [np.ptp(s[:, 0]) if len(s) else 0 for s in ink]
    h = [np.ptp(s[:, 1]) if len(s) else 0 for s in ink]
    mark = [(wi < MARK * k and hi < MARK * k) for wi, hi in zip(w, h)]
    out = [s.copy() for s in ink]
    parts = rule.split("+")
    for p in parts:
        if p.startswith("flip"):
            for i, s in enumerate(out):
                if p == "flip_body" and mark[i]:
                    continue
                if len(s) >= 2 and abs(s[-1, 0] - s[0, 0]) >= 2.0 * k and s[0, 0] < s[-1, 0]:
                    out[i] = s[::-1].copy()
    for p in parts:
        if p.startswith("order"):
            bodies = [i for i in range(len(out)) if not mark[i]]
            marks = [i for i in range(len(out)) if mark[i]]
            if not bodies:
                bodies, marks = marks, []
            bodies.sort(key=lambda i: -out[i][:, 0].max())
            xc = {i: 0.5 * (out[i][:, 0].min() + out[i][:, 0].max()) for i in range(len(out))}
            if p == "order_ml":
                seq = bodies + sorted(marks, key=lambda i: -xc[i])
            else:
                slot = {b: [] for b in bodies}
                for m in marks:
                    cand = [b for b in bodies if out[b][:, 0].min() <= xc[m] <= out[b][:, 0].max()]
                    b = cand[0] if cand else min(bodies, key=lambda b: abs(xc[b] - xc[m]))
                    slot[b].append(m)
                seq = []
                for b in bodies:
                    seq += [b] + sorted(slot[b], key=lambda i: -xc[i])
            out = [out[i] for i in seq]
            mark = [mark[i] for i in seq]
    return out


def _init(bundle, pred):
    import pandas as pd
    from uhwr_exp.common import load_lines
    G["lines"] = load_lines(bundle)
    G["idx"] = pd.read_csv(os.path.join(bundle, "ligatures_v3_test.csv"))
    G["pred"] = pred


def run_line(sid):
    from uhwr_eval import score
    from uhwr_exp.common import line_sample
    from uhwr_exp.e2_score import REF, partition
    from uhwr_exp.predict_run import G as PG, units_for_line
    PG["idx"] = G["idx"]
    d = G["lines"][sid]
    L = line_sample(d, sid)
    units = units_for_line(sid, L)
    P = pickle.load(open(os.path.join(G["pred"], f"pred_{sid}.pkl"), "rb"))
    for c in set(REF.values()) | {"dataset"}:
        if ("line", "momna_s123", c, sid) in P:
            for uid, ink in partition(P[("line", "momna_s123", c, sid)], L, units).items():
                P[("ligature", "momna_s123_part", c, uid)] = ink
    jobs = []
    for (lev, meth, cond, uid), pr in P.items():
        if cond in (REF[meth], "dataset"):
            jobs.append((lev, meth, cond, uid, pr))
    jobs.append(("line", "GT", "-", sid, L.ink))
    for u in units:
        jobs.append(("ligature", "GT", "-", u["unit_id"], u["ink"]))
    gts = {("line", sid): (L.ink, L.canvas, 1.0)}
    for u in units:
        gts[("ligature", u["unit_id"])] = (u["ink"], u["canvas"], u["T"][0])
    rows = []
    for lev, meth, cond, uid, pr in jobs:
        gt, canvas, k = gts[(lev, uid)]
        for rule in RULES:
            q = apply_rule(pr, rule, k)
            try:
                r, e = score(gt, q, canvas, protocol="native", with_lpips=False), ""
            except Exception as ex:
                r, e = {}, f"{type(ex).__name__}: {ex}"
            rows.append(dict(sample_id=sid, writer_id=d["writer_id"], unit=lev, unit_id=uid,
                             method=meth, input=cond, rule=rule, error=e, **r))
    return sid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "rows"), exist_ok=True)
    sids = sorted(f[5:-4] for f in os.listdir(a.pred) if f.startswith("pred_"))
    if a.limit:
        sids = sids[: a.limit]
    todo = [s for s in sids if not os.path.exists(os.path.join(a.out, "rows", f"{s}.pkl"))]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.bundle, a.pred),
                                      maxtasksperchild=20) as pool:
        for k, (sid, rows) in enumerate(pool.imap_unordered(run_line, todo)):
            pickle.dump(rows, open(os.path.join(a.out, "rows", f"{sid}.pkl"), "wb"))
            if k % 10 == 0:
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    rows = []
    for s in sids:
        rows += pickle.load(open(os.path.join(a.out, "rows", f"{s}.pkl"), "rb"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "m1a_scores_per_unit.csv.gz"), index=False)
    print("done", len(rows), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
