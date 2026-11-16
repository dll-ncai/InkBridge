"""Tests for e0_frames.py (run on the Mac, CPU; uhwr_eval 0.3.1 required on PYTHONPATH).

python test_e0_frames.py --npz e0_val.npz [--n 50] [--all-splits e0_train.npz e0_test.npz e0_gold.npz] --out test_e0_frames_report.json

T1  round trip frame∘inverse == identity on the first N units of the npz in stored order:
    bitwise on the (n,4) float32 seq, pixelwise on the uint8 condition, for every frame.
T2  edge-row/column background check (the shifted flip is lossless iff true) on every unit of every split given.
T3  renderer consistency: render_condition(frame_ink(native GT ink)) vs frame_cond(stored cond) — pixel
    mismatch statistics (informational; antialiasing is not exactly flip-symmetric), after first checking that
    render_condition(native GT ink) reproduces the stored cond_rendered.
T4  uhwr_eval 0.3.1 native protocol: decode(inverse(frame(seq))) scored against decode(seq) must give
    aiou = 1, kendall_tau = 1, direction_acc = 1 (and dtw = ldtw = 0) for every unit and frame; the same
    scored against the native GT ink (gt_points) is reported for context (= the encoding-oracle ceiling).
T5  padded (L=320) arrays with the collate rule (repeat last row): decode(inverse_frame_preds(frame_preds(P)))
    == decode(P) exactly; and decode(frame_seq(seq)) == frame_ink(decode(seq)) (array-level vs ink-level).
T6  dataset hook: OuhdlE0Dataset(frame=f)[i] equals the manual transform of item i; frame_scripts filter.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_adapter as A
import e0_frames as F
from e0_dataset import OuhdlE0Dataset
import uhwr_eval
from uhwr_eval import score

SEQ_LEN = 320


def pad_collate(seq, L=SEQ_LEN):
    """collate_fn rule of the repo: truncate to L or pad by repeating the final row."""
    a = np.asarray(seq, np.float32)
    if len(a) >= L:
        return a[:L].copy()
    return np.concatenate([a, np.repeat(a[-1:], L - len(a), 0)])


def ink_equal(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--all-splits", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rep = {"uhwr_eval_version": uhwr_eval.__version__, "npz": a.npz, "n_units": a.n, "frames": list(F.FRAMES), "tests": {}}
    assert uhwr_eval.__version__ == "0.3.1", uhwr_eval.__version__
    z = np.load(a.npz, allow_pickle=False)
    cond, seq, off = z["cond_rendered"], z["seq"], z["seq_offsets"]
    ids = z["unit_ids"].astype(str)
    N = min(a.n, len(ids))
    rep["unit_ids"] = ids[:N].tolist()

    # ---- T1 round trip
    t1 = {f: {"seq_bitwise_ok": 0, "cond_pixelwise_ok": 0, "seq_max_abs_diff": 0.0, "cond_n_pixels_diff": 0} for f in F.FRAMES}
    for i in range(N):
        s = seq[off[i]:off[i + 1]]
        c = cond[i]
        for f in F.FRAMES:
            s2 = F.inverse_frame_seq(F.frame_seq(s, f), f)
            c2 = F.inverse_frame_cond(F.frame_cond(c, f), f)
            ok_s = s2.dtype == s.dtype and s2.shape == s.shape and np.array_equal(s2.view(np.uint32), s.view(np.uint32))
            ok_c = np.array_equal(c2, c)
            t1[f]["seq_bitwise_ok"] += int(ok_s)
            t1[f]["cond_pixelwise_ok"] += int(ok_c)
            t1[f]["seq_max_abs_diff"] = max(t1[f]["seq_max_abs_diff"], float(np.abs(s2.astype(np.float64) - s).max()))
            t1[f]["cond_n_pixels_diff"] += int((c2 != c).sum())
    for f in F.FRAMES:
        t1[f]["pass"] = t1[f]["seq_bitwise_ok"] == N and t1[f]["cond_pixelwise_ok"] == N
    rep["tests"]["T1_round_trip"] = {"n": N, **t1, "pass": all(t1[f]["pass"] for f in F.FRAMES)}

    # ---- T2 edge background on all splits
    t2 = {}
    for p in [a.npz] + list(a.all_splits):
        zz = np.load(p, allow_pickle=False)
        for key in [k for k in zz.files if k in ("cond_rendered", "cond_dataset")]:
            cc = zz[key]
            t2[f"{Path(p).name}:{key}"] = {"n": int(len(cc)),
                                           "row0_clean": int(F.edge_clean(cc, "vt").sum()),
                                           "col0_clean": int(F.edge_clean(cc, "mirror_x").sum()),
                                           "row63_clean": int((cc[:, 63, :] == 255).all(1).sum()),
                                           "col63_clean": int((cc[:, :, 63] == 255).all(1).sum()),
                                           "min_ink_row": int(np.flatnonzero((cc < 255).any((0, 2)))[0]),
                                           "max_ink_row": int(np.flatnonzero((cc < 255).any((0, 2)))[-1]),
                                           "min_ink_col": int(np.flatnonzero((cc < 255).any((0, 1)))[0]),
                                           "max_ink_col": int(np.flatnonzero((cc < 255).any((0, 1)))[-1])}
    rep["tests"]["T2_edge_background"] = {**t2, "pass_rendered": all(v["row0_clean"] == v["n"] and v["col0_clean"] == v["n"] for k, v in t2.items() if k.endswith("cond_rendered"))}

    # ---- T3 renderer consistency (needs native GT ink)
    if "gt_points" in z.files:
        gp, gs, go = z["gt_points"], z["gt_stroke_id"], z["gt_offsets"]
        t3 = {"stored_cond_reproduced": 0, "n": N}
        for f in ("mirror_x", "vt"):
            t3[f] = {"n_pixels_diff_total": 0, "n_pixels_diff_max": 0, "n_units_exact": 0, "abs_diff_sum": 0}
        for i in range(N):
            ink = A.unpack_ink(gp, gs, go, i)
            r0 = A.render_condition(ink)
            t3["stored_cond_reproduced"] += int(np.array_equal(r0, cond[i]))
            for f in ("mirror_x", "vt"):
                r1 = A.render_condition(F.frame_ink(ink, f, A.CANVAS))
                c1 = F.frame_cond(cond[i], f)
                d = r1.astype(int) - c1.astype(int)
                nd = int((d != 0).sum())
                t3[f]["n_pixels_diff_total"] += nd
                t3[f]["n_pixels_diff_max"] = max(t3[f]["n_pixels_diff_max"], nd)
                t3[f]["n_units_exact"] += int(nd == 0)
                t3[f]["abs_diff_sum"] += int(np.abs(d).sum())
        # the same statistic for the WRONG (unshifted) flip, as the control that the shifted flip is the consistent one
        t3["control_plain_flip"] = {}
        for f, fn in (("mirror_x", np.fliplr), ("vt", np.flipud)):
            nd_tot = 0
            for i in range(N):
                ink = A.unpack_ink(gp, gs, go, i)
                r1 = A.render_condition(F.frame_ink(ink, f, A.CANVAS))
                nd_tot += int((r1 != fn(cond[i])).sum())
            t3["control_plain_flip"][f] = {"n_pixels_diff_total": nd_tot}
        rep["tests"]["T3_renderer_consistency"] = t3

    # ---- T4 uhwr_eval metrics of the inverted transformed GT vs original GT.
    # NOTE (measured): uhwr_eval 0.3.1 adaptive IoU renders GT at thickness 2 and the prediction at thickness 1
    # (then dilates), so AIoU(ink, ink) < 1 by construction (0.90-0.99 on OUHD-L units).  The attainable ceiling
    # is the self-score of the original GT; the criterion is therefore: aiou == self-aiou of the same unit,
    # kendall_tau == 1 (|1 - tau| < 1e-9), direction_acc == 1 (or NaN when no eligible stroke), dtw == ldtw == 0.
    t4 = {f: {"n": N, "aiou_equals_self": 0, "aiou_min": 1.0, "self_aiou_min": 1.0, "self_aiou_mean": 0.0,
              "tau_min": 1.0, "direction_min": 1.0, "dtw_max": 0.0, "ldtw_max": 0.0, "n_all_ok": 0, "direction_nan": 0, "tau_nan": 0}
          for f in F.FRAMES}
    t4_native = {f: {"aiou_mean": 0.0, "tau_mean": 0.0, "direction_mean": 0.0} for f in F.FRAMES}
    for i in range(N):
        s = seq[off[i]:off[i + 1]]
        gt_dec = A.decode(s)
        r_self = score(gt_dec, gt_dec, A.CANVAS, protocol="native", with_lpips=False)
        gt_nat = A.unpack_ink(z["gt_points"], z["gt_stroke_id"], z["gt_offsets"], i) if "gt_points" in z.files else None
        for f in F.FRAMES:
            pr = A.decode(F.inverse_frame_seq(F.frame_seq(s, f), f))
            r = score(gt_dec, pr, A.CANVAS, protocol="native", with_lpips=False)
            d_ok = np.isnan(r["direction_acc"]) or r["direction_acc"] == 1.0
            tau_ok = np.isnan(r["kendall_tau"]) or abs(1.0 - r["kendall_tau"]) < 1e-9
            t4[f]["tau_nan"] += int(np.isnan(r["kendall_tau"]))
            t4[f]["direction_nan"] += int(np.isnan(r["direction_acc"]))
            t4[f]["aiou_equals_self"] += int(r["aiou"] == r_self["aiou"])
            t4[f]["aiou_min"] = min(t4[f]["aiou_min"], r["aiou"]); t4[f]["self_aiou_min"] = min(t4[f]["self_aiou_min"], r_self["aiou"])
            t4[f]["self_aiou_mean"] += r_self["aiou"] / N
            t4[f]["tau_min"] = min(t4[f]["tau_min"], r["kendall_tau"]) if not np.isnan(r["kendall_tau"]) else t4[f]["tau_min"]
            if not np.isnan(r["direction_acc"]):
                t4[f]["direction_min"] = min(t4[f]["direction_min"], r["direction_acc"])
            t4[f]["dtw_max"] = max(t4[f]["dtw_max"], r["dtw"]); t4[f]["ldtw_max"] = max(t4[f]["ldtw_max"], r["ldtw"])
            t4[f]["n_all_ok"] += int(r["aiou"] == r_self["aiou"] and tau_ok and d_ok and r["dtw"] == 0.0 and r["ldtw"] == 0.0)
            if gt_nat is not None:
                rn = score(gt_nat, pr, A.CANVAS, protocol="native", with_lpips=False)
                for k, m in (("aiou", "aiou"), ("tau", "kendall_tau"), ("direction", "direction_acc")):
                    t4_native[f].setdefault(f"{k}_vals", []).append(float(rn[m]))
    for f in F.FRAMES:
        for k in ("aiou", "tau", "direction"):
            v = np.array(t4_native[f].pop(f"{k}_vals", [np.nan]))
            t4_native[f][f"{k}_mean"] = float(np.nanmean(v)); t4_native[f][f"{k}_n_defined"] = int(np.isfinite(v).sum())
    for f in F.FRAMES:
        t4[f]["pass"] = t4[f]["n_all_ok"] == N
    rep["tests"]["T4_metrics_vs_original_GT"] = {**t4, "criterion": "aiou == self-score of the original GT (metric ceiling, <1 by construction), |1-tau|<1e-9, direction 1 or NaN, dtw = ldtw = 0",
                                                "vs_native_gt_points_context": t4_native, "pass": all(t4[f]["pass"] for f in F.FRAMES)}

    # ---- T5 padded arrays and array/ink equivalence
    t5 = {f: {"padded_roundtrip_ok": 0, "array_vs_ink_ok": 0} for f in F.FRAMES}
    for i in range(N):
        s = seq[off[i]:off[i + 1]]
        P = pad_collate(s)[None]
        for f in F.FRAMES:
            Q = F.inverse_frame_preds(F.frame_preds(P, f), f)
            t5[f]["padded_roundtrip_ok"] += int(np.array_equal(Q, P) and ink_equal(A.decode(Q[0]), A.decode(P[0])))
            d1 = A.decode(F.frame_seq(s, f))
            d2 = F.frame_ink(A.decode(s), f, A.CANVAS)
            t5[f]["array_vs_ink_ok"] += int(len(d1) == len(d2) and all(np.allclose(x, y, atol=1e-9) for x, y in zip(d1, d2)))
    for f in F.FRAMES:
        t5[f]["pass"] = t5[f]["padded_roundtrip_ok"] == N and t5[f]["array_vs_ink_ok"] == N
    rep["tests"]["T5_padded_and_ink_equivalence"] = {"n": N, **t5, "pass": all(t5[f]["pass"] for f in F.FRAMES)}

    # ---- T6 dataset hook
    t6 = {}
    for f in F.FRAMES:
        ds = OuhdlE0Dataset(a.npz, "rendered", indices=np.arange(N), frame=f)
        ok = 0
        for i in range(N):
            it = ds[i]
            img_ref = (255 - F.frame_cond(cond[i], f).astype(np.float32)) / 255.0
            seq_ref = F.frame_seq(seq[off[i]:off[i + 1]], f)
            ok += int(np.array_equal(it["img"].numpy()[0], img_ref) and np.array_equal(it["coords"].numpy(), seq_ref))
        t6[f] = {"items_match_manual_transform": ok}
    # frame_scripts filter: tag everything 'ouhdl', restrict to {'iam'} -> identity everywhere
    ds0 = OuhdlE0Dataset(a.npz, "rendered", indices=np.arange(N), frame="vt", frame_scripts={"iam"}, script_tag="ouhdl")
    ds1 = OuhdlE0Dataset(a.npz, "rendered", indices=np.arange(N), frame="vt", frame_scripts={"ouhdl"}, script_tag="ouhdl")
    id_ok = sum(int(np.array_equal(ds0[i]["coords"].numpy(), seq[off[i]:off[i + 1]]) and ds0.unit_frame(i) == "identity") for i in range(N))
    vt_ok = sum(int(np.array_equal(ds1[i]["coords"].numpy(), F.frame_seq(seq[off[i]:off[i + 1]], "vt")) and ds1.unit_frame(i) == "vt") for i in range(N))
    t6["frame_scripts_filter"] = {"excluded_tag_identity_ok": id_ok, "included_tag_vt_ok": vt_ok}
    rep["tests"]["T6_dataset_hook"] = {"n": N, **t6, "pass": all(t6[f]["items_match_manual_transform"] == N for f in F.FRAMES) and id_ok == N and vt_ok == N}

    rep["pass"] = all(v.get("pass", True) for k, v in rep["tests"].items() if k != "T3_renderer_consistency") and rep["tests"]["T2_edge_background"]["pass_rendered"]
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v.get("pass", "info") for k, v in rep["tests"].items()}), "PASS" if rep["pass"] else "FAIL")


if __name__ == "__main__":
    main()
