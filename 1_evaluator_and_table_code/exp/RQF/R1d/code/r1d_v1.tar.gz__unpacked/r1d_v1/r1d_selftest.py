#!/usr/bin/env python3
"""R1d unit test (20 units per set, 7 test sets). Exit 0 iff every gated test passes. Writes OUT/selftest_report.json.
T1 mirrored condition renders bit-exact: frame_cond(img,'mirror_x')[:, c] == img[:, W-c] (c>=1), col 0 background, edge clean,
   involution frame_cond(frame_cond(img)) == img; on the 64-px cond_rendered of every set and on the 224-px renders (warm backbones).
T2 (informational, as E137 T3) renderer consistency render(frame_ink(GT ink)) vs mirrored stored render: pixel-diff counts.
T3 preds inverse bit-exact on the K1 LUO mirror_x file: inverse(preds_in_frame) == preds and inverse(preds) == preds_in_frame.
T4 deterministic choice: r1d_unit twice on 20 real units (LUO ms6 K1 identity/mirror pools) -> identical dicts; identical pools in both
   frames -> identity on all units (tie rule); swapped pools -> choice flips on every non-tied unit; chosen rows bit-exact copies."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r1d_infer as R
import e0_frames as F
P = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True); N = 20
rep = dict(tests={}, n_per_set=N); ok = True
def shifted_ok(img, m, W):
    return bool(np.array_equal(m[..., :, 1:], img[..., :, W - np.arange(1, W)]) and (m[..., :, 0] == 255).all())
# T1
t1 = {}
for s, e in P["bases"]["LUO_ms6"]["sets"].items():
    z = np.load(e["gt"], allow_pickle=False); c = z["cond_rendered"][:N]; m = F.frame_cond(c, "mirror_x")
    t1[f"{s}_64"] = dict(shifted_flip_exact=shifted_ok(c, m, 64), edge_clean=int(F.edge_clean(c, "mirror_x").sum()),
                         involution_exact=bool(np.array_equal(F.frame_cond(m, "mirror_x"), c)))
for s, e in P["bases"]["NOFM_W_bb"]["sets"].items():
    rz = np.load(e["render"], allow_pickle=False); img = rz["img224"][:N]; m = F.frame_cond(img, "mirror_x", background=255)
    t1[f"{s}_224"] = dict(shifted_flip_exact=shifted_ok(img, m, 224), edge_clean=int(F.edge_clean(img, "mirror_x", 255).sum()),
                          involution_exact=bool(np.array_equal(F.frame_cond(m, "mirror_x", background=255), img)))
p1 = all(v["shifted_flip_exact"] and v["involution_exact"] and v["edge_clean"] == N for v in t1.values())
rep["tests"]["T1_mirrored_renders_bit_exact"] = dict(per_set=t1, n_sets=len(t1), pass_=p1); ok &= p1
# T2 informational
try:
    import e0_adapter as A
    t2 = {}
    for s, e in P["bases"]["LUO_ms6"]["sets"].items():
        z = np.load(e["gt"], allow_pickle=False); nd = ne = 0
        for i in range(N):
            ink = A.unpack_ink(z["gt_points"], z["gt_stroke_id"], z["gt_offsets"], i)
            r1 = A.render_condition(F.frame_ink(ink, "mirror_x", A.CANVAS)); c1 = F.frame_cond(z["cond_rendered"][i], "mirror_x")
            d = int((r1 != c1).sum()); nd += d; ne += int(d == 0)
        t2[s] = dict(n_pixels_diff_total=nd, n_units_exact=ne)
    rep["tests"]["T2_renderer_consistency_informational"] = t2
except Exception as ex:
    rep["tests"]["T2_renderer_consistency_informational"] = dict(error=f"{type(ex).__name__}: {ex}")
# T3
kf = P["bases"]["LUO_ms6"]["sets"]["test_ouhdl"]["mir_pool"][0]; z = np.load(kf, allow_pickle=False)
pr, raw = z["preds"][:N], z["preds_in_frame"][:N]
p3 = bool(np.array_equal(F.inverse_frame_preds(raw, "mirror_x"), pr) and np.array_equal(F.inverse_frame_preds(pr, "mirror_x"), raw))
rep["tests"]["T3_preds_inverse_bit_exact"] = dict(file=kf, sha256=R.sha(kf), pass_=p3); ok &= p3
# T4
e = P["bases"]["LUO_ms6"]["sets"]["test_ouhdl"]; G = np.load(e["gt"], allow_pickle=False); ids = G["unit_ids"].astype(str); cond = G["cond_rendered"]
PI, _ = R.load_pool(e["id_pool"], ids); PM, _ = R.load_pool(e["mir_pool"], ids)
a = [R.r1d_unit([p[k] for p in PI], [p[k] for p in PM], cond[k]) for k in range(N)]
b = [R.r1d_unit([p[k] for p in PI], [p[k] for p in PM], cond[k]) for k in range(N)]
tie = [R.r1d_unit([p[k] for p in PI], [p[k] for p in PI], cond[k])["frame"] for k in range(N)]
sw = [R.r1d_unit([p[k] for p in PM], [p[k] for p in PI], cond[k])["frame"] for k in range(N)]
nt = [k for k in range(N) if round(a[k]["score_id"], 9) != round(a[k]["score_mir"], 9)]
fr = np.array([r["frame"] for r in a]); SI = np.stack(PI)[:, :N]; SM = np.stack(PM)[:, :N]; ar = np.arange(N)
Wr = np.where(fr[:, None, None] == 1, SM[[r["sel_mir"] for r in a], ar], SI[[r["sel_id"] for r in a], ar])   # run_set indexing path
copy_ok = all(np.array_equal(Wr[k], (PM if r["frame"] else PI)[r["sel"]][k]) for k, r in enumerate(a))
p4 = bool(a == b and all(t == 0 for t in tie) and all(sw[k] == 1 - a[k]["frame"] for k in nt) and copy_ok)
rep["tests"]["T4_deterministic_choice"] = dict(repeat_identical=a == b, tie_to_identity=all(t == 0 for t in tie), n_untied=len(nt),
    swap_flips=all(sw[k] == 1 - a[k]["frame"] for k in nt), n_mirror_chosen=int(sum(r["frame"] for r in a)),
    choices=[[ids[k], r["frame"], round(r["score_id"], 6), round(r["score_mir"], 6), r["sel"]] for k, r in enumerate(a)], pass_=p4); ok &= p4
rep["pass"] = bool(ok); rep["code_sha256"] = {f: R.sha(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sh"))}
json.dump(rep, open(f"{OUT}/selftest_report.json", "w"), indent=1)
print(json.dumps({k: v.get("pass_", "info") if isinstance(v, dict) else v for k, v in rep["tests"].items()}), "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
