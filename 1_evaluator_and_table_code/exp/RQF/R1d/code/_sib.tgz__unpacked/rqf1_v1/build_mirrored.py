"""RQF R1a: build mirrored left-to-right test sets (x -> -x on trajectory, shifted flip c -> 64-c on the 64x64 condition,
gt ink x -> 64-x) from E0-format npz files. Same unit_ids/sample_ids/writer_ids as the source. Also checks the 224-px
shifted flip (c -> 224-c) against a re-render of the mirrored ink for the R1c backbone arms. Writes PROVENANCE json."""
import argparse, hashlib, json, os, sys, time
import numpy as np
import e0_frames as F
import e0_adapter as A

def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

SRC = {"iam500": "data/m5_multiscript_v1/m5_test_iam500.npz", "casia_en": "exp/X4a/data/x4a_casia_en.npz",
       "hpl_tamil": "exp/X4a/data/x4a_hpl_tamil.npz"}
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nrender", type=int, default=100); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
prov = dict(kind="mirrored_ltr_sets", experiment="RQF R1a", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            transform="mirror_x: seq x->-x (e0_frames.frame_seq); cond_rendered shifted flip out[:,0]=255,out[:,i]=in[:,64-i] (e0_frames.frame_cond); gt_points x->64-x (e0_frames.frame_ink convention); other cond_* keys dropped; unit order, ids, writers unchanged",
            seed="none (deterministic transform)", code_tag=open(os.path.join(os.path.dirname(F.__file__), "CODE_TAG.txt")).read().strip() if os.path.exists(os.path.join(os.path.dirname(F.__file__), "CODE_TAG.txt")) else None, sets={})
for name, rel in SRC.items():
    src = os.path.join(a.stage, rel); z = np.load(src, allow_pickle=False); d = {k: z[k] for k in z.files}
    cond = d["cond_rendered"]; n = len(d["unit_ids"])
    col0_bg = int((cond[:, :, 0] == 255).all(axis=1).sum())
    out = {k: v for k, v in d.items() if not k.startswith("cond_") or k == "cond_rendered"}
    out["cond_rendered"] = F.frame_cond(cond, "mirror_x")
    seq, off = d["seq"], d["seq_offsets"]; ms = seq.copy()
    for i in range(n):
        ms[off[i]:off[i + 1]] = F.frame_seq(seq[off[i]:off[i + 1]], "mirror_x")
    out["seq"] = ms
    gp = d["gt_points"].copy(); gp[:, 0] = 64.0 - gp[:, 0]; out["gt_points"] = gp
    # involution checks
    ok_cond = bool(np.array_equal(F.frame_cond(out["cond_rendered"], "mirror_x"), cond))
    ok_seq = bool(np.array_equal(np.where(True, -ms[:, 0], 0), seq[:, 0]))
    ok_gt = float(np.abs((64.0 - gp[:, 0]) - d["gt_points"][:, 0]).max())
    # render check: mirrored gt ink rendered vs flipped stored condition (and unmirrored baseline)
    m = min(a.nrender, n); ex = dif = 0; dif0 = 0.0; ex0 = 0
    for i in range(m):
        r_m = A.render_condition(A.unpack_ink(gp, d["gt_stroke_id"], d["gt_offsets"], i)).astype(int)
        r_o = A.render_condition(A.unpack_ink(d["gt_points"], d["gt_stroke_id"], d["gt_offsets"], i)).astype(int)
        ex += int(np.array_equal(r_m, out["cond_rendered"][i])); dif += int(np.abs(r_m - out["cond_rendered"][i]).sum())
        ex0 += int(np.array_equal(r_o, cond[i])); dif0 += int(np.abs(r_o - cond[i]).sum())
    dst = os.path.join(a.out, f"rqf_mir_{name}.npz"); tmp = dst + ".part.npz"
    np.savez_compressed(tmp, **out); os.replace(tmp, dst)
    wr = d["writer_ids"]
    prov["sets"][name] = dict(source=src, source_sha256=sha(src), output=dst, output_sha256=sha(dst), n_units=int(n),
        n_writers=int(len(np.unique(wr))), n_sample_ids=int(len(np.unique(d["sample_ids"]))), dropped_keys=[k for k in d if k.startswith("cond_") and k != "cond_rendered"],
        cond_col0_background_units=col0_bg, involution_cond_ok=ok_cond, involution_seq_ok=ok_seq, involution_gt_max_abs=ok_gt,
        render_check=dict(n=m, mirrored_render_equals_flipped_cond=ex, abs_diff_sum_mirrored=dif, unmirrored_render_equals_stored_cond=ex0, abs_diff_sum_unmirrored=dif0))
    print(name, json.dumps(prov["sets"][name])[:400], flush=True)
# 224-px frame check on OUHD-L test renders
try:
    from uhwr_eval import render_image
    zt = np.load(os.path.join(a.stage, "data/ouhdl_e0_v3/e0_test.npz"), allow_pickle=False)
    R = np.load(os.path.join(a.stage, "exp/FMD/data/renders_224/test.npz"), allow_pickle=False)
    assert (zt["unit_ids"].astype(str) == R["unit_ids"].astype(str)).all()
    ex = dif = 0; m = 50
    for i in range(m):
        ink = A.unpack_ink(zt["gt_points"], zt["gt_stroke_id"], zt["gt_offsets"], i)
        mir = [np.c_[64.0 - s[:, 0], s[:, 1]] * 3.5 for s in ink]
        r = render_image(mir, (224, 224), 7, antialias=True).astype(int)
        f = F.frame_cond(R["render_224"][i], "mirror_x").astype(int)
        ex += int(np.array_equal(r, f)); dif += int(np.abs(r - f).sum())
    prov["render224_check"] = dict(set="test_ouhdl", n=m, exact=ex, abs_diff_sum=dif, note="shifted flip c->224-c vs re-render of mirrored ink at f=3.5, thickness 7")
    print(prov["render224_check"], flush=True)
except Exception as e:
    prov["render224_check"] = dict(error=f"{type(e).__name__}: {e}"); print(prov["render224_check"], flush=True)
json.dump(prov, open(os.path.join(a.out, "PROVENANCE_R1_mirrored_sets.json"), "w"), indent=1)
