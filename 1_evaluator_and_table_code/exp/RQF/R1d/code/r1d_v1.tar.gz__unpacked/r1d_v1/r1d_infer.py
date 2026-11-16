#!/usr/bin/env python3
"""RQF A2.5 R1d: label-free frame selection over the existing mirror_x frame (frame_code_a44e78b0, CODE_TAG 5dd59e88; E137 / CAND K1)
and the frozen endpoint-agreement verifier (ORDER/code/verify.py sha256 579c9b22, vendored here unchanged, sha asserted at import).
Per unit: k = 3 samples in identity and k = 3 in mirror_x (mirror preds stored in the ORIGINAL frame, inverse applied by the sampler).
  frame score  = mean pairwise endpoint agreement verify.pu() over the 3 unordered sample pairs of that frame (no GT read);
  frame choice = mirror_x iff round(score_mirror, 9) > round(score_identity, 9); tie -> identity;
  output       = verify.select() sample of the chosen frame (rows copied bit-exact).
Also written: idv3 = verify.select() over the identity pool (the identity-k3-verifier comparator) and mirv3 = same over the mirror pool
(oracle-frame is formed at read time from idv3/mirv3 rows). Inputs/outputs carry sha256; GT npz is used only to align unit order and for
the verifier's ink-F tie break on the model's own 64-px condition image (cond_rendered), as in the A1 sealed read.
Library use (stacked read):  from r1d_infer import r1d_unit, frame_score;  CLI:  r1d_infer.py --plan PLAN.json --base BASE [--sets ...]"""
import argparse, csv, hashlib, json, os, sys, time, platform
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VERIFY_SHA = "579c9b220434c3d80ce23ddb90923addf6081fd70acc9ae9461b1b8086de2e22"
def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
assert sha(os.path.join(HERE, "verify.py")) == VERIFY_SHA, "vendored verify.py is not 579c9b22"
import verify as V

def frame_score(cands):
    """mean of verify.pu over the unordered pairs of a frame's candidates (pu is symmetric, so = mean of select()'s u)."""
    S = [V.strokes(a) for a in cands]; n = len(S)
    if n < 2: return 0.0
    return float(np.mean([V.pu(S[i], S[j]) for i in range(n) for j in range(i + 1, n)]))

def r1d_unit(id_cands, mir_cands, img):
    """-> dict(frame 0=identity/1=mirror_x, score_id, score_mir, sel_id, sel_mir, sel (index in the chosen pool))."""
    s_id, s_mir = frame_score(id_cands), frame_score(mir_cands)
    k_id = V.select(id_cands, img)[0]; k_mir = V.select(mir_cands, img)[0]
    fr = int(round(s_mir, 9) > round(s_id, 9))
    return dict(frame=fr, score_id=s_id, score_mir=s_mir, sel_id=int(k_id), sel_mir=int(k_mir), sel=int(k_mir if fr else k_id))

def load_pool(files, gt_ids):
    """load the seed files; rows re-indexed to the GT unit order (asserting the same unit set)."""
    P, meta = [], []
    for f in files:
        z = np.load(f, allow_pickle=False); u = z["unit_ids"].astype(str); p = np.asarray(z["preds"])
        assert len(u) == len(gt_ids) and set(u) == set(gt_ids), ("unit set", f)
        if not (u == gt_ids).all():
            pos = {x: i for i, x in enumerate(u)}; p = p[[pos[x] for x in gt_ids]]
        P.append(p)
        sc = f[:-4] + ".json"; side = json.load(open(sc)) if os.path.exists(sc) else {}
        meta.append(dict(path=f, sha256=sha(f), device=side.get("device"), code=side.get("code") or side.get("kind"),
                         ckpt_sha256=side.get("checkpoint_sha256") or side.get("ckpt_sha256"), batch_size=side.get("batch_size"), seed=side.get("seed")))
    return P, meta

def run_set(base, setname, ent, out_dir, score_dir, prov_extra):
    gt = ent["gt"]; G = np.load(gt, allow_pickle=False); ids = G["unit_ids"].astype(str); cond = G["cond_rendered"]
    PI, mi = load_pool(ent["id_pool"], ids); PM, mm = load_pool(ent["mir_pool"], ids)
    n = len(ids); t0 = time.time()
    rec = [r1d_unit([p[k] for p in PI], [p[k] for p in PM], cond[k]) for k in range(n)]
    fr = np.array([r["frame"] for r in rec]); sid = np.array([r["sel_id"] for r in rec]); smi = np.array([r["sel_mir"] for r in rec])
    SI = np.stack(PI); SM = np.stack(PM); ar = np.arange(n)
    W = {"idv3": SI[sid, ar], "mirv3": SM[smi, ar], "r1d": np.where(fr[:, None, None] == 1, SM[smi, ar], SI[sid, ar])}
    for k in range(n):   # bit-exact copy check
        assert np.array_equal(W["r1d"][k], SM[smi[k], k] if fr[k] else SI[sid[k], k])
    os.makedirs(out_dir, exist_ok=True); os.makedirs(score_dir, exist_ok=True); gsha = sha(gt); outs = {}
    devs = sorted({str(m["device"]) for m in mi + mm})
    for tag, arr in W.items():
        lab = f"{base}_{tag}"; fn = os.path.join(out_dir, f"{setname}__{lab}__seed0_preds.npz")
        np.savez_compressed(fn, preds=arr.astype(np.float32), unit_ids=ids, unit_indices=ar.astype(np.int64), specialist=np.array([lab] * n),
                            frame_choice=fr.astype(np.int8), sel_identity=sid.astype(np.int8), sel_mirror=smi.astype(np.int8))
        hs = sha(fn)
        side = dict(set=setname, npz_clusterA=gt, source_sha256_expected=gsha, route_mode=lab, seed=0, preds_sha256=hs, label=lab, kind="rqf_r1d",
                    r1d=dict(tag=tag, rule="A2.5 R1d", code_sha256={f: sha(os.path.join(HERE, f)) for f in ("r1d_infer.py", "verify.py")},
                             id_pool=mi, mir_pool=mm, n_units=n, n_mirror_chosen=int(fr.sum()), frac_mirror=float(fr.mean()),
                             sel_identity_counts=np.bincount(sid, minlength=3).tolist(), sel_mirror_counts=np.bincount(smi, minlength=3).tolist(),
                             device_mixed=len(devs) > 1, devices=devs), **prov_extra)
        json.dump(side, open(fn[:-4] + ".json", "w"), indent=1)
        d = os.path.join(score_dir, os.path.basename(fn))
        for x, y in ((fn, d), (fn[:-4] + ".json", d[:-4] + ".json")):
            if os.path.lexists(y): os.remove(y)
            os.symlink(os.path.realpath(x), y)
        outs[tag] = dict(path=fn, sha256=hs)
    with open(os.path.join(out_dir, f"{setname}__{base}_choices.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unit_id", "frame", "score_identity", "score_mirror", "sel_identity", "sel_mirror"])
        for u, r in zip(ids, rec): w.writerow([u, r["frame"], f"{r['score_id']:.9f}", f"{r['score_mir']:.9f}", r["sel_id"], r["sel_mir"]])
    print(json.dumps(dict(base=base, set=setname, n=n, n_mirror=int(fr.sum()), devices=devs, sec=round(time.time() - t0, 1))), flush=True)
    return dict(outputs=outs, n_units=n, n_mirror_chosen=int(fr.sum()), devices=devs)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--sets", nargs="*"); a = ap.parse_args()
    P = json.load(open(a.plan)); B = P["bases"][a.base]; root = P["root"]
    missing = [f for s, e in B["sets"].items() for f in e["id_pool"] + e["mir_pool"] if not os.path.exists(f)]
    if missing: print("MISSING pool files", len(missing), missing[:5], flush=True); sys.exit(2)
    prov_extra = dict(agent=P["agent"], lead_reason=P["lead_reason"], prereg=P["prereg"])
    res = {s: run_set(a.base, s, B["sets"][s], f"{root}/apply/{a.base}", f"{root}/score_in/{a.base}", prov_extra) for s in (a.sets or list(B["sets"]))}
    prov = dict(kind="rqf_r1d_apply", base=a.base, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), plan=a.plan, plan_sha256=sha(a.plan), results=res, **prov_extra)
    json.dump(prov, open(f"{root}/apply/{a.base}/PROVENANCE_apply_{a.base}.json", "w"), indent=1)
    open(f"{root}/score_in/{a.base}/tasks.txt", "w").write("\n".join(f"{s}__{a.base}_*_preds.npz" for s in res) + "\n")

if __name__ == "__main__":
    main()
