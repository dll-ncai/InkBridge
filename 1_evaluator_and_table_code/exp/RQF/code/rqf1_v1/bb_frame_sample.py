"""RQF R1c: FMDX backbone sampler (bb_sample.py logic, fmdx_backbone_v1) with frames on the 224-px render and Route A output layout.
Frame mirror_x = e0_frames.frame_cond shifted flip on the (224,224) uint8 black-on-white render (c -> 224-c, col 0 := white);
the sampled (N,320,4) output gets the exact inverse (x -> -x) so preds are in the original frame and score against the unmirrored GT.
Per (set, frame, seed): torch.manual_seed(seed); np.random.seed(seed) once, units in index order, fixed batches (frame_sample.py convention).
Files: <out>/preds/<set>__<label>_<frame>__seed<s>_preds.npz (+ .json sidecar usable by ORDER/code/lane_score.py)."""
import argparse, hashlib, json, os, sys, time, platform
from pathlib import Path
import numpy as np, torch
BB = os.environ["BB_DIR"]
sys.path.insert(0, BB + "/vendor"); sys.path.insert(0, BB); sys.path.insert(0, os.environ["FCODE"])
import uhwr_route_common as U
import bb_model as BM
import e0_frames as F

def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--label", required=True)
ap.add_argument("--set", required=True); ap.add_argument("--render", type=Path, required=True)
ap.add_argument("--render-key", default="render_224"); ap.add_argument("--gt-npz", type=Path, required=True)
ap.add_argument("--frames", nargs="+", default=["identity", "mirror_x"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--max-units", type=int)
ap.add_argument("--out-dir", required=True); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
unet, ck = BM.load_backbone_for_inference(a.ckpt, U, dev, "ema")
diff = U.build_diffusion(unet, sampling_timesteps=50, ddim_eta=0.0).to(dev).eval()
G = np.load(a.gt_npz, allow_pickle=False); ids = G["unit_ids"].astype(str)
Rz = np.load(a.render, allow_pickle=False); rid = Rz["unit_ids"].astype(str); img = Rz[a.render_key]
rpos = {u: k for k, u in enumerate(rid)}
sel = np.arange(len(ids) if a.max_units is None else min(a.max_units, len(ids)))
out = Path(a.out_dir) / "preds"; out.mkdir(parents=True, exist_ok=True)
ck_sha = sha256_file(a.ckpt); gt_sha = sha256_file(a.gt_npz); r_sha = sha256_file(a.render)
gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
here = Path(__file__).resolve().parent
code_sha = {p.name: sha256_file(p) for p in [Path(__file__), Path(BB) / "bb_model.py", Path(BB) / "vendor" / "uhwr_route_common.py", Path(F.__file__)]}
for fr in a.frames:
    for sd in a.seeds:
        label = f"{a.label}_{fr}"; name = f"{a.set}__{label}__seed{sd}_preds" + ("" if a.max_units is None else f"_smoke{a.max_units}")
        fn = out / (name + ".npz")
        if fn.is_file(): continue
        torch.manual_seed(sd); np.random.seed(sd); res = []; t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(sel), a.batch_size):
                ix = sel[b:b + a.batch_size]
                im = np.stack([img[rpos[str(ids[i])]] for i in ix])
                im = F.frame_cond(im, fr, background=255)
                ims = torch.from_numpy((255.0 - im.astype(np.float32)) / 255.0).unsqueeze(1).to(dev)
                (x, _), _, _ = diff.sample(condition=ims, ground_truth=None, batch_size=len(ix), is_testing=True)
                res.append(x.transpose(1, 2).float().cpu().numpy())
        raw = np.concatenate(res).astype(np.float32); preds = F.inverse_frame_preds(raw, fr)
        meta = dict(set=a.set, route_mode=label, specialist=label, label=a.label, checkpoint_label=a.label, frame=fr, frame_desc=F.describe(fr), inverse_applied=fr != "identity",
                    kind="rqf_bb_frame_sample", ckpt=str(a.ckpt), ckpt_sha256=ck_sha, milestone=int(ck["milestone"]), step=int(ck["step"]), arm_id=ck["config"].get("arm_id"),
                    npz=str(a.gt_npz), npz_sha256=gt_sha, render=str(a.render), render_sha256=r_sha, render_key=a.render_key, sampler="DDIM", sampling_timesteps=50, ddim_eta=0.0,
                    weights="EMA", seed=sd, batch_size=a.batch_size, n_units=int(len(sel)), seconds=round(time.time() - t0, 1), device=gpu, host=platform.node(),
                    torch=torch.__version__, code="bb_frame_sample.py", code_sha256=code_sha, max_units=a.max_units, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        tmp = out / (name + ".part.npz")
        np.savez_compressed(tmp, preds=preds, preds_in_frame=raw, unit_ids=ids[sel], unit_indices=sel.astype(np.int64), specialist=np.array([label] * len(sel)), meta=json.dumps(meta))
        os.replace(tmp, fn); meta["preds_sha256"] = sha256_file(fn); meta["preds_file"] = fn.name
        json.dump(meta, open(str(fn)[:-4] + ".json", "w"), indent=1)
        print(json.dumps({k: meta[k] for k in ("set", "frame", "seed", "n_units", "seconds", "device", "preds_sha256")}), flush=True)
