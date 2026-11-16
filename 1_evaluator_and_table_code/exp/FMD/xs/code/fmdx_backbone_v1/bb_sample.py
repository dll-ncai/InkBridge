"""FMDX backbone sampler (FM-free): DDIM-50 eta 0 with the EMA weights of one milestone file bb_ms<k>.pt on the 224-px renders
of an E0-format set. Unit selection = e0_sample.py rule (--subset N --subset-seed s: sorted default_rng(s).choice(n, N);
--max-units N: first N; else all). Noise: torch.manual_seed(seed) once, units in index order, batches of --batch-size
(e0_sample.py convention). Output = e0_sample.py format (preds (N,320,4) float32 Nagamatsu space, unit_ids, meta json) so
e0_score.py e0 scores it unchanged; + <out>.json sidecar with sha256 values.
The render file: npz with img224 (M,224,224) uint8 black-on-white + unit_ids (rows looked up by unit id; may be a subset of
the gt npz, e.g. fmd_render224_val1000.npz), or an .npy holding every gt unit in gt order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE))
import uhwr_route_common as U  # noqa: E402
import bb_model as BM          # noqa: E402

CODE_FILES = ["bb_sample.py", "bb_model.py", "vendor/student_model.py", "vendor/uhwr_route_common.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True, help="npz with img224 + unit_ids, or .npy (N,224,224) in gt order")
    ap.add_argument("--gt-npz", type=Path, required=True)
    ap.add_argument("--weights", choices=("ema", "online"), default="ema")
    ap.add_argument("--sampling-timesteps", type=int, default=50)
    ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int); ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-units", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    unet, ck = BM.load_backbone_for_inference(a.ckpt, U, dev, a.weights)
    for m in ("transformers", "diffusers"):
        assert m not in sys.modules, f"{m} imported in the FM-free sampler"
    diff = U.build_diffusion(unet, sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta).to(dev).eval()
    G = np.load(a.gt_npz, allow_pickle=False)
    gt_ids = G["unit_ids"].astype(str)
    if a.render.suffix == ".npy":
        img = np.load(a.render, mmap_mode="r"); rid = gt_ids
        if img.shape[0] != len(gt_ids):
            raise SystemExit(".npy render must hold every gt unit in gt order")
    else:
        Rz = np.load(a.render, allow_pickle=False); img = Rz["img224"]; rid = Rz["unit_ids"].astype(str)
    rpos = {u: k for k, u in enumerate(rid)}            # render row of each unit (render npz may hold a subset, e.g. val1000)
    n_all = len(gt_ids)
    if a.subset is not None:
        sel = np.sort(np.random.default_rng(a.subset_seed).choice(n_all, min(a.subset, n_all), replace=False))
    elif a.max_units is not None:
        sel = np.arange(min(a.max_units, n_all))
    else:
        sel = np.arange(n_all)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = []
    with torch.no_grad():
        for b in range(0, len(sel), a.batch_size):
            ids = sel[b:b + a.batch_size]
            rows = [rpos[str(gt_ids[i])] for i in ids]         # KeyError = unit missing from the render file
            ims = torch.from_numpy((255.0 - np.asarray(img[rows], np.float32)) / 255.0).unsqueeze(1).to(dev)
            (x, _), _, _ = diff.sample(condition=ims, ground_truth=None, batch_size=len(ids), is_testing=True)
            out.append(x.transpose(1, 2).float().cpu().numpy())
    preds = np.concatenate(out).astype(np.float32)
    per_file, csha = U.code_sha256([HERE / f for f in CODE_FILES])
    meta = dict(kind="fmdx_bb_sample", ckpt=str(a.ckpt), ckpt_sha256=U.sha256_file(a.ckpt), milestone=int(ck["milestone"]), step=int(ck["step"]),
                arm_id=ck["config"].get("arm_id"), encoder=ck["config"].get("encoder"), weights=a.weights, fm_free=True,
                npz=str(a.gt_npz), gt_npz_sha256=U.sha256_file(a.gt_npz), render=str(a.render), render_sha256=U.sha256_file(a.render),
                sampler="DDIM", sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta, seed=a.seed, n_units=int(len(sel)),
                subset=a.subset, subset_seed=a.subset_seed if a.subset is not None else None, max_units=a.max_units, batch_size=a.batch_size,
                seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__,
                code_sha256=per_file, code_sha256_combined=csha, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_name(op.name[:-4] + ".part.npz")
    np.savez_compressed(tmp, preds=preds, unit_ids=gt_ids[sel], meta=json.dumps(meta))
    tmp.replace(op)
    U.write_json_atomic(str(op)[:-4] + ".json", dict(meta, preds_file=op.name, preds_sha256=U.sha256_file(op)))
    print(json.dumps({k: meta[k] for k in ("arm_id", "milestone", "n_units", "seconds", "device")}))


if __name__ == "__main__":
    main()
