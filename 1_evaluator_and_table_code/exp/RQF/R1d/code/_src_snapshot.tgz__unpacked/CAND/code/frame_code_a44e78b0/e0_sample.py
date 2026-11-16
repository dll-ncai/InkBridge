"""E0 inference: DDIM-sample trajectories for every unit of an e0_<split>.npz with the EMA weights
of one checkpoint.  Model construction mirrors src/standard_diffusion_test.py (arguments read
from the run's run_config.json); sampling calls the repo's GaussianDiffusion1D.sample unchanged.

python e0_sample.py --run-dir RUN --milestone 10|latest --npz e0_test.npz --condition rendered \
    --sampling-timesteps 50 --ddim-eta 0 --batch-size 512 --seed 0 [--max-units N] --out preds.npz
Output: preds (N, L, 4) float32 in Nagamatsu space [x, y, pen, eos], unit_ids, meta json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA

from diffusion_models.ddpm_conditional import GaussianDiffusion1D, Unet1D
from diffusion_models.conditional_1d.runtime_utils import checkpoint_path, latest_checkpoint_milestone
from utils.casia_grid import TRAJECTORY_CHANNELS
from e0_dataset import OuhdlE0Dataset


def build(run_dir, sampling_timesteps, ddim_eta):
    a = json.load(open(Path(run_dir) / "run_config.json"))["arguments"]
    model = Unet1D(dim=int(a["model_dim"]), dim_mults=tuple(a["dim_mults"]), channels=TRAJECTORY_CHANNELS,
                   attn_dim_head=int(a["attn_dim_head"]), attn_heads=int(a["attn_heads"]),
                   cross_attn_dim_head=int(a["cross_attn_dim_head"]), cross_attn_heads=int(a["cross_attn_heads"]),
                   linear_attn_dim_head=int(a["linear_attn_dim_head"]), linear_attn_heads=int(a["linear_attn_heads"]),
                   attention_max_positions=int(a["attention_max_positions"]),
                   condition_image_size=int(a["image_size"]), encoder_hidden_dims=tuple(a["encoder_hidden_dims"]))
    diff = GaussianDiffusion1D(model, seq_length=int(a["sequence_length"]), timesteps=int(a["timesteps"]),
                               sampling_timesteps=sampling_timesteps, objective=a["objective"],
                               beta_schedule=a["beta_schedule"], ddim_sampling_eta=ddim_eta,
                               linear_beta_start=float(a["linear_beta_start"]), linear_beta_end=float(a["linear_beta_end"]),
                               cosine_schedule_s=float(a["cosine_schedule_s"]))
    return diff, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--milestone", default="latest")
    ap.add_argument("--npz", required=True); ap.add_argument("--condition", default="rendered")
    ap.add_argument("--sampling-timesteps", type=int, default=50); ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=512); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-units", type=int, help="first N units in index order")
    ap.add_argument("--subset", type=int, help="random N units (sorted indices), drawn with --subset-seed")
    ap.add_argument("--subset-seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1,
                    help="keep every nshards-th selected unit starting at shard (CPU fan-out); seeds are per shard")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ms = latest_checkpoint_milestone(a.run_dir) if a.milestone == "latest" else int(a.milestone)
    diff, targs = build(a.run_dir, a.sampling_timesteps, a.ddim_eta)
    assert diff.is_ddim_sampling == (a.sampling_timesteps < int(targs["timesteps"]))
    ck = torch.load(str(checkpoint_path(a.run_dir, ms)), map_location="cpu")
    ema = EMA(diff, beta=float(targs["ema_decay"]), update_every=int(targs["ema_update_every"]))
    ema.load_state_dict(ck["ema"])
    model = ema.ema_model.to(dev).eval()
    n_all = len(OuhdlE0Dataset(a.npz, a.condition))
    if a.subset is not None:
        sel = np.sort(np.random.default_rng(a.subset_seed).choice(n_all, min(a.subset, n_all), replace=False))
    elif a.max_units is not None:
        sel = np.arange(min(a.max_units, n_all))
    else:
        sel = None
    if a.nshards > 1:
        sel = (np.arange(n_all) if sel is None else sel)[a.shard::a.nshards]
    ds = OuhdlE0Dataset(a.npz, a.condition, indices=sel)
    torch.manual_seed(a.seed + 1000 * a.shard); np.random.seed(a.seed + 1000 * a.shard)
    out, t0 = [], time.time()
    with torch.no_grad():
        for b in range(0, len(ds), a.batch_size):
            items = [ds[i] for i in range(b, min(b + a.batch_size, len(ds)))]
            imgs = torch.stack([it["img"] for it in items]).to(dev)
            (x, _), _, _ = model.sample(condition=imgs, ground_truth=None, batch_size=len(items), is_testing=True)
            out.append(x.transpose(1, 2).float().cpu().numpy())
    preds = np.concatenate(out)
    meta = dict(run_dir=str(a.run_dir), milestone=ms, step=int(ck.get("step", -1)), npz=a.npz, condition=a.condition,
                sampler="DDIM" if diff.is_ddim_sampling else "ancestral", sampling_timesteps=a.sampling_timesteps,
                ddim_eta=a.ddim_eta, seed=a.seed, shard=a.shard, nshards=a.nshards, torch_seed=a.seed + 1000 * a.shard, n_units=len(ds), subset=a.subset, subset_seed=a.subset_seed if a.subset else None, max_units=a.max_units, batch_size=a.batch_size, weights="EMA",
                seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    np.savez_compressed(a.out, preds=preds.astype(np.float32), unit_ids=ds.unit_ids[ds.idx], meta=json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
