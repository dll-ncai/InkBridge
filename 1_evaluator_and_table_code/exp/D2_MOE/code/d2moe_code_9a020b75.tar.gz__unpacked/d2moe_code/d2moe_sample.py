"""D2 inference: DDIM-sample trajectories with the frozen backbone + EMA (default) adapters/gate of one d2moe
checkpoint.  Output = e0_sample.py format (preds (N,L,4), unit_ids, meta) plus scripts, gate (N,E) softmax weights
per unit (the gate depends only on the condition image) and expert_names.

python d2moe_sample.py --run-dir RUN --milestone latest --backbone m5joint_ms9.pt --npz m5_val_ouhdl.npz \
    --sampling-timesteps 50 --batch-size 256 --seed 0 [--weights ema|online] [--force-expert NAME] --out preds.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uhwr_route_common as U  # noqa: E402
from d2moe_model import MoEUnet1D  # noqa: E402


def load_moe(ck_path, backbone_path, weights="ema", backbone_weights=None):
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    backbone, binfo = U.load_unet_from_checkpoint(backbone_path, which=backbone_weights or cfg["backbone_weights"])
    if binfo["param_sha256"] != cfg["backbone_param_sha256"]:
        raise SystemExit(f"backbone {backbone_path} does not match the checkpoint's backbone sha {cfg['backbone_param_sha256'][:12]}")
    moe = MoEUnet1D(backbone, n_experts=cfg["n_experts"], rank=cfg["rank"], alpha=cfg["lora_alpha"],
                    targets=tuple(cfg["lora_targets"].split(",")), gate_hidden=cfg["gate_hidden"], gate_feat=cfg["gate_feat"],
                    expert_names=cfg["expert_names"])
    sd = ck["ema_trainable"] if weights == "ema" else ck["trainable"]
    moe.load_trainable_state_dict(sd)
    return moe, ck, binfo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--milestone", default="latest")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--weights", choices=("ema", "online"), default="ema")
    ap.add_argument("--force-expert", help="route every unit to this expert (one-hot gate) instead of the learned gate")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--condition", default="rendered")
    ap.add_argument("--scripts", nargs="*")
    ap.add_argument("--script-override")
    ap.add_argument("--sampling-timesteps", type=int, default=50)
    ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-units", type=int)
    ap.add_argument("--subset", type=int)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.checkpoint is None:
        ms = U.latest_milestone(a.run_dir) if a.milestone == "latest" else int(a.milestone)
        ck_path = U.checkpoint_file(a.run_dir, ms)
    else:
        ck_path, ms = a.checkpoint, None
    moe, ck, binfo = load_moe(ck_path, a.backbone, a.weights)
    moe = moe.to(dev).eval()
    if a.force_expert is not None:
        e = moe.expert_names.index(a.force_expert)
        orig = moe.gate_from_condition

        def forced(classes):
            fm, g = orig(classes)
            g = torch.zeros_like(g)
            g[:, e] = 1.0
            return fm, g
        moe.gate_from_condition = forced
    diff = U.build_diffusion(moe, sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta).to(dev).eval()

    ds = U.MultiScriptDataset(a.npz, a.condition, scripts_keep=a.scripts, script_override=a.script_override)
    n_all = len(ds)
    if a.subset is not None:
        sel = np.sort(np.random.default_rng(a.subset_seed).choice(n_all, min(a.subset, n_all), replace=False))
    elif a.max_units is not None:
        sel = np.arange(min(a.max_units, n_all))
    else:
        sel = np.arange(n_all)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out, gates, t0 = [], [], time.time()
    with torch.no_grad():
        for b in range(0, len(sel), a.batch_size):
            items = [ds[int(i)] for i in sel[b:b + a.batch_size]]
            imgs = torch.stack([it["img"] for it in items]).to(dev)
            _, g = moe.gate_from_condition(imgs)
            gates.append(g.float().cpu().numpy())
            (x, _), _, _ = diff.sample(condition=imgs, ground_truth=None, batch_size=len(items), is_testing=True)
            out.append(x.transpose(1, 2).float().cpu().numpy())
    preds = np.concatenate(out)
    gates = np.concatenate(gates)
    meta = dict(kind="d2moe_sample", checkpoint=str(ck_path), checkpoint_sha256=U.sha256_file(ck_path), milestone=ms, step=int(ck["step"]),
                weights=a.weights, force_expert=a.force_expert, backbone=str(a.backbone), backbone_param_sha256=binfo["param_sha256"],
                expert_names=moe.expert_names, npz=a.npz, condition=a.condition, scripts=a.scripts,
                sampler="DDIM" if diff.is_ddim_sampling else "ancestral", sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta,
                seed=a.seed, n_units=int(len(sel)), subset=a.subset, subset_seed=a.subset_seed if a.subset else None, max_units=a.max_units,
                batch_size=a.batch_size, seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                mean_gate=gates.mean(0).round(5).tolist(),
                code_sha256_combined=U.code_sha256([HERE / "d2moe_train.py", HERE / "d2moe_model.py", HERE / "d2moe_sample.py", HERE / "uhwr_route_common.py"])[1])
    np.savez_compressed(a.out, preds=preds.astype(np.float32), unit_ids=ds.unit_ids[sel], scripts=ds.scripts[sel], gate=gates.astype(np.float32),
                        expert_names=np.array(moe.expert_names), meta=json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
