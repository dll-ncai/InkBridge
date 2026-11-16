"""SM-A inference: frozen backbone + the EMA adapter of one sma_train checkpoint; per-unit gate g in {0,1}:
--route all          g = 1 for every unit (target-val milestone selection on val1000)
--route manifest CSV g = 1 iff the D3 ScriptCNN router (script_clf.pt sha256 22bb8802...) gives p_rtl >= 0.5 for the unit
                     (exp/D3_ROUTE/router/manifest_<set>.csv); every unit of the set must be in the manifest.
g = 0 multiplies the LoRA delta by 0, i.e. the exact backbone forward.  DDIM-50, eta 0, seed as given (torch.manual_seed
once, batches in npz order).  Output = bb_sample.py format: preds (N,L,4) float32, unit_ids, meta (+ gate (N,)) and a
.json sidecar (meta + preds_sha256), so fmdx_monitor_v1/adapt_sidecars.py + lane_score.py score it unchanged."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sma_model as SM  # noqa: E402
U = SM.U
THR = 0.5
ROUTER_SHA = "22bb8802f2da46fc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("luo", "bb"), required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="sma_train checkpoints/checkpoint_<ms>.pt")
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--gt-npz", type=Path, required=True)
    ap.add_argument("--render", type=Path, help="bb kind: render224 npz/npy")
    ap.add_argument("--set-name", required=True)
    ap.add_argument("--route", required=True, help="'all' or a router manifest csv")
    ap.add_argument("--weights", choices=("ema", "online"), default="ema")
    ap.add_argument("--sampling-timesteps", type=int, default=50)
    ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)   # D2 pitfall: MoELoRA einsum at 256 filled 24 GB on the 4090
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int); ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-units", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["n_experts"] == 1, cfg["n_experts"]
    unet, binfo = SM.load_backbone(a.kind, a.backbone, cfg["backbone_weights"])
    if binfo["param_sha256"] != cfg["backbone_param_sha256"]:
        raise SystemExit("backbone does not match the checkpoint's backbone sha")
    moe = SM.make_moe(a.kind, unet, cfg["rank"], cfg["lora_alpha"], cfg["lora_targets"].split(","), cfg["gate_hidden"], cfg["gate_feat"])
    moe.load_trainable_state_dict(ck["ema_trainable"] if a.weights == "ema" else ck["trainable"])
    moe = moe.to(dev).eval()
    if a.kind == "luo":
        ds = U.MultiScriptDataset(str(a.gt_npz), "rendered", script_override=a.set_name)
    else:
        ds = SM.RenderDataset(a.gt_npz, a.render, a.set_name)
    ids = ds.unit_ids
    n_all = len(ids)
    if a.subset is not None:
        sel = np.sort(np.random.default_rng(a.subset_seed).choice(n_all, min(a.subset, n_all), replace=False))
    elif a.max_units is not None:
        sel = np.arange(min(a.max_units, n_all))
    else:
        sel = np.arange(n_all)
    if a.route == "all":
        gate = np.ones(len(sel), np.float32); rinfo = dict(route="all")
    else:
        prt = {}
        with open(a.route) as f:
            for r in csv.DictReader(f):
                prt[r["unit_id"]] = (float(r["p_rtl"]), r["pred_family"])
        miss = [u for u in ids[sel] if u not in prt]
        if miss:
            raise SystemExit(f"{len(miss)} units missing from router manifest {a.route}, e.g. {miss[:3]}")
        gate = np.array([1.0 if prt[u][0] >= THR else 0.0 for u in ids[sel]], np.float32)
        fam = np.array([prt[u][1] == "RTL" for u in ids[sel]])
        rinfo = dict(route="router_manifest", manifest=str(a.route), manifest_sha256=U.sha256_file(a.route), router_sha256_prefix=ROUTER_SHA,
                     threshold=THR, n_adapter_on=int(gate.sum()), n_units=int(len(sel)), agree_pred_family_RTL=bool((fam == (gate > 0)).all()))
    cur = {}
    orig = moe.gate_from_condition

    def routed(classes):
        fm, g = orig(classes)
        gb = cur["g"]
        assert gb.shape[0] == g.shape[0], (gb.shape, g.shape)
        return fm, gb.to(g.dtype).view(-1, 1)
    moe.gate_from_condition = routed
    diff = U.build_diffusion(moe, sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta).to(dev).eval()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = []
    with torch.no_grad():
        for b in range(0, len(sel), a.batch_size):
            jj = sel[b:b + a.batch_size]
            imgs = torch.stack([ds[int(i)]["img"] for i in jj]).to(dev)
            cur["g"] = torch.from_numpy(gate[b:b + a.batch_size]).to(dev)
            (x, _), _, _ = diff.sample(condition=imgs, ground_truth=None, batch_size=len(jj), is_testing=True)
            out.append(x.transpose(1, 2).float().cpu().numpy())
    preds = np.concatenate(out).astype(np.float32)
    files = [HERE / f for f in ("sma_model.py", "sma_train.py", "sma_sample.py")] + [SM.D2 / "d2moe_model.py", SM.D2 / "uhwr_route_common.py"]
    per_file, csha = U.code_sha256(files)
    meta = dict(kind="sma_sample", backbone_kind=a.kind, ckpt=str(a.ckpt), ckpt_sha256=U.sha256_file(a.ckpt), milestone=int(ck["milestone"]),
                step=int(ck["step"]), weights=a.weights, backbone=str(a.backbone), backbone_file_sha256=binfo["file_sha256"],
                backbone_param_sha256=binfo["param_sha256"], set=a.set_name, npz=str(a.gt_npz), gt_npz_sha256=U.sha256_file(a.gt_npz),
                render=str(a.render) if a.render else None, render_sha256=U.sha256_file(a.render) if a.render else None, routing=rinfo,
                sampler="DDIM", sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta, seed=a.seed, n_units=int(len(sel)),
                subset=a.subset, subset_seed=a.subset_seed if a.subset is not None else None, max_units=a.max_units, batch_size=a.batch_size,
                seconds=round(time.time() - t0, 1), device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__,
                code_sha256=per_file, code_sha256_combined=csha, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_name(op.name[:-4] + ".part.npz")
    np.savez_compressed(tmp, preds=preds, unit_ids=ids[sel], gate=gate, meta=json.dumps(meta))
    tmp.replace(op)
    U.write_json_atomic(str(op)[:-4] + ".json", dict(meta, preds_file=op.name, preds_sha256=U.sha256_file(op)))
    print(json.dumps({k: meta[k] for k in ("set", "milestone", "n_units", "seconds", "device")} | {"adapter_on": int(gate.sum())}))


if __name__ == "__main__":
    main()
