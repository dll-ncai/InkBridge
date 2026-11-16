"""Canonical-frame sampling wrapper (K1 frame kill check and C1 arm readouts), Route A layout.

Samples ONE checkpoint on a list of sets, each in one or more frames (e0_frames.py): the frame is applied to the
64x64 condition of EVERY unit of the set (zero-shot: the frame is the experimental variable, no script filter),
the model is sampled (EMA weights, DDIM, eta 0, GaussianDiffusion1D.sample(is_testing=True), fp32 — the
e0_sample.py / routeA_sample.py path unchanged) and the EXACT INVERSE frame is applied to the (N, L, 4) output, so
the saved preds are in the original frame and score against the unmirrored GT with e0_score.py / routeA_score.py
unchanged.  Seed protocol = routeA_sample.py: torch.manual_seed(seed); np.random.seed(seed) once per
(set, frame, seed), all units in index order, fixed batches.

Files (Route A layout): <out-dir>/preds/<set>__<label>_<frame>__seed<s>_preds.npz
    preds (N,L,4) float32 ORIGINAL frame (inverse applied), preds_in_frame (raw model output, for audits),
    unit_ids (source order), unit_indices, specialist, meta (json)
  + sidecar .json (frame, frame description, inverse_applied, checkpoint sha256, npz sha256, code sha256, GPU, seed,
    indices rule, ...) and <label>_sample_log.jsonl.

Sets (relative to --stage; identical relative paths on CLUSTER-A $U, VPS $VPS_ROOT\\uhwr\\stage,
CLUSTER-B $U_B):
  val250_ouhdl   data/ouhdl_e0_v3/e0_val.npz, the FIRST 250 units in stored order (indices 0..249; ids in the sidecar)
  test_ouhdl     data/ouhdl_e0_v3/e0_test.npz, all 4,406
  gold_ouhdl     data/ouhdl_e0_v3/e0_gold.npz, all 354
  val250_iam     data/m5_multiscript_v1/m5_val_iam.npz, all 250 (the symmetric LTR control: sample it in mirror_x)
  german_unipen_de_test, val_ouhdl, val_casia_zh, val_casia_en, val_hpl_tamil, test_iam500 as in lane_sample.SETS
Groups: k1 = val250_ouhdl test_ouhdl val250_iam; k1_val = val250_ouhdl val250_iam.

python frame_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage $U \
    --sets k1 --frames identity mirror_x vt --seeds 0 1 2 --out-dir OUT [--batch-size 256] [--max-units N]
    [--ltr-frames mirror_x identity]   (frames for the non-Urdu sets, i.e. the IAM control; default = --frames)
  val1000_ouhdl  data/ouhdl_e0_v3/e0_val.npz, e0_sample --subset 1000 --subset-seed 0 units (the Route A selection set)
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_frames as F
from e0_dataset import OuhdlE0Dataset
from ema_pytorch import EMA
from diffusion_models.ddpm_conditional import GaussianDiffusion1D, Unet1D
from diffusion_models.conditional_1d.runtime_utils import checkpoint_path
from utils.casia_grid import TRAJECTORY_CHANNELS

CONDITION = "rendered"
SETS = {  # name -> (stage-relative npz, script tag, index rule)
    "val250_ouhdl": ("data/ouhdl_e0_v3/e0_val.npz", "ouhdl", "first250"),
    "val1000_ouhdl": ("data/ouhdl_e0_v3/e0_val.npz", "ouhdl", "subset1000"),
    "val_ouhdl": ("data/m5_multiscript_v1/m5_val_ouhdl.npz", "ouhdl", "all"),
    "test_ouhdl": ("data/ouhdl_e0_v3/e0_test.npz", "ouhdl", "all"),
    "gold_ouhdl": ("data/ouhdl_e0_v3/e0_gold.npz", "ouhdl", "all"),
    "val250_iam": ("data/m5_multiscript_v1/m5_val_iam.npz", "iam", "all"),
    "val_casia_zh": ("data/m5_multiscript_v1/m5_val_casia_zh.npz", "casia_zh", "all"),
    "val_casia_en": ("data/m5_multiscript_v1/m5_val_casia_en.npz", "casia_en", "all"),
    "val_hpl_tamil": ("data/m5_multiscript_v1/m5_val_hpl_tamil.npz", "hpl_tamil", "all"),
    "test_iam500": ("data/m5_multiscript_v1/m5_test_iam500.npz", "iam", "all"),
    "german_unipen_de_test": ("data/d2_v2/unipen_de/e0_test.npz", "unipen_de", "all"),
}
GROUPS = {"k1": ["val250_ouhdl", "test_ouhdl", "val250_iam"], "k1_val": ["val250_ouhdl", "val250_iam"],
          "ouhdl_eval": ["val250_ouhdl", "test_ouhdl", "gold_ouhdl"]}
URDU_TAG = "ouhdl"


def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def build(run_dir, sampling_timesteps, ddim_eta):
    """Model construction from the run's run_config.json (d3a_sample.build / e0_sample.build, unchanged)."""
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


def set_indices(rule, n, max_units=None):
    """first250: indices 0..249 in stored order; subset1000: e0_sample.py --subset 1000 --subset-seed 0 rule
    (the E0 / M5 / Route A val1000 selection units); all: every unit."""
    if rule == "first250":
        idx = np.arange(min(250, n))
    elif rule == "subset1000":
        idx = np.sort(np.random.default_rng(0).choice(n, min(1000, n), replace=False))
    else:
        idx = np.arange(n)
    return idx if max_units is None else idx[:max_units]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="checkpoint label, e.g. LUO_ms6; the frame is appended: LUO_ms6_vt")
    ap.add_argument("--run-dir", required=True, help="run dir relative to --stage (has run_config.json, checkpoints/)")
    ap.add_argument("--milestone", type=int, required=True); ap.add_argument("--checkpoint", help="explicit checkpoint path (overrides run-dir/checkpoints)")
    ap.add_argument("--stage", required=True, help="data/run root: $U | $VPS_ROOT\\uhwr\\stage | $U_B")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", default=list(F.FRAMES), choices=F.FRAMES)
    ap.add_argument("--ltr-frames", nargs="*", choices=F.FRAMES, help="frames for non-Urdu (LTR) sets, e.g. IAM control in mirror_x; default: --frames")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--sampling-timesteps", type=int, default=50)
    ap.add_argument("--ddim-eta", type=float, default=0.0); ap.add_argument("--max-units", type=int, help="smoke: first N units of each set")
    a = ap.parse_args()
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    sets = []
    for s in a.sets:
        sets += GROUPS.get(s, [s])
    run_dir = Path(a.stage) / a.run_dir
    diff, targs = build(run_dir, a.sampling_timesteps, a.ddim_eta)
    assert diff.is_ddim_sampling == (a.sampling_timesteps < int(targs["timesteps"]))
    ckp = Path(a.checkpoint) if a.checkpoint else checkpoint_path(run_dir, a.milestone)
    ck_sha = sha256_file(ckp)
    ck = torch.load(str(ckp), map_location="cpu")
    ema = EMA(diff, beta=float(targs["ema_decay"]), update_every=int(targs["ema_update_every"]))
    ema.load_state_dict(ck["ema"])
    model = ema.ema_model.to(dev).eval()
    step = int(ck.get("step", -1)); del ck
    out = Path(a.out_dir) / "preds"; out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    code_sha = {p.name: sha256_file(p) for p in [here / "frame_sample.py", here / "e0_frames.py", here / "e0_dataset.py", here / "e0_adapter.py"] if p.is_file()}
    log = open(Path(a.out_dir) / f"{a.label}_frame_sample_log.jsonl", "a", encoding="utf-8")
    for s in sets:
        rel, tag, rule = SETS[s]
        src = Path(a.stage) / rel
        n_all = len(np.load(src, allow_pickle=False)["unit_ids"])
        idx = set_indices(rule, n_all, a.max_units)
        frames = (a.ltr_frames if (a.ltr_frames and tag != URDU_TAG) else a.frames)
        npz_sha = sha256_file(src)
        for fr in frames:
            ds = OuhdlE0Dataset(str(src), CONDITION, indices=idx, frame=fr)      # frame on every unit (no script filter)
            label = f"{a.label}_{fr}"
            for sd in a.seeds:
                name = f"{s}__{label}__seed{sd}_preds" + ("" if a.max_units is None else f"_smoke{a.max_units}")
                fn = out / (name + ".npz")
                if fn.is_file():
                    continue
                torch.manual_seed(sd); np.random.seed(sd)
                res, t0 = [], time.time()
                with torch.no_grad():
                    for b in range(0, len(ds), a.batch_size):
                        items = [ds[i] for i in range(b, min(b + a.batch_size, len(ds)))]
                        imgs = torch.stack([it["img"] for it in items]).to(dev)
                        (x, _), _, _ = model.sample(condition=imgs, ground_truth=None, batch_size=len(items), is_testing=True)
                        res.append(x.transpose(1, 2).float().cpu().numpy())
                raw = np.concatenate(res).astype(np.float32)
                preds = F.inverse_frame_preds(raw, fr)
                meta = dict(set=s, route_mode=label, specialist=label, checkpoint_label=a.label, frame=fr, frame_desc=F.describe(fr),
                            inverse_applied=fr != "identity", preds_key_note="preds = original frame (inverse applied); preds_in_frame = raw model output",
                            run_dir=str(run_dir), milestone=a.milestone, step=step, checkpoint=str(ckp), checkpoint_sha256=ck_sha,
                            npz=str(src), npz_rel=rel, npz_sha256=npz_sha, script_tag=tag, condition=CONDITION, indices_rule=rule,
                            n_units=int(len(idx)), unit_indices_first=int(idx[0]), unit_indices_last=int(idx[-1]), n_units_in_npz=int(n_all),
                            sampler="DDIM" if diff.is_ddim_sampling else "ancestral", sampling_timesteps=a.sampling_timesteps, ddim_eta=a.ddim_eta,
                            seed=sd, batch_size=a.batch_size, weights="EMA", seconds=round(time.time() - t0, 1), device=gpu, torch=torch.__version__,
                            cuda=torch.version.cuda, host=platform.node(), code="frame_sample.py", code_sha256=code_sha, max_units=a.max_units,
                            peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None,
                            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), route_counts={label: int(len(idx))},
                            specialists={label: dict(run_dir=str(run_dir), milestone=a.milestone, step=step, checkpoint_sha256=ck_sha, device=gpu, frame=fr)})
                tmp = out / (name + ".part.npz")
                np.savez_compressed(tmp, preds=preds, preds_in_frame=raw, unit_ids=ds.unit_ids[ds.idx], unit_indices=idx.astype(np.int64),
                                    specialist=np.array([label] * len(idx)), meta=json.dumps(meta))
                os.replace(tmp, fn)
                meta["preds_sha256"] = sha256_file(fn); meta["preds_file"] = fn.name
                meta["unit_ids"] = ds.unit_ids[ds.idx].tolist() if rule == "first250" else None   # val250 ids recorded explicitly
                json.dump(meta, open(str(fn)[:-4] + ".json", "w"), indent=1)
                rec = dict(event="done", **{k: meta[k] for k in ("set", "frame", "seed", "n_units", "seconds", "preds_file", "preds_sha256")})
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    summ = dict(event="label_done", label=a.label, sets=sets, frames=a.frames, ltr_frames=a.ltr_frames, seeds=a.seeds,
                wall_seconds=round(time.time() - t_start, 1), device=gpu, checkpoint_sha256=ck_sha)
    log.write(json.dumps(summ) + "\n"); log.close(); print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
