"""Shared helpers for the D3 Stage B (routed-teacher distillation) and D2 (adapter experts + gate)
trainers built on the Nagamatsu Unet1D / GaussianDiffusion1D code (repo commit 7c5cf623) and the
E0 / M5 data format (uhwr, 24 Oct 2026).

Requires the Nagamatsu repo ``src`` directory on PYTHONPATH (as e0_train.py / m5_train.py do).

Data format (E0 / m5_multiscript_v1 npz): cond_rendered (N,64,64) uint8 black-on-white,
seq (sum L_i, 4) float32 [x, y, pen, eos] in Nagamatsu space, seq_offsets (N+1,), unit_ids,
scripts (N,) script tag in {casia_zh, casia_en, hpl_tamil, iam, ouhdl}, writer_ids / sample_ids.
"""
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from diffusion_models.ddpm_conditional import GaussianDiffusion1D, Unet1D
from utils.casia_grid import TRAJECTORY_CHANNELS, collate_fn

SCRIPTS = ("casia_zh", "casia_en", "hpl_tamil", "iam", "ouhdl")

# E0 / M5 / X5 / E27 architecture (identical in all four run_config.json files, checked 24 Oct 2026)
E0_ARCH = dict(model_dim=64, dim_mults=(1, 2, 4, 8), encoder_hidden_dims=(64, 128, 256, 512),
               attn_dim_head=32, attn_heads=4, cross_attn_dim_head=64, cross_attn_heads=8,
               linear_attn_dim_head=32, linear_attn_heads=4, attention_max_positions=5000,
               timesteps=1000, objective="pred_noise", beta_schedule="cosine", sequence_length=320,
               image_size=64, linear_beta_start=0.0001, linear_beta_end=0.02, cosine_schedule_s=0.008,
               ema_decay=0.995, ema_update_every=10)


# ----------------------------------------------------------------------------- provenance helpers
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_state(state_dict):
    """sha256 over the sorted (key, bytes) of a state dict / parameter dict (cpu, contiguous)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        v = state_dict[k]
        if not torch.is_tensor(v):
            continue
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sha256_module_params(module):
    return sha256_state({k: v for k, v in module.state_dict().items()})


def code_sha256(files):
    """Per-file sha256 plus one combined digest (sorted by name), for run_config provenance."""
    per = {Path(f).name: sha256_file(f) for f in files}
    comb = hashlib.sha256("".join(f"{k}:{per[k]}\n" for k in sorted(per)).encode()).hexdigest()
    return per, comb


def environment_record():
    rec = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               hostname=socket.gethostname(), platform=platform.platform(), argv=sys.argv,
               slurm_job_id=os.environ.get("SLURM_JOB_ID"), cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["gpu_count"] = torch.cuda.device_count()
    return rec


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def save_torch_atomic(data, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except Exception:
        return None


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


# ----------------------------------------------------------------------------- data
class MultiScriptDataset(Dataset):
    """E0-format npz (one or several files) -> samples in the PotDataset format expected by collate_fn,
    plus the script tag.  Several files are concatenated; the script of each unit is read from the
    file's ``scripts`` array (or forced with ``script_override`` for files without one, e.g. e0_train.npz)."""

    def __init__(self, npz_paths, condition="rendered", scripts_keep=None, script_override=None,
                 max_units_per_script=None, seed=0):
        self.cond, self.seq, self.off, self.unit_ids, self.scripts, self.writer = [], [], [], [], [], []
        self.files = []
        paths = [npz_paths] if isinstance(npz_paths, (str, Path)) else list(npz_paths)
        for p in paths:
            z = np.load(p, allow_pickle=False)
            key = f"cond_{condition}"
            if key not in z.files:
                raise KeyError(f"{p} has no {key}; available {[f for f in z.files if f.startswith('cond_')]}")
            n = len(z["unit_ids"])
            if "scripts" in z.files:
                sc = z["scripts"].astype(str)
            elif script_override is not None:
                sc = np.array([script_override] * n)
            else:
                raise KeyError(f"{p} has no 'scripts' array; pass script_override")
            keep = np.arange(n)
            if scripts_keep is not None:
                keep = keep[np.isin(sc[keep], list(scripts_keep))]
            if max_units_per_script is not None:
                rng = np.random.default_rng(seed)
                sel = []
                for s in np.unique(sc[keep]):
                    idx = keep[sc[keep] == s]
                    if len(idx) > max_units_per_script:
                        idx = np.sort(rng.choice(idx, max_units_per_script, replace=False))
                    sel.append(idx)
                keep = np.concatenate(sel) if sel else keep[:0]
            cond = z[key]
            seq, off = z["seq"], z["seq_offsets"]
            wr = z["writer_ids"] if "writer_ids" in z.files else np.zeros(n, np.int64)
            for j in keep:
                self.cond.append(cond[j])
                self.seq.append(np.ascontiguousarray(seq[off[j]:off[j + 1]], np.float32))
                self.unit_ids.append(str(z["unit_ids"][j]))
                self.scripts.append(str(sc[j]))
                self.writer.append(int(wr[j]))
            self.files.append(dict(path=str(p), n_file=int(n), n_kept=int(len(keep))))
        self.cond = np.stack(self.cond) if self.cond else np.zeros((0, 64, 64), np.uint8)
        self.scripts = np.array(self.scripts)
        self.unit_ids = np.array(self.unit_ids)
        self.writer = np.array(self.writer)
        if self.cond.shape[1:] != (64, 64):
            raise ValueError(f"condition is {self.cond.shape[1:]}, expected 64x64")

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, i):
        img = (255 - self.cond[i].astype(np.float32)) / 255.0  # black-on-white -> white-on-black (E0 convention)
        return {"img": torch.from_numpy(img).unsqueeze(0), "coords": torch.from_numpy(self.seq[i]),
                "tag_char": self.unit_ids[i], "script": self.scripts[i], "index": int(i)}

    def script_counts(self):
        u, c = np.unique(self.scripts, return_counts=True)
        return {str(a): int(b) for a, b in zip(u, c)}


def collate_with_script(batch, sequence_length=320):
    out = collate_fn(batch, sequence_length=sequence_length)  # imgs (b,1,64,64), coords (b,4,L)
    out["scripts"] = [b["script"] for b in batch]
    out["index"] = torch.tensor([b["index"] for b in batch])
    return out


def make_loader(ds, batch_size, shuffle, num_workers, seed, sequence_length=320, drop_last=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                      collate_fn=lambda b: collate_with_script(b, sequence_length), generator=g,
                      persistent_workers=num_workers > 0)


def cycle(dl):
    while True:
        for d in dl:
            yield d


# ----------------------------------------------------------------------------- model
def build_unet(arch=E0_ARCH):
    a = arch
    return Unet1D(dim=a["model_dim"], dim_mults=tuple(a["dim_mults"]), channels=TRAJECTORY_CHANNELS,
                  attn_dim_head=a["attn_dim_head"], attn_heads=a["attn_heads"],
                  cross_attn_dim_head=a["cross_attn_dim_head"], cross_attn_heads=a["cross_attn_heads"],
                  linear_attn_dim_head=a["linear_attn_dim_head"], linear_attn_heads=a["linear_attn_heads"],
                  attention_max_positions=a["attention_max_positions"], condition_image_size=a["image_size"],
                  encoder_hidden_dims=tuple(a["encoder_hidden_dims"]))


def build_diffusion(model, arch=E0_ARCH, sampling_timesteps=None, ddim_eta=0.0):
    a = arch
    return GaussianDiffusion1D(model, seq_length=a["sequence_length"], timesteps=a["timesteps"],
                               sampling_timesteps=sampling_timesteps, objective=a["objective"],
                               beta_schedule=a["beta_schedule"], ddim_sampling_eta=ddim_eta,
                               linear_beta_start=a["linear_beta_start"], linear_beta_end=a["linear_beta_end"],
                               cosine_schedule_s=a["cosine_schedule_s"])


def load_checkpoint_weights(path, which="ema"):
    """Return (unet_state_dict, info) from either a full Trainer1D checkpoint
    ({'model','ema','opt',...}; 'ema' holds 'ema_model.model.<k>') or an fp16 extract written by
    tmp/teachers_fp16/extract.py ({'ema_model': {'model.<k>',...}, 'model': {...}}).  ``which`` = 'ema' | 'model'.
    Keys are returned relative to the Unet1D (no 'model.' prefix), cast to float32."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    info = dict(path=str(path), step=int(ck.get("step", -1)), which=which)
    if "ema_model" in ck and which == "ema":            # fp16 extract
        src, prefix = ck["ema_model"], "model."
        info["format"] = "fp16_extract"
    elif "model" in ck and which == "model" and "opt" not in ck:  # fp16 extract, online weights
        src, prefix = ck["model"], "model."
        info["format"] = "fp16_extract"
    elif which == "ema":                                 # full Trainer1D checkpoint
        src, prefix = ck["ema"], "ema_model.model."
        info["format"] = "trainer1d"
    else:
        src, prefix = ck["model"], "model."
        info["format"] = "trainer1d"
    sd = {k[len(prefix):]: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in src.items() if k.startswith(prefix)}
    if not sd:
        raise KeyError(f"no keys with prefix {prefix!r} in {path}")
    info["n_keys"] = len(sd)
    info["source_sha256"] = ck.get("source_sha256")
    return sd, info


def load_unet_from_checkpoint(path, which="ema", arch=E0_ARCH):
    unet = build_unet(arch)
    sd, info = load_checkpoint_weights(path, which)
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    # Unet1D._load_from_state_dict drops legacy keys itself; anything else missing is an error
    missing = [m for m in missing if not m.endswith("div_term")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch for {path}: missing {missing[:5]} unexpected {unexpected[:5]}")
    info["param_sha256"] = sha256_module_params(unet)
    return unet, info


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latest_milestone(run_dir):
    ms = []
    for p in (Path(run_dir) / "checkpoints").glob("checkpoint_*.pt"):
        try:
            ms.append(int(p.stem.split("_")[1]))
        except ValueError:
            pass
    return max(ms) if ms else None


def checkpoint_file(run_dir, milestone):
    return Path(run_dir) / "checkpoints" / f"checkpoint_{milestone:06d}.pt"


def prune_checkpoints(run_dir, keep_last, keep_every, save_every):
    """Delete old checkpoints except the last ``keep_last`` and every milestone whose step is a multiple of
    ``keep_every`` (0 = keep everything).  Never touches anything outside run_dir/checkpoints."""
    if not keep_last:
        return []
    files = sorted((Path(run_dir) / "checkpoints").glob("checkpoint_*.pt"))
    removed = []
    for p in files[:-keep_last]:
        ms = int(p.stem.split("_")[1])
        if keep_every and (ms * save_every) % keep_every == 0:
            continue
        p.unlink()
        removed.append(p.name)
    return removed


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **rec):
        rec.setdefault("time", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def decode_preds_meta(**kw):
    return json.dumps(kw, default=str)
