"""SM-A (PREREG_RQ_finalise amendment A2.3): ONE LoRA adapter (rank 16, alpha 16, the D2 projection set
cross,attn,linear = CrossAttention.to_q/to_kv/to_out[0], Attention.to_qkv/to_out, LinearAttention.to_qkv/to_out[0])
on a FROZEN backbone, reusing D2's MoEUnet1D/MoELoRA (d2moe_code_9a020b75, unchanged) with n_experts = 1.
With E = 1 the D2 softmax gate is identically 1 during training (single adapter, no routing, load-balance term
constant); at test the per-unit gate is REPLACED by the ScriptCNN router decision (1 = adapter on, 0 = adapter off,
which is the exact backbone because the delta is multiplied by the gate).
Backbone kinds: 'luo' = Nagamatsu Unet1D from a Trainer1D checkpoint (64-px ImageEncoder, e0 'rendered' condition);
'bb' = FMDX backbone milestone bb_ms<k>.pt (Unet1D with the 224-px student encoder, FMD A4.2 render224 path).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

import os
D2 = Path(os.environ.get("SMA_D2_DIR", "$U/exp/D2_MOE/code/d2moe_code_9a020b75"))   # VPS/CLUSTER-B override
BB = Path(os.environ.get("SMA_BB_DIR", "$U/exp/FMD/xs/code/fmdx_backbone_v1"))
for p in (str(D2), str(BB), str(BB / "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)
import uhwr_route_common as U  # noqa: E402  (byte-identical in D2 and FMDX vendor, sha256 c4242765...)
from d2moe_model import MoEUnet1D  # noqa: E402


class SMAUnet1D(MoEUnet1D):
    probe_hw = 64

    def _feature_dim(self):
        with torch.no_grad():
            fm = self.backbone.encoder(torch.zeros(1, 1, self.probe_hw, self.probe_hw))
        from d2moe_model import pooled_features
        return pooled_features(fm, self.gate_feat).shape[1]


class SMAUnet1D_224(SMAUnet1D):
    probe_hw = 224


def load_backbone(kind, path, which="ema"):
    """-> (unet, info) with info keys param_sha256, step, file_sha256 (D2 build_moe contract)."""
    if kind == "luo":
        unet, info = U.load_unet_from_checkpoint(path, which=which)
    else:
        import bb_model as BM
        unet, ck = BM.load_backbone_for_inference(path, U, "cpu", which)
        info = dict(param_sha256=U.sha256_state(unet.state_dict()), step=int(ck["step"]), milestone=int(ck["milestone"]),
                    arm_id=ck["config"].get("arm_id"), encoder=ck["config"].get("encoder"), kind="fmdx_bb/1")
    info["file_sha256"] = U.sha256_file(path)
    return unet, info


def make_moe(kind, unet, rank=16, alpha=16.0, targets=("cross", "attn", "linear"), gate_hidden=128, gate_feat="deepest"):
    cls = SMAUnet1D if kind == "luo" else SMAUnet1D_224
    return cls(unet, n_experts=1, rank=rank, alpha=alpha, targets=tuple(targets), gate_hidden=gate_hidden, gate_feat=gate_feat,
               expert_names=["sma"])


class RenderDataset(Dataset):
    """bb kind: E0-format npz + 224-px render (.npy in npz order, or npz with img224 + unit_ids); MultiScriptDataset item format."""

    def __init__(self, npz, render, script_override="ouhdl"):
        z = np.load(npz, allow_pickle=False)
        self.unit_ids = z["unit_ids"].astype(str)
        n = len(self.unit_ids)
        self.scripts = z["scripts"].astype(str) if "scripts" in z.files else np.array([script_override] * n)
        self.writer = z["writer_ids"] if "writer_ids" in z.files else np.zeros(n, np.int64)
        seq, off = z["seq"], z["seq_offsets"]
        self.seq = [np.ascontiguousarray(seq[off[j]:off[j + 1]], np.float32) for j in range(n)]
        render = Path(render)
        if render.suffix == ".npy":
            self.img = np.load(render, mmap_mode="r"); self.row = np.arange(n)
            if self.img.shape[0] != n:
                raise SystemExit(f"render {render} {self.img.shape} != {n} units of {npz}")
        else:
            R = np.load(render, allow_pickle=False); self.img = R["img224"]; rid = R["unit_ids"].astype(str)
            pos = {u: k for k, u in enumerate(rid)}
            self.row = np.array([pos.get(u, -1) for u in self.unit_ids])
        self.files = [dict(path=str(npz), render=str(render), n_file=int(n), n_kept=int(n))]
        self.cond = None

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, i):
        r = int(self.row[i])
        if r < 0:
            raise KeyError(f"unit {self.unit_ids[i]} missing from the render file")
        img = (255.0 - np.asarray(self.img[r], np.float32)) / 255.0
        return {"img": torch.from_numpy(img).unsqueeze(0), "coords": torch.from_numpy(self.seq[i]),
                "tag_char": self.unit_ids[i], "script": self.scripts[i], "index": int(i)}

    def script_counts(self):
        u, c = np.unique(self.scripts, return_counts=True)
        return {str(a): int(b) for a, b in zip(u, c)}
