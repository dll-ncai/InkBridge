"""FMDX backbone student model (track 'Backbone build', 3 Nov 2026).

Nagamatsu Unet1D (E0_ARCH, = LUO ms6 architecture) whose 64-px ImageEncoder is replaced by a 224-px encoder with a 16x16
token grid and per-level heads that rebuild the five cross-attention context maps (64@64, 64@32, 128@16, 256@8, 512@4).

Encoders (--encoder):
  vits14          vendor/student_model.StudentEncoder (byte-identical copy of fmd_student_v1/student_model.py): ViT-S/14,
                  224 px, 16x16 tokens, width 384, 12 blocks, fixed 2-D sin-cos positions, seeded init (20261104).
  pe_spatial_s16  PEINIT (prereg A4.2): pe-spatial-s16-512 tower at 224 px / patch 14 (FlexiViT pinv kernel resize 16->14,
                  bilinear position-embedding interpolation to 16x16), PE block code incl. RoPE via timm; see
                  PEStudentEncoder. Per-level heads = the StudentEncoder heads (same construction, same seeded init).
Both encoders expose .last_tokens (b, 256, 384) = the REPA alignment point (final-norm tokens, row-major (r, c)).
FM-free at inference: no FM library call, no FM tensor and no REPA projector is part of the inference model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import student_model as SM  # noqa: E402  (byte-identical copy of fmd_student_v1/student_model.py)

ENCODERS = ("vits14", "pe_spatial_s16")
PE_TIMM_NAME = "vit_pe_spatial_small_patch16_512"


def _make_heads(D, grid, levels):
    heads = []
    for c, s in levels:
        if s >= grid:
            r = s // grid
            heads.append(nn.ModuleDict(dict(norm=nn.LayerNorm(D, eps=1e-6), lin=nn.Linear(D, c * r * r))))
        else:
            r = grid // s
            heads.append(nn.ModuleDict(dict(norm=nn.LayerNorm(D, eps=1e-6), lin=nn.Linear(D * r * r, c))))
    return nn.ModuleList(heads)


@torch.no_grad()
def _init_heads(heads, seed):
    """Same rule as StudentEncoder._init for Linear / LayerNorm modules (private generator)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for m in heads.modules():
        if isinstance(m, nn.Linear):
            w = torch.empty_like(m.weight, device="cpu")
            nn.init.trunc_normal_(w, std=0.02, a=-0.04, b=0.04, generator=g)
            m.weight.copy_(w)
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.fill_(1.0); m.bias.zero_()


def _maps_from_tokens(t, heads, levels, grid):
    """Identical to StudentEncoder.forward after the token computation."""
    b, n, D = t.shape
    g = grid
    maps = []
    for (c, s), h in zip(levels, heads):
        u = h["norm"](t)
        if s >= g:
            r = s // g
            m = h["lin"](u).transpose(1, 2).reshape(b, c * r * r, g, g)
            maps.append(F.pixel_shuffle(m, r))
        else:
            r = g // s
            u = u.reshape(b, s, r, s, r, D).permute(0, 1, 3, 2, 4, 5).reshape(b, s * s, r * r * D)
            maps.append(h["lin"](u).transpose(1, 2).reshape(b, c, s, s))
    return maps


class PEStudentEncoder(nn.Module):
    """PEINIT encoder (prereg A4.2): the pe-spatial-s16-512 vision tower (12 blocks, width 384; PE block code incl. its RoPE,
    as implemented by timm 'vit_pe_spatial_small_patch16_512') built at img_size 224 / patch 14 so the 224-px render gives
    16x16 tokens. Weights: the staged original checkpoint is converted by timm's own eva.checkpoint_filter_fn at the native
    geometry (512 px, patch 16), then the patch-embedding kernel is resized 16 -> 14 by the FlexiViT pseudo-inverse resize
    (timm.layers.resample_patch_embed, bilinear, no antialias) and the absolute position embedding is interpolated
    bilinearly 32x32 -> 16x16 (timm.layers.resample_abs_pos_embed, prefix tokens kept). Input: the render inverted to black
    ink on white (screen polarity), RGB replicate, normalised with the tower's mean/std; prefix tokens dropped. Level heads =
    the StudentEncoder heads (same construction, same seeded init). weights=None builds the architecture only."""

    def __init__(self, weights=None, levels=SM.LEVELS, init_seed=SM.ENCODER_INIT_SEED, timm_name=PE_TIMM_NAME):
        super().__init__()
        import timm  # architecture code only (pinned side-install, see DECISION_RECORD_backbone.md)
        self.levels, self.grid, self.timm_name = tuple(levels), 16, timm_name
        self.trunk = timm.create_model(timm_name, pretrained=False, img_size=224, patch_size=14, num_classes=0)
        D = self.trunk.num_features
        assert D == SM.VIT_S14["dim"], f"PE trunk width {D} != 384"
        self.npre = int(getattr(self.trunk, "num_prefix_tokens", 0))
        cfg = getattr(self.trunk, "pretrained_cfg", {}) or {}
        mean, std = cfg.get("mean", (0.5, 0.5, 0.5)), cfg.get("std", (0.5, 0.5, 0.5))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=True)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=True)
        self.heads = _make_heads(D, self.grid, self.levels)
        _init_heads(self.heads, init_seed)
        self.weights_record = None
        if weights is not None:
            self.weights_record = self.load_pe_weights(weights)
        self.last_tokens = None

    def load_pe_weights(self, path):
        import hashlib
        import timm
        import timm.models.eva as eva
        from timm.layers import resample_patch_embed, resample_abs_pos_embed
        p = Path(path)
        if p.is_dir():
            cands = sorted(list(p.glob("*.pt")) + list(p.glob("*.pth")) + list(p.glob("*.safetensors")) + list(p.glob("*.bin")))
            if not cands:
                raise FileNotFoundError(f"no weight file in {p}")
            p = cands[0]
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(p))
        else:
            sd = torch.load(str(p), map_location="cpu", weights_only=False)
            for k in ("model", "state_dict"):
                if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
        native = timm.create_model(self.timm_name, pretrained=False, num_classes=0)
        sdn = eva.checkpoint_filter_fn(sd, native)
        miss, unexp = native.load_state_dict(sdn, strict=False)
        miss = [m for m in miss if not m.startswith("head")]
        if miss or unexp:
            raise RuntimeError(f"PE native load mismatch: missing {miss[:8]} unexpected {unexp[:8]}")
        nsd = native.state_dict()
        g_old = native.patch_embed.grid_size
        w = nsd["patch_embed.proj.weight"]
        nsd["patch_embed.proj.weight"] = resample_patch_embed(w, [14, 14], interpolation="bilinear", antialias=False)
        if "pos_embed" in nsd and nsd["pos_embed"] is not None:
            nsd["pos_embed"] = resample_abs_pos_embed(nsd["pos_embed"], new_size=[16, 16], old_size=list(g_old),
                                                      num_prefix_tokens=self.npre, interpolation="bilinear", antialias=False)
        tsd = self.trunk.state_dict()
        bad = {k: (tuple(v.shape), tuple(tsd[k].shape)) for k, v in nsd.items() if k in tsd and tuple(v.shape) != tuple(tsd[k].shape)}
        if bad:
            raise RuntimeError(f"PE resized shapes do not match the 224/14 tower: {list(bad.items())[:5]}")
        miss, unexp = self.trunk.load_state_dict({k: v for k, v in nsd.items() if k in tsd}, strict=False)
        miss = [m for m in miss if not m.startswith("head")]
        if miss:
            raise RuntimeError(f"PE 224/14 load missing keys {miss[:8]}")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        rope = getattr(self.trunk, "rope", None)
        return dict(path=str(p), sha256=h.hexdigest(), n_keys=len(nsd), timm_name=self.timm_name, timm_version=timm.__version__,
                    native_grid=list(g_old), patch_resize="FlexiViT pinv 16->14 (timm resample_patch_embed, bilinear, antialias False)",
                    pos_embed_resize="bilinear 32x32->16x16 (timm resample_abs_pos_embed, antialias False)" if "pos_embed" in nsd else "none",
                    num_prefix_tokens=self.npre, rope=type(rope).__name__ if rope is not None else None,
                    unused_checkpoint_keys=sorted(set(nsd) - set(tsd))[:20])

    def tokens(self, x):
        x = 1.0 - x                                                   # white-on-black -> black-on-white (screen polarity)
        x = (x.expand(-1, 3, -1, -1) - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        t = self.trunk.forward_features(x)
        t = t[:, self.npre:]
        assert t.shape[1] == self.grid * self.grid, t.shape
        return t

    def forward(self, x):
        t = self.tokens(x)
        self.last_tokens = t
        return _maps_from_tokens(t, self.heads, self.levels, self.grid)


def make_encoder(kind, pe_weights=None, init_seed=SM.ENCODER_INIT_SEED):
    if kind == "vits14":
        return SM.StudentEncoder(init_seed=init_seed)
    if kind == "pe_spatial_s16":
        return PEStudentEncoder(weights=pe_weights, init_seed=init_seed)
    raise ValueError(kind)


def count(m):
    return sum(p.numel() for p in m.parameters())


def build_backbone_unet(U, encoder="vits14", pe_weights=None, enc_seed=SM.ENCODER_INIT_SEED):
    """Plain Unet1D (E0_ARCH) with the 64-px encoder replaced. Weight initialisation of the U-Net is whatever the global
    RNG gives (the caller seeds it); the encoder uses its private seeded generator."""
    unet = U.build_unet(U.E0_ARCH)
    old = count(unet.encoder)
    unet.encoder = make_encoder(encoder, pe_weights, enc_seed)
    info = dict(encoder=encoder, old_encoder_params=old, new_encoder_params=count(unet.encoder), unet_params_total=count(unet),
                pe_weights=getattr(unet.encoder, "weights_record", None))
    return unet, info


def load_non_encoder(unet, sd):
    """Load every non-encoder key of a plain Unet1D state dict (LUO ms6); returns the number of loaded keys."""
    sd = {k: v for k, v in sd.items() if not k.startswith("encoder.")}
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("encoder.") and not m.endswith("div_term")]
    if missing or unexpected:
        raise RuntimeError(f"non-encoder load mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return len(sd)


def load_backbone_for_inference(path, U, dev="cpu", which="ema"):
    """bb_ms<k>.pt (format fmdx_bb/1) -> eval Unet1D with the EMA (default) or online weights."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck["format_version"] == "fmdx_bb/1", ck.get("format_version")
    unet, _ = build_backbone_unet(U, ck["config"]["encoder"], None)
    sd = ck["ema_unet" if which == "ema" else "model_unet"]
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("div_term")]
    if missing or unexpected:
        raise RuntimeError(f"bb checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return unet.to(dev).eval(), ck
