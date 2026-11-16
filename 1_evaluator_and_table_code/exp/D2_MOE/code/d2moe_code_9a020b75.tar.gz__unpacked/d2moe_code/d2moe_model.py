"""D2: LoRA adapter experts + image-conditioned gate on a frozen Nagamatsu Unet1D backbone.

MoELoRA   wraps one projection (nn.Linear or 1x1 nn.Conv1d) of the backbone:
          y = base(x) + (alpha / r) * sum_e g[b, e] * B_e(A_e(x)),  A_e: in->r (Kaiming-uniform), B_e: r->out (zeros)
          so every expert starts as an exact zero delta.  Experts are separate Parameters (ParameterList), so
          single experts can be frozen (stage 2) or added (add_expert).
Gate      2-layer MLP on the pooled ImageEncoder feature maps of the frozen backbone (default: global average of
          the deepest 4x4x512 map -> 512-d; 'all' = concat of the pooled 5 maps, 1024-d) -> softmax over experts.
MoEUnet1D forward(x, t, cond): encoder once -> gate -> set g on every wrapper -> the unchanged Unet1D forward
          body (down / mid / up / final blocks of the backbone).  Only adapters + gate are trainable; the backbone is
          kept in eval mode (BatchNorm2d running stats, positional-encoding dropout off) and its parameters frozen.
Load-balance penalty (Switch-Transformer form): L_lb = E * sum_e f_e * P_e, f_e = fraction of units routed
          (argmax) to e, P_e = mean gate probability of e in the batch.

Targets (--lora-targets): 'cross' = CrossAttention.to_q / to_kv / to_out[0] (all U-Net levels, mid, final),
'attn' = mid Attention.to_qkv / to_out, 'linear' = LinearAttention.to_qkv / to_out[0] (all levels).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_models.conditional_1d.unet1d import Attention, CrossAttention, LinearAttention


class MoELoRA(nn.Module):
    def __init__(self, base, n_experts, rank, alpha):
        super().__init__()
        if isinstance(base, nn.Linear):
            self.kind, in_f, out_f = "linear", base.in_features, base.out_features
        elif isinstance(base, nn.Conv1d) and base.kernel_size == (1,):
            self.kind, in_f, out_f = "conv1x1", base.in_channels, base.out_channels
        else:
            raise TypeError(f"MoELoRA supports nn.Linear or 1x1 nn.Conv1d, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_f, self.out_f, self.rank, self.alpha = in_f, out_f, rank, alpha
        self.scaling = alpha / rank
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in range(n_experts):
            self._append_expert(*self._fresh_expert())
        self.gate = None  # (b, E) set by MoEUnet1D before each forward

    def _fresh_expert(self):
        A = torch.empty(self.rank, self.in_f)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = torch.zeros(self.out_f, self.rank)
        return A, B

    def _append_expert(self, A, B):
        self.A.append(nn.Parameter(A.clone()))
        self.B.append(nn.Parameter(B.clone()))

    @property
    def n_experts(self):
        return len(self.A)

    def add_expert(self, weights, mode="gate_mix"):
        """New expert from the existing ones.  gate_mix: best rank-r approximation (SVD) of the mixed delta
        sum_e w_e B_e A_e; fresh: Kaiming A, zero B (zero delta)."""
        if mode == "fresh":
            A, B = self._fresh_expert()
        else:
            with torch.no_grad():
                w = torch.as_tensor(weights, dtype=torch.float32, device=self.A[0].device)
                delta = sum(w[e] * (self.B[e].float() @ self.A[e].float()) for e in range(self.n_experts))  # (out, in)
                Uu, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                r = self.rank
                sq = S[:r].clamp_min(0).sqrt()
                B = Uu[:, :r] * sq[None, :]
                A = sq[:, None] * Vh[:r, :]
                if S[:r].sum() <= 0:  # all-zero experts -> fresh init (zero delta)
                    A, B = self._fresh_expert()
        self._append_expert(A.detach(), B.detach())

    def set_expert_trainable(self, e, flag):
        self.A[e].requires_grad_(flag)
        self.B[e].requires_grad_(flag)

    def forward(self, x):
        y = self.base(x)
        g = self.gate
        if g is None:
            return y
        A = torch.stack(list(self.A))  # (E, r, in)
        B = torch.stack(list(self.B))  # (E, out, r)
        if self.kind == "linear":      # x (b, n, in)
            h = torch.einsum("bni,eri->bern", x.to(A.dtype), A)
            d = torch.einsum("bern,eor->beon", h, B)
            d = torch.einsum("beon,be->bno", d, g.to(d.dtype))
        else:                          # x (b, in, n)
            h = torch.einsum("bin,eri->bern", x.to(A.dtype), A)
            d = torch.einsum("bern,eor->beon", h, B)
            d = torch.einsum("beon,be->bon", d, g.to(d.dtype))
        return y + self.scaling * d.to(y.dtype)


class Gate(nn.Module):
    def __init__(self, in_dim, hidden, n_experts):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_experts)
        nn.init.zeros_(self.fc2.weight)   # uniform routing at step 0
        nn.init.zeros_(self.fc2.bias)

    @property
    def n_experts(self):
        return self.fc2.out_features

    def forward(self, feat):
        return self.fc2(F.silu(self.fc1(feat)))

    def add_expert(self, weights):
        """New output row = gate-weighted mix of the existing rows (weight and bias)."""
        with torch.no_grad():
            w = torch.as_tensor(weights, dtype=torch.float32, device=self.fc2.weight.device)
            new_w = (w[:, None] * self.fc2.weight).sum(0, keepdim=True)
            new_b = (w * self.fc2.bias).sum().view(1)
            fc = nn.Linear(self.fc2.in_features, self.n_experts + 1).to(self.fc2.weight.device)
            fc.weight.copy_(torch.cat([self.fc2.weight, new_w], 0))
            fc.bias.copy_(torch.cat([self.fc2.bias, new_b], 0))
        self.fc2 = fc


def pooled_features(feature_maps, mode="deepest"):
    if mode == "deepest":
        return feature_maps[-1].float().mean(dim=(2, 3))
    return torch.cat([f.float().mean(dim=(2, 3)) for f in feature_maps], dim=1)


class MoEUnet1D(nn.Module):
    def __init__(self, backbone, n_experts=3, rank=16, alpha=16.0, targets=("cross", "attn", "linear"),
                 gate_hidden=128, gate_feat="deepest", expert_names=None):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.channels = backbone.channels
        self.downsample_factor = getattr(backbone, "downsample_factor", 1)
        self.targets = tuple(targets)
        self.rank, self.alpha, self.gate_feat = rank, alpha, gate_feat
        self.expert_names = list(expert_names) if expert_names else [f"expert{e}" for e in range(n_experts)]
        self._wrappers, self.adapter_names = [], []
        self._wrap(n_experts)
        feat_dim = self._feature_dim()
        self.gate = Gate(feat_dim, gate_hidden, n_experts)
        self._last_gate = None

    # ---- wrapping
    def _feature_dim(self):
        with torch.no_grad():
            fm = self.backbone.encoder(torch.zeros(1, 1, 64, 64))
        return pooled_features(fm, self.gate_feat).shape[1]

    def _wrap(self, n_experts):
        specs = []
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, CrossAttention) and "cross" in self.targets:
                specs += [(mod, "to_q", name + ".to_q"), (mod, "to_kv", name + ".to_kv"), (mod.to_out, "0", name + ".to_out.0")]
            elif isinstance(mod, Attention) and "attn" in self.targets:
                specs += [(mod, "to_qkv", name + ".to_qkv"), (mod, "to_out", name + ".to_out")]
            elif isinstance(mod, LinearAttention) and "linear" in self.targets:
                specs += [(mod, "to_qkv", name + ".to_qkv"), (mod.to_out, "0", name + ".to_out.0")]
        for parent, attr, full in specs:
            base = getattr(parent, attr)
            w = MoELoRA(base, n_experts, self.rank, self.alpha)
            setattr(parent, attr, w)
            self._wrappers.append(w)
            self.adapter_names.append(full)

    @property
    def n_experts(self):
        return self.gate.n_experts

    # ---- parameter groups
    def adapter_parameters(self):
        for w in self._wrappers:
            yield from w.A
            yield from w.B

    def trainable_state_dict(self):
        sd = {}
        for name, w in zip(self.adapter_names, self._wrappers):
            for e in range(w.n_experts):
                sd[f"adapter.{name}.A.{e}"] = w.A[e].detach()
                sd[f"adapter.{name}.B.{e}"] = w.B[e].detach()
        for k, v in self.gate.state_dict().items():
            sd[f"gate.{k}"] = v.detach()
        return sd

    def load_trainable_state_dict(self, sd, strict=True):
        with torch.no_grad():
            for name, w in zip(self.adapter_names, self._wrappers):
                for e in range(w.n_experts):
                    w.A[e].copy_(sd[f"adapter.{name}.A.{e}"])
                    w.B[e].copy_(sd[f"adapter.{name}.B.{e}"])
            gsd = {k[len("gate."):]: v for k, v in sd.items() if k.startswith("gate.")}
            self.gate.load_state_dict(gsd, strict=strict)

    def backbone_state_dict(self):
        """Frozen backbone weights only (wrapper bases included, adapters excluded)."""
        return {k: v for k, v in self.backbone.state_dict().items() if ".A." not in k and ".B." not in k}

    def add_expert(self, weights, name, mode="gate_mix"):
        for w in self._wrappers:
            w.add_expert(weights, mode)
        self.gate.add_expert(weights)
        self.expert_names.append(name)

    def set_experts_trainable(self, experts, flag):
        for w in self._wrappers:
            for e in experts:
                w.set_expert_trainable(e, flag)

    def count_params(self):
        per_expert = sum(w.A[0].numel() + w.B[0].numel() for w in self._wrappers)
        return dict(backbone=sum(p.numel() for n, p in self.backbone.named_parameters() if ".A." not in n and ".B." not in n),
                    adapters_per_expert=per_expert, adapters_total=per_expert * self.n_experts,
                    gate=sum(p.numel() for p in self.gate.parameters()), n_wrapped_modules=len(self._wrappers),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone always in eval mode
        return self

    # ---- forward (mirrors Unet1D.forward)
    def gate_from_condition(self, classes):
        fm = self.backbone.encoder(classes)
        return fm, F.softmax(self.gate(pooled_features(fm, self.gate_feat)), dim=-1)

    def forward(self, x, time, classes):
        bb = self.backbone
        feature_maps, g = self.gate_from_condition(classes)
        self._last_gate = g
        for w in self._wrappers:
            w.gate = g
        x = bb.init_conv(x)
        te = bb.time_mlp(time)
        residual = x.clone()
        x, hs = bb._forward_down_blocks(x, te, feature_maps)
        x = bb._forward_middle_blocks(x, te, feature_maps[-1])
        rfm = list(reversed(feature_maps))
        x = bb._forward_up_blocks(x, te, rfm, hs)
        x = torch.cat((x, residual), dim=1)
        x = bb.final_res_block(x, te)
        x = x + bb.final_attn(x, rfm[-1])
        return bb.final_conv(x)


def load_balance_loss(g):
    """Switch-Transformer balance loss: E * sum_e f_e * P_e (= 1 at perfect balance)."""
    E = g.shape[1]
    f = F.one_hot(g.argmax(dim=1), E).float().mean(0)
    P = g.float().mean(0)
    return E * (f * P).sum()


def routing_counts(g):
    return torch.bincount(g.argmax(dim=1), minlength=g.shape[1]).tolist()
