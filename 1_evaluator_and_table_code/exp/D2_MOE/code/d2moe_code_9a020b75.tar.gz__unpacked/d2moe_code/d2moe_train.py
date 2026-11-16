"""D2: adapter experts + gate on the frozen M5 joint ms9 backbone (Nagamatsu Unet1D).

Stage 1  (--stage 1): E experts (default 3) + gate trained on the non-Urdu scripts (default casia_zh casia_en
         hpl_tamil iam) of the train npz.  loss = L_den + lb_coef * L_lb  (L_den = Nagamatsu p_losses, pred_noise;
         L_lb = Switch load-balance penalty).  Only adapters + gate are trainable; the backbone is frozen (eval mode).
Stage 2  (--stage 2 --init-from-stage1 CKPT): adds one new 'rtl' expert initialised from the gate-weighted mix of the
         stage-1 experts (mix weights = mean stage-1 gate on --gate-init-sample Urdu units; rank-r SVD of the mixed
         delta) plus a matching gate row, then trains on a Urdu fraction p (--urdu-frac p of the Urdu units in the
         train file, seeded, or an explicit --train-npz such as m5_ft_ouhdl_p10.npz) with the old experts frozen
         (--old-experts frozen, default) or trainable; the gate is trainable unless --no-train-gate.
Checkpoints every --save-every steps in --run-dir/checkpoints (adapters + gate + their EMA + optimiser + step;
backbone sha256 recorded, never saved), auto-resume from the latest one (seed = --seed + step).  At every
validation the mean gate vector and argmax histogram per script are written to run_dir/gate_weights/step_*.json
and appended to gate_log.csv.  EMA of the trainable parameters follows ema_pytorch semantics (beta 0.995,
update every 10, update_after_step 100, warm-up (1+n)^-2/3) and is used for sampling.

python d2moe_train.py --stage 1 --run-dir RUN --backbone m5joint_ms9.pt --train-npz m5_joint_train.npz \
    --scripts casia_zh casia_en hpl_tamil iam --val-npz casia_zh=... --n-experts 3 --rank 16 --lb-coef 0.01 \
    --batch-size 64 --grad-accum 4 --train-steps 20000 --save-every 1000
python d2moe_train.py --stage 2 --run-dir RUN2 --backbone m5joint_ms9.pt --init-from-stage1 RUN/checkpoints/checkpoint_000020.pt \
    --train-npz m5_joint_train.npz --scripts ouhdl --urdu-frac 0.10 --val-npz ouhdl=... [--old-experts trainable]
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uhwr_route_common as U  # noqa: E402
from d2moe_model import MoEUnet1D, load_balance_loss, routing_counts  # noqa: E402

CODE_FILES = [HERE / "d2moe_train.py", HERE / "d2moe_model.py", HERE / "d2moe_sample.py", HERE / "uhwr_route_common.py"]
STAGE1_SCRIPTS = ["casia_zh", "casia_en", "hpl_tamil", "iam"]


class ParamEMA:
    """EMA over a dict of parameters, ema_pytorch semantics (beta, update_every, update_after_step, inv_gamma=1, power=2/3)."""

    def __init__(self, params, beta=0.995, update_every=10, update_after_step=100, power=2 / 3):
        self.params = params
        self.shadow = {k: p.detach().clone().float() for k, p in params.items()}
        self.beta, self.update_every, self.update_after_step, self.power = beta, update_every, update_after_step, power
        self.step, self.initted = 0, False

    def current_decay(self):
        epoch = self.step - self.update_after_step - 1
        if epoch <= 0:
            return 0.0
        return min(max(1 - (1 + epoch) ** (-self.power), 0.0), self.beta)

    @torch.no_grad()
    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        if self.step <= self.update_after_step or not self.initted:
            for k, p in self.params.items():
                self.shadow[k].copy_(p.detach().float())
            self.initted = True
            return
        d = self.current_decay()
        for k, p in self.params.items():
            self.shadow[k].lerp_(p.detach().float(), 1 - d)

    def state_dict(self):
        return dict(shadow={k: v.cpu() for k, v in self.shadow.items()}, step=self.step, initted=self.initted)

    def load_state_dict(self, sd):
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
        self.step, self.initted = int(sd["step"]), bool(sd["initted"])

    def rebind(self, params):
        """After add_expert: keep existing shadows, add new keys."""
        self.params = params
        for k, p in params.items():
            if k not in self.shadow:
                self.shadow[k] = p.detach().clone().float()


def named_trainable(moe):
    return {k: v for k, v in moe.trainable_state_dict().items()}


def live_params(moe):
    """name -> live Parameter (same naming as trainable_state_dict)."""
    out = {}
    for name, w in zip(moe.adapter_names, moe._wrappers):
        for e in range(w.n_experts):
            out[f"adapter.{name}.A.{e}"] = w.A[e]
            out[f"adapter.{name}.B.{e}"] = w.B[e]
    for k, p in moe.gate.named_parameters():
        out[f"gate.{k}"] = p
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--backbone", type=Path, required=True, help="M5 joint ms9 checkpoint (full or fp16 extract)")
    ap.add_argument("--backbone-weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--train-npz", type=Path, nargs="+", required=True)
    ap.add_argument("--scripts", nargs="*", help="script tags kept from the train files (default: stage 1 = non-Urdu, stage 2 = ouhdl)")
    ap.add_argument("--script-override", help="script tag for train files without a 'scripts' array (e.g. m5_ft_ouhdl_p10.npz)")
    ap.add_argument("--urdu-frac", type=float, help="stage 2: fraction p of the (Urdu) train units kept (seeded)")
    ap.add_argument("--max-units-per-script", type=int, help="smoke: cap units per script")
    ap.add_argument("--val-npz", nargs="*", action="extend", default=[], help="script=path; repeatable")
    ap.add_argument("--condition", default="rendered")
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-targets", default="cross,attn,linear")
    ap.add_argument("--gate-hidden", type=int, default=128)
    ap.add_argument("--gate-feat", choices=("deepest", "all"), default="deepest")
    ap.add_argument("--lb-coef", type=float, default=0.01)
    ap.add_argument("--init-from-stage1", type=Path, help="stage 2: stage-1 checkpoint")
    ap.add_argument("--new-expert-name", default="rtl")
    ap.add_argument("--new-expert-init", choices=("gate_mix", "fresh"), default="gate_mix")
    ap.add_argument("--gate-init-sample", type=int, default=256, help="Urdu units used for the mix weights")
    ap.add_argument("--old-experts", choices=("frozen", "trainable"), default="frozen")
    ap.add_argument("--train-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=20_000)
    ap.add_argument("--save-every", type=int, default=1_000)
    ap.add_argument("--val-every", type=int, default=1_000)
    ap.add_argument("--val-max-units", type=int, default=250)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.99))
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-update-every", type=int, default=10)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--keep-last", type=int, default=0)
    ap.add_argument("--keep-every", type=int, default=10_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear LR warm-up from 0 over this many steps (0 = none, E0 behaviour)")
    ap.add_argument("--smoke-fixed-batch", action="store_true", help="smoke: reuse one batch and fixed (t, noise) every step")
    ap.add_argument("--notes", default="D2 adapter experts + gate")
    return ap.parse_args()


def build_moe(a, dev):
    backbone, binfo = U.load_unet_from_checkpoint(a.backbone, which=a.backbone_weights)
    binfo["file_sha256"] = U.sha256_file(a.backbone)
    moe = MoEUnet1D(backbone, n_experts=a.n_experts, rank=a.rank, alpha=a.lora_alpha,
                    targets=tuple(a.lora_targets.split(",")), gate_hidden=a.gate_hidden, gate_feat=a.gate_feat)
    return moe.to(dev), binfo


def main():
    a = parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    on_cuda = dev.type == "cuda"
    use_amp = a.amp and on_cuda
    amp_dtype = torch.float16 if a.amp_dtype == "fp16" else torch.bfloat16
    run_dir = a.run_dir.expanduser().resolve()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_weights").mkdir(exist_ok=True)
    resume_ms = U.latest_milestone(run_dir)
    seed_now = a.seed
    U.seed_all(seed_now)
    scripts = a.scripts if a.scripts else (STAGE1_SCRIPTS if a.stage == 1 else ["ouhdl"])

    # ---- data
    ds = U.MultiScriptDataset(a.train_npz, a.condition, scripts_keep=scripts, script_override=a.script_override,
                              max_units_per_script=a.max_units_per_script, seed=a.seed)
    urdu_subset = None
    if a.stage == 2 and a.urdu_frac is not None:
        n = len(ds)
        k = int(np.floor(a.urdu_frac * n + 0.5))
        perm = np.random.default_rng([20261016, 9, a.seed]).permutation(n)[:k]
        sel = np.sort(perm)
        urdu_subset = dict(frac=a.urdu_frac, n_pool=n, n_kept=int(k), rule="default_rng([20261016,9,seed]).permutation(n)[:k], k=floor(p*n+0.5)")
        ds.cond, ds.seq = ds.cond[sel], [ds.seq[i] for i in sel]
        ds.unit_ids, ds.scripts, ds.writer = ds.unit_ids[sel], ds.scripts[sel], ds.writer[sel]
    counts = ds.script_counts()
    print(f"stage {a.stage}: train units {len(ds)} per script {counts}", flush=True)
    val_sets = {}
    for item in a.val_npz:
        s, p = item.split("=", 1)
        val_sets[s] = U.MultiScriptDataset(p, a.condition, script_override=s, max_units_per_script=a.val_max_units, seed=1234)

    # ---- model (architecture of the adapters/gate comes from the checkpoint when resuming / continuing)
    peek = None
    if resume_ms is not None:
        peek = torch.load(str(U.checkpoint_file(run_dir, resume_ms)), map_location="cpu", weights_only=False)["config"]
    elif a.stage == 2 and a.init_from_stage1 is not None:
        peek = torch.load(str(a.init_from_stage1), map_location="cpu", weights_only=False)["config"]
    elif a.stage == 2:
        raise SystemExit("--stage 2 needs --init-from-stage1 (or an existing run dir to resume)")
    if peek is not None:
        for k_cli, k_cfg in [("n_experts", "n_experts"), ("rank", "rank"), ("lora_alpha", "lora_alpha"), ("lora_targets", "lora_targets"),
                             ("gate_hidden", "gate_hidden"), ("gate_feat", "gate_feat")]:
            setattr(a, k_cli, peek[k_cfg])
    moe, binfo = build_moe(a, dev)
    if peek is not None:
        moe.expert_names = list(peek["expert_names"])
    backbone_sha0 = U.sha256_state(moe.backbone_state_dict())
    diff = U.build_diffusion(moe).to(dev)
    stage2_info = None
    if a.stage == 2 and resume_ms is None:
        s1 = torch.load(str(a.init_from_stage1), map_location="cpu", weights_only=False)
        if s1["config"]["backbone_param_sha256"] != binfo["param_sha256"]:
            raise SystemExit("stage-1 checkpoint was trained on a different backbone")
        moe.load_trainable_state_dict(s1["ema_trainable"] if "ema_trainable" in s1 else s1["trainable"])
        # mix weights = mean stage-1 gate on a sample of the (Urdu) training units
        moe.eval()
        idx = np.random.default_rng(a.seed).choice(len(ds), min(a.gate_init_sample, len(ds)), replace=False)
        with torch.no_grad():
            imgs = torch.stack([ds[int(i)]["img"] for i in idx]).to(dev)
            _, g = moe.gate_from_condition(imgs)
        mix = g.mean(0).cpu()
        moe.add_expert(mix, a.new_expert_name, a.new_expert_init)
        moe.to(dev)
        stage2_info = dict(init_from_stage1=str(a.init_from_stage1), stage1_sha256=U.sha256_file(a.init_from_stage1),
                           stage1_step=int(s1["step"]), mix_weights={n: float(v) for n, v in zip(moe.expert_names[:-1], mix)},
                           mix_sample_units=int(len(idx)), new_expert=a.new_expert_name, new_expert_init=a.new_expert_init,
                           old_experts=a.old_experts, train_gate=a.train_gate)
        print(f"stage 2: added expert '{a.new_expert_name}' from mix {stage2_info['mix_weights']}", flush=True)
        del s1
    if a.stage == 2:
        old = list(range(moe.n_experts - 1))
        moe.set_experts_trainable(old, a.old_experts == "trainable")
        for p in moe.gate.parameters():
            p.requires_grad_(a.train_gate)
    diff.train()
    pcount = moe.count_params()
    params = live_params(moe)
    train_params = [p for p in params.values() if p.requires_grad]
    opt = Adam(train_params, lr=a.learning_rate, betas=tuple(a.adam_betas))
    ema = ParamEMA(params, beta=a.ema_decay, update_every=a.ema_update_every)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = 0
    if resume_ms is not None:
        ck = torch.load(str(U.checkpoint_file(run_dir, resume_ms)), map_location="cpu", weights_only=False)
        if ck["config"]["backbone_param_sha256"] != binfo["param_sha256"]:
            raise SystemExit("resume: backbone differs from the checkpoint's backbone")
        moe.load_trainable_state_dict(ck["trainable"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        if ck.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step = int(ck["step"])
        stage2_info = ck["config"].get("stage2")
        seed_now = (a.seed + step) % (2 ** 32)
        U.seed_all(seed_now)
        print(f"resumed milestone {resume_ms} step {step} seed {seed_now}", flush=True)
        del ck
    moe.to(dev)

    per_file, code_sha = U.code_sha256(CODE_FILES)
    config = dict(n_experts=moe.n_experts, rank=a.rank, lora_alpha=a.lora_alpha, lora_targets=a.lora_targets,
                  gate_hidden=a.gate_hidden, gate_feat=a.gate_feat, expert_names=list(moe.expert_names),
                  backbone=str(a.backbone), backbone_weights=a.backbone_weights, backbone_file_sha256=binfo["file_sha256"],
                  backbone_param_sha256=binfo["param_sha256"], backbone_step=binfo["step"], stage=a.stage, stage2=stage2_info,
                  adapter_names=list(moe.adapter_names), arch=U.E0_ARCH)
    cfg = dict(schema="d2moe_train/1", kind="training", run_dir=str(run_dir), created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, config=config,
               code_sha256=per_file, code_sha256_combined=code_sha, environment=U.environment_record(), device=str(dev),
               amp=use_amp, amp_dtype=str(amp_dtype), params=pcount, train_files=ds.files, train_units_per_script=counts,
               urdu_subset=urdu_subset, resume_milestone=resume_ms, resumed_step=step, seed_used=seed_now,
               effective_batch=a.batch_size * a.grad_accum, backbone_sha256_at_start=backbone_sha0,
               train_npz_sha256={str(p): U.sha256_file(p) for p in a.train_npz},
               val_sets={s: dict(n=len(v), files=v.files) for s, v in val_sets.items()})
    name = "run_config.json" if not (run_dir / "run_config.json").exists() else f"run_config_resume_{step:07d}.json"
    U.write_json_atomic(run_dir / name, cfg)
    log = U.JsonlLogger(run_dir / "train_log.jsonl")
    print(f"params {pcount}; experts {moe.expert_names}; trainable tensors {len(train_params)}; eff. batch {a.batch_size * a.grad_accum}", flush=True)

    dl = U.cycle(U.make_loader(ds, a.batch_size, True, a.num_workers, seed_now))

    fixed = {}

    def den_loss(coords, imgs, t=None, noise=None):
        b = coords.shape[0]
        if t is None:
            t = torch.randint(0, diff.num_timesteps, (b,), device=dev).long()
        x0 = diff.normalize(coords)
        if noise is None:
            noise = torch.randn_like(x0)
        if a.smoke_fixed_batch and "t" in fixed and fixed["t"].shape[0] == b:
            t, noise = fixed["t"], fixed["noise"]
        elif a.smoke_fixed_batch and "t" not in fixed:
            fixed["t"], fixed["noise"] = t, noise
        x_t = diff.q_sample(x_start=x0, t=t, noise=noise)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = diff.model(x_t, t, imgs)
        out = out.float()
        l = F.mse_loss(out, noise, reduction="none").mean(dim=(1, 2)) * diff.loss_weight[t]
        return l, moe._last_gate

    def swap_in_ema():
        saved = {k: p.detach().clone() for k, p in params.items()}
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(ema.shadow[k].to(p.dtype))
        return saved

    def swap_back(saved):
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(saved[k])

    @torch.no_grad()
    def validate(step_now):
        moe.eval()
        res = {}
        gate_rows = []
        for s, vds in val_sets.items():
            g_cpu = torch.Generator(device="cpu").manual_seed(1234)
            tot, n, gsum, hist = 0.0, 0, torch.zeros(moe.n_experts), torch.zeros(moe.n_experts, dtype=torch.long)
            for d in U.make_loader(vds, a.val_batch_size, False, 0, 1234, drop_last=False):
                coords, imgs = d["coords"].to(dev), d["imgs"].to(dev)
                b = coords.shape[0]
                t = torch.randint(0, diff.num_timesteps, (b,), generator=g_cpu).to(dev)
                noise = torch.randn(coords.shape, generator=g_cpu).to(dev)
                l, g = den_loss(coords, imgs, t, noise)
                tot += l.sum().item()
                n += b
                gsum += g.float().sum(0).cpu()
                hist += torch.bincount(g.argmax(1).cpu(), minlength=moe.n_experts)
            mean_gate = (gsum / max(n, 1)).tolist()
            res[s] = dict(n=n, val_den=tot / max(n, 1), mean_gate={e: round(v, 5) for e, v in zip(moe.expert_names, mean_gate)},
                          argmax_hist={e: int(v) for e, v in zip(moe.expert_names, hist.tolist())})
            for e, v, h in zip(moe.expert_names, mean_gate, hist.tolist()):
                gate_rows.append([step_now, s, e, round(v, 6), int(h), n])
        U.write_json_atomic(run_dir / "gate_weights" / f"step_{step_now:07d}.json", dict(step=step_now, experts=moe.expert_names, per_script=res))
        new = not (run_dir / "gate_log.csv").exists()
        with open(run_dir / "gate_log.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["step", "script", "expert", "mean_gate", "argmax_count", "n_units"])
            w.writerows(gate_rows)
        log.log(event="val", step=step_now, **res)
        print(f"[val step {step_now}] " + " | ".join(f"{s}: den {v['val_den']:.4f} gate {v['mean_gate']}" for s, v in res.items()), flush=True)
        moe.train()
        return res

    def save(step_now):
        ms = step_now // a.save_every
        bsha = U.sha256_state(moe.backbone_state_dict())
        data = {"format_version": "d2moe/1", "milestone": ms, "step": step_now, "trainable": {k: v.cpu() for k, v in named_trainable(moe).items()},
                "ema_trainable": {k: v.cpu() for k, v in ema.shadow.items()}, "ema": ema.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None, "config": config,
                "backbone_sha256_now": bsha, "backbone_sha256_at_start": backbone_sha0, "code_sha256_combined": code_sha,
                "seed_used": seed_now, "gpu": torch.cuda.get_device_name(0) if on_cuda else "cpu"}
        U.save_torch_atomic(data, U.checkpoint_file(run_dir, ms))
        removed = U.prune_checkpoints(run_dir, a.keep_last, a.keep_every, a.save_every)
        log.log(event="checkpoint", step=step_now, milestone=ms, backbone_unchanged=(bsha == backbone_sha0), removed=removed)
        if bsha != backbone_sha0:
            print("WARNING: backbone weights changed!", flush=True)

    if step == 0 and val_sets:
        validate(0)
    moe.train()
    t_last = time.time()
    while step < a.train_steps:
        tot = dict(loss=0.0, den=0.0, lb=0.0)
        rc_step = Counter()
        gate_by_script = {}
        for _ in range(a.grad_accum):
            if a.smoke_fixed_batch:
                d = fixed.setdefault("batch", next(dl))
            else:
                d = next(dl)
            coords, imgs = d["coords"].to(dev, non_blocking=True), d["imgs"].to(dev, non_blocking=True)
            l, g = den_loss(coords, imgs)
            l_den = l.mean()
            l_lb = load_balance_loss(g)
            loss = (l_den + a.lb_coef * l_lb) / a.grad_accum
            scaler.scale(loss).backward()
            tot["loss"] += loss.item()
            tot["den"] += l_den.item() / a.grad_accum
            tot["lb"] += l_lb.item() / a.grad_accum
            rc_step.update({moe.expert_names[e]: c for e, c in enumerate(routing_counts(g))})
            for s in set(d["scripts"]):
                m = torch.tensor([x == s for x in d["scripts"]], device=g.device)
                gate_by_script.setdefault(s, []).append(g[m].detach().float().mean(0).cpu())
        if a.warmup_steps and step < a.warmup_steps:
            for grp in opt.param_groups:
                grp["lr"] = a.learning_rate * (step + 1) / a.warmup_steps
        elif a.warmup_steps and step == a.warmup_steps:
            for grp in opt.param_groups:
                grp["lr"] = a.learning_rate
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(train_params, a.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        ema.update()
        if step % a.log_every == 0 or step <= 5:
            now = time.time()
            rec = dict(event="train", step=step, loss=tot["loss"], loss_den=tot["den"], loss_lb=tot["lb"], grad_norm=float(gn),
                       lr=opt.param_groups[0]["lr"], sec_per_step=(now - t_last) / (a.log_every if step % a.log_every == 0 else 1),
                       routing=dict(rc_step), gate_by_script={s: [round(x, 4) for x in torch.stack(v).mean(0).tolist()] for s, v in gate_by_script.items()},
                       peak_rss_mb=U.peak_rss_mb())
            if on_cuda:
                rec["cuda_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log.log(**rec)
            print(f"step {step} loss {tot['loss']:.4f} den {tot['den']:.4f} lb {tot['lb']:.4f} gn {float(gn):.2f} route {dict(rc_step)} {rec['sec_per_step']:.2f}s/step", flush=True)
            t_last = now
        if step % a.val_every == 0 and val_sets:
            validate(step)
        if step % a.save_every == 0:
            save(step)
    if step % a.save_every != 0:
        save(step)
    bsha = U.sha256_state(moe.backbone_state_dict())
    U.write_json_atomic(run_dir / "training_complete.json", dict(status="complete", final_step=step, backbone_sha256_at_start=backbone_sha0,
                        backbone_sha256_end=bsha, backbone_unchanged=(bsha == backbone_sha0), completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        peak_rss_mb=U.peak_rss_mb(), cuda_max_alloc_gb=torch.cuda.max_memory_allocated() / 1e9 if on_cuda else None, params=pcount))
    print(f"training complete; backbone unchanged: {bsha == backbone_sha0}", flush=True)


if __name__ == "__main__":
    main()
