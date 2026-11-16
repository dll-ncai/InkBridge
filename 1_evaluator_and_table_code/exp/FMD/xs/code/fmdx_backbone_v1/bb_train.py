"""FMDX backbone trainer (track 'Backbone build', 3 Nov 2026): multi-script Urdu-left-out (LUO) backbone with the 224-px
FM-sized encoder, optional REPA alignment to one FM, optional pseudo-label source, and the p01 fine-tune regime.

Regimes (--regime):
  warm     U-Net = LUO ms6 EMA weights for every non-encoder module (online AND EMA <- ck['ema'], prereg A4.2),
           encoder = fresh (vits14, seeded) or PE-Spatial-S16 tower (pe_spatial_s16, A4.2 PEINIT); fresh Adam, step 0.
  scratch  U-Net built right after torch.manual_seed(seed) (the LUO m5_train.py order: seed -> Unet1D(...)), encoder replaced.
  ft       U-Net + encoder = a backbone milestone file bb_ms<k>.pt (online <- model_unet, EMA <- ema_unet); fresh Adam.
Recipe (LUO / Trainer1D + prereg A4.2): Adam betas (0.9, 0.99); two parameter groups: (i) the new encoder, its level heads
and the REPA projector: lr --enc-lr 2e-4 with a --enc-warmup 200-step linear warm-up (fmd_student_v1 values); (ii) every
other parameter: lr 8e-5, no warm-up (LUO). The ft regime (COMP-p01 recipe) uses one lr 8e-5 for all (--enc-lr 8e-5
--enc-warmup 0).
global batch 128 split over the ranks x grad-accum 4 (= 512 units per optimiser step), fp16 autocast + GradScaler, grad-clip
1.0 on the diffusion parameters, EMA (ema_pytorch, beta 0.995, update_every 10, rank 0), train_fraction 0.99 random_split
seed 0 (Trainer1D) -> 1 % held out for a validation loss every --val-every steps, seed 20261016.
Loss = GaussianDiffusion1D(coords, classes=imgs) (the repo's p_losses, unchanged)  [+ repa_lambda * REPA]
REPA = mean over the units of the micro-batch that HAVE a precomputed target of mean_tokens(-cos(projector(encoder
final-norm tokens), FM gate-layer tokens on the 16x16 grid)); units without a target contribute the denoising loss only.
Projector = Linear(384,2048)-SiLU-Linear(2048,2048)-SiLU-Linear(2048,D_fm), seed 20261105, own grad-clip 1.0, same Adam;
never written into a milestone/inference file.
Gate (A4.3 c): --verdict-file (lane mirror of exp/FMD/xs/verdicts/verdict_<fm>.json) is read at start (also after every
requeue) and at every --verdict-every (10,000) step save; status FAIL -> file GATE_FAIL ({fm, step, verdict sha256, utc})
in the run dir, checkpoints kept, exit 0. Any other status (PENDING, PASS_XS, PASS_URDU_ONLY) or a missing file -> continue.
Checkpoints: checkpoints/latest.pt (full resume state) every --resume-every steps; milestone files bb_ms<k>.pt (online +
EMA U-Net weights, config; no optimiser, no projector, no FM tensor) every --save-every steps (k = step / save-every).
Requeue-safe: a restarted process resumes from latest.pt with seed = seed + step (E0 change 3).
Launch: torchrun --nproc_per_node 2 bb_train.py ...   (1 process without torchrun)
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE))
import uhwr_route_common as U  # noqa: E402  (byte-identical copy, loop_code_944a7401 via fmd_student_v1)
import bb_model as BM          # noqa: E402
import bb_data as BD           # noqa: E402

PROJ_SEED = 20261105
CODE_FILES = ["bb_train.py", "bb_model.py", "bb_data.py", "vendor/student_model.py", "vendor/uhwr_route_common.py"]
FORMAT_TRAIN, FORMAT_MS = "fmdx_bb_train/1", "fmdx_bb/1"


def sha_file(p):
    return U.sha256_file(p)


def read_verdict(path, key="status"):
    """-> dict(status in PASS|FAIL|PENDING|MISSING|UNREADABLE, sha256, path)."""
    if path is None:
        return dict(status="NOT_GATED", path=None, sha256=None)
    p = Path(path)
    if not p.is_file():
        return dict(status="MISSING", path=str(p), sha256=None)
    try:
        d = json.loads(p.read_text())
        st = str(d.get(key, "PENDING")).strip().upper()
        return dict(status=st, path=str(p), sha256=sha_file(p), fm=d.get("fm"))
    except Exception as e:  # a half-written file is treated as pending and re-read next time
        return dict(status="UNREADABLE", path=str(p), sha256=None, error=repr(e))


def write_gate_fail(run_dir, a, step, vrec):
    U.write_json_atomic(run_dir / "GATE_FAIL", dict(fm=a.fm, step=int(step), verdict_sha256=vrec.get("sha256"), verdict_status=vrec.get("status"),
                        verdict_path=vrec.get("path"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), arm_id=a.arm_id,
                        slurm=os.environ.get("SLURM_JOB_ID"), lane=a.lane, note="prereg A4.3(c): checkpoints kept, exit 0"))


class Projector(nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    """DDP unit: diffusion (+ projector). forward -> (denoising loss, REPA loss or None, n units with a target)."""

    def __init__(self, diff, proj=None):
        super().__init__()
        self.diff, self.proj = diff, proj

    def forward(self, coords, imgs, tgt=None, mask=None):
        den = self.diff(coords, classes=imgs)
        if self.proj is None:
            return den, None
        h = self.proj(self.diff.model.encoder.last_tokens.float())
        cos = F.cosine_similarity(h.float(), tgt.float(), dim=-1).mean(-1)          # (b,)
        rep = -(cos * mask).sum() / mask.sum().clamp(min=1.0)                        # 0 with zero grads if no target
        return den, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--regime", choices=("warm", "scratch", "ft"), required=True)
    ap.add_argument("--encoder", choices=BM.ENCODERS, default="vits14")
    ap.add_argument("--pe-weights", type=Path, help="PE-Spatial-S16 weights (dir or file) for --encoder pe_spatial_s16 (warm$SCRATCH_ROOT)")
    ap.add_argument("--init-ckpt", type=Path, help="warm: LUO ms6 Trainer1D checkpoint; ft: bb_ms<k>.pt")
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--render-npy", type=Path, required=True)
    ap.add_argument("--pseudo-npz", type=Path); ap.add_argument("--pseudo-render-npy", type=Path)
    ap.add_argument("--pseudo-frac", type=float, default=0.0, help="0 = pooled with the labelled units (A4.2); >0 = fixed share")
    ap.add_argument("--fm", default=None, help="FM name (REPA and/or gate); None = not gated")
    ap.add_argument("--repa-targets", type=Path, help=".npy (M,256,D) float16, rows = --repa-unit-ids order")
    ap.add_argument("--repa-unit-ids", type=Path, help=".npy unit ids of the target rows")
    ap.add_argument("--repa-lambda", type=float, default=0.5)
    ap.add_argument("--verdict-file", type=Path)
    ap.add_argument("--verdict-key", default="status")
    ap.add_argument("--verdict-every", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=128, help="global micro-batch (split over ranks)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=8e-5)
    ap.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.99))
    ap.add_argument("--warmup-steps", type=int, default=0, help="warm-up of the non-encoder group (LUO: 0)")
    ap.add_argument("--enc-lr", type=float, default=2e-4, help="lr of the encoder+heads+projector group (A4.2)")
    ap.add_argument("--enc-warmup", type=int, default=200)
    ap.add_argument("--train-steps", type=int, required=True)
    ap.add_argument("--save-every", type=int, required=True)
    ap.add_argument("--resume-every", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-update-every", type=int, default=10)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--train-fraction", type=float, default=0.99)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-units", type=int, help="smoke only: first N units of the main source")
    ap.add_argument("--stop-after", type=int, help="smoke only: exit (no completion marker) after this many steps in this process")
    ap.add_argument("--lane", default="clusterA")
    a = ap.parse_args()

    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, lrank = 0, 1, 0
    on_cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{lrank}" if on_cuda else "cpu")
    if on_cuda:
        torch.cuda.set_device(dev)
    main0 = rank == 0
    use_amp = a.amp and on_cuda
    run_dir = a.run_dir.expanduser().resolve()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = U.JsonlLogger(run_dir / "train_log.jsonl") if main0 else None

    def P(*x):
        if main0:
            print(*x, flush=True)

    def bcast_flag(v):
        t = torch.tensor([1 if v else 0], device=dev)
        if ddp:
            dist.broadcast(t, 0)
        return bool(t.item())

    def finish(code=0):
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)          # skip interpreter teardown (persistent DataLoader workers over an infinite sampler, memmaps)

    if (run_dir / "training_complete.json").exists():
        P(f"training_complete.json present in {run_dir}; nothing to do"); finish(0)
    if (run_dir / "GATE_FAIL").exists():
        P(f"GATE_FAIL present in {run_dir}; nothing to do"); finish(0)
    for m in ("transformers", "diffusers"):
        assert m not in sys.modules, f"{m} imported in the backbone trainer"
    if a.repa_targets is not None and a.fm is None:
        raise SystemExit("--repa-targets needs --fm")
    if a.batch_size % world:
        raise SystemExit(f"global batch {a.batch_size} not divisible by {world} ranks")

    # ---- gate check at start
    vrec = read_verdict(a.verdict_file, a.verdict_key) if a.fm else read_verdict(None)
    fail = bcast_flag(main0 and vrec["status"] == "FAIL")
    if fail:
        if main0:
            write_gate_fail(run_dir, a, 0, vrec)
            log.log(event="gate_fail", step=0, verdict=vrec)
        P(f"GATE_FAIL: verdict {vrec}"); finish(0)
    if main0:
        log.log(event="gate_check", step=0, verdict=vrec)

    # ---- data
    src = [BD.RenderSource(a.train_npz, a.render_npy, "main", max_units=a.max_units)]
    tr_pos, va_pos = BD.trainer1d_split(len(src[0].unit_ids), a.train_fraction, a.split_seed)
    if a.max_units is not None:
        tr_pos, va_pos = tr_pos[tr_pos < src[0].n], va_pos[va_pos < src[0].n]
    positions, fracs = [tr_pos], [1.0]
    if a.pseudo_npz is not None:
        src.append(BD.RenderSource(a.pseudo_npz, a.pseudo_render_npy, "pseudo"))
        if a.pseudo_frac > 0:        # fixed share per batch (not used by the pre-registered arms)
            positions.append(np.arange(src[1].n)); fracs = [1.0 - a.pseudo_frac, a.pseudo_frac]
        else:                        # A4.2: kept pseudo units POOLED with the labelled training units (one uniform stream)
            positions = [np.concatenate([tr_pos, 10 ** 9 + np.arange(src[1].n)])]
    target_rows, T, trec = None, None, None
    if a.repa_targets is not None:
        T = np.load(a.repa_targets, mmap_mode="r")
        tu = np.load(a.repa_unit_ids, allow_pickle=False).astype(str)
        assert T.ndim == 3 and T.shape[1] == 256 and T.shape[0] == len(tu), (T.shape, len(tu))
        row = {u: i for i, u in enumerate(tu)}
        tr_map = np.array([row.get(u, -1) for u in src[0].unit_ids], np.int64)
        target_rows = {0: tr_map}
        n_cov = int((tr_map[tr_pos] >= 0).sum())
        trec = dict(path=str(a.repa_targets), shape=list(T.shape), dtype=str(T.dtype), unit_ids=str(a.repa_unit_ids),
                    n_train_units_with_target=n_cov, n_train_units=int(len(tr_pos)))
        if main0:
            trec.update(sha256=sha_file(a.repa_targets), unit_ids_sha256=sha_file(a.repa_unit_ids))
    ds = BD.MixDataset(src, target_rows)

    tds = TargetDS(ds, str(a.repa_targets) if a.repa_targets is not None else None)
    coll = functools.partial(collate_targets, with_targets=T is not None)

    # ---- model
    U.seed_all(a.seed)                     # scratch: same order as m5_train.py (seed -> Unet1D)
    pe_w = a.pe_weights if (a.encoder == "pe_spatial_s16" and a.regime != "ft") else None
    if a.encoder == "pe_spatial_s16" and a.regime != "ft" and pe_w is None:
        raise SystemExit("--encoder pe_spatial_s16 needs --pe-weights")
    init_rec, ema_init_sd = None, None
    if a.regime == "ft":
        ck = torch.load(str(a.init_ckpt), map_location="cpu", weights_only=False)
        assert ck["format_version"] == FORMAT_MS, ck.get("format_version")
        a.encoder = ck["config"]["encoder"]
        unet, info = BM.build_backbone_unet(U, a.encoder, None)
        unet.load_state_dict(ck["model_unet"], strict=True)
        ema_init_sd = ck["ema_unet"]
        init_rec = dict(kind="bb_milestone", path=str(a.init_ckpt), sha256=sha_file(a.init_ckpt) if main0 else None,
                        source_arm=ck["config"].get("arm_id"), source_step=int(ck["step"]), loaded=["model_unet->online", "ema_unet->ema"])
        del ck
    else:
        unet, info = BM.build_backbone_unet(U, a.encoder, pe_w)
        if a.regime == "warm":
            esd, minfo = U.load_checkpoint_weights(a.init_ckpt, "ema")
            n = BM.load_non_encoder(unet, esd)
            ema_init_sd = {k: v for k, v in esd.items() if not k.startswith("encoder.")}
            init_rec = dict(kind="luo_trainer1d", path=str(a.init_ckpt), sha256=sha_file(a.init_ckpt) if main0 else None,
                            source_step=minfo["step"], n_non_encoder_keys=n, loaded=["ema(non-encoder)->online", "ema(non-encoder)->ema"],
                            encoder="fresh" if a.encoder == "vits14" else "PE-Spatial-S16 trunk + fresh heads")
            del esd
        else:
            init_rec = dict(kind="scratch", seed=a.seed, note="torch.manual_seed(seed) -> Unet1D(E0_ARCH) -> encoder replaced")
    diff = U.build_diffusion(unet).to(dev)
    proj = None
    if T is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(PROJ_SEED)
            proj = Projector(BM.SM.VIT_S14["dim"], int(T.shape[2]))
        proj = proj.to(dev)
    net = Net(diff, proj).to(dev)
    model_params = list(diff.parameters())
    enc_ids = {id(p) for p in unet.encoder.parameters()}
    base_params = [p for p in model_params if id(p) not in enc_ids]
    enc_params = list(unet.encoder.parameters()) + (list(proj.parameters()) if proj is not None else [])
    groups = [{"params": base_params, "lr": a.learning_rate, "base_lr": a.learning_rate, "warmup": a.warmup_steps, "name": "unet"},
              {"params": enc_params, "lr": a.enc_lr, "base_lr": a.enc_lr, "warmup": a.enc_warmup, "name": "encoder+heads+projector"}]
    opt = Adam(groups, lr=a.learning_rate, betas=tuple(a.adam_betas))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = None
    if main0:
        from ema_pytorch import EMA
        ema = EMA(diff, beta=a.ema_decay, update_every=a.ema_update_every).to(dev)
        if ema_init_sd is not None:
            sd = ema.ema_model.model.state_dict()
            miss = [k for k in sd if k not in ema_init_sd and not k.startswith("encoder.") and not k.endswith("div_term")]
            if a.regime == "ft":
                miss = [k for k in sd if k not in ema_init_sd and not k.endswith("div_term")]
            assert not miss, miss[:5]
            ema.ema_model.model.load_state_dict({**sd, **{k: v for k, v in ema_init_sd.items() if k in sd}}, strict=True)
    ddp_net = (DDP(net, device_ids=[lrank] if on_cuda else None, find_unused_parameters=(a.encoder == "pe_spatial_s16"))
               if ddp else net)   # timm PE trunk carries modules forward_features does not use

    # ---- resume
    step = 0
    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.exists():
        ck = torch.load(str(latest), map_location="cpu", weights_only=False)
        assert ck["format_version"] == FORMAT_TRAIN
        diff.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        if proj is not None:
            proj.load_state_dict(ck["projector"])
        if main0:
            ema.load_state_dict(ck["ema"])
        step = int(ck["step"])
        del ck
    seed_now = (a.seed + step) % (2 ** 32)
    U.seed_all(seed_now + rank)            # per-rank noise streams (timesteps / noise)
    sampler = BD.RankBatchSampler(positions, a.batch_size, rank, world, seed_now, fracs)
    dl = torch.utils.data.DataLoader(tds, batch_sampler=sampler, num_workers=a.num_workers, collate_fn=coll,
                                     pin_memory=on_cuda, persistent_workers=a.num_workers > 0,
                                     prefetch_factor=4 if a.num_workers > 0 else None)
    it = iter(dl)

    per_file, csha = U.code_sha256([HERE / f for f in CODE_FILES])
    config = dict(arm_id=a.arm_id, regime=a.regime, encoder=a.encoder, fm=a.fm, repa_lambda=a.repa_lambda if proj is not None else 0.0,
                  repa_targets=trec, projector=dict(hidden=2048, layers=3, act="SiLU", seed=PROJ_SEED, discarded_at_save=True) if proj is not None else None,
                  init=init_rec, arch=U.E0_ARCH, params=info, pseudo=dict(npz=str(a.pseudo_npz), frac=a.pseudo_frac) if a.pseudo_npz else None,
                  recipe=dict(global_batch=a.batch_size, grad_accum=a.grad_accum, units_per_step=a.batch_size * a.grad_accum, lr=a.learning_rate,
                              betas=list(a.adam_betas), warmup=a.warmup_steps, enc_lr=a.enc_lr, enc_warmup=a.enc_warmup, steps=a.train_steps, save_every=a.save_every, ema=[a.ema_decay, a.ema_update_every],
                              clip=a.max_grad_norm, amp="fp16" if use_amp else None, seed=a.seed, train_fraction=a.train_fraction, split_seed=a.split_seed))
    if main0:
        cfg = dict(schema=FORMAT_TRAIN, run_dir=str(run_dir), created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, config=config, code_sha256=per_file,
                   code_sha256_combined=csha, environment=U.environment_record(), world_size=world, device=str(dev), resumed_step=step,
                   seed_used=seed_now, train_units=int(len(tr_pos)), heldout_units=int(len(va_pos)), sources=[dict(name=s.name, n=s.n) for s in src],
                   train_npz_sha256=sha_file(a.train_npz), render_npy_sha256=sha_file(a.render_npy), verdict_at_start=vrec)
        name = "run_config.json" if not (run_dir / "run_config.json").exists() else f"run_config_resume_{step:07d}.json"
        U.write_json_atomic(run_dir / name, cfg)
        P(f"arm {a.arm_id} regime {a.regime} encoder {a.encoder} fm {a.fm} world {world} step {step} seed {seed_now}; params {json.dumps(info, default=str)}")

    def save_latest(s):
        if not main0:
            return
        data = {"format_version": FORMAT_TRAIN, "step": s, "model": diff.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None, "projector": proj.state_dict() if proj is not None else None,
                "config": config, "code_sha256_combined": csha}
        U.save_torch_atomic(data, latest)

    def save_milestone(s):
        if not main0:
            return
        ms = s // a.save_every
        path = run_dir / f"bb_ms{ms}.pt"
        data = {"format_version": FORMAT_MS, "milestone": ms, "step": s, "model_unet": {k: v.detach().cpu() for k, v in unet.state_dict().items()},
                "ema_unet": {k: v.detach().cpu() for k, v in ema.ema_model.model.state_dict().items()}, "config": config,
                "code_sha256_combined": csha, "fm_free": True, "projector_in_file": False}
        U.save_torch_atomic(data, path)
        ck2 = torch.load(str(path), map_location="cpu", weights_only=False)
        assert not any(k in ck2 for k in ("projector", "opt", "ema")), list(ck2)
        net2, _ = BM.load_backbone_for_inference(path, U, "cpu")
        rec = dict(path=str(path), sha256=sha_file(path), step=s, milestone=ms, roundtrip_load_ok=True,
                   n_ema_keys=len(data["ema_unet"]), projector_in_file=False, fm_tensors_in_file=False)
        U.write_json_atomic(run_dir / f"bb_ms{ms}.verify.json", rec)
        log.log(event="milestone", **rec)
        del ck2, net2

    @torch.no_grad()
    def val_loss():
        if not main0 or len(va_pos) == 0:
            return None
        net.eval()
        with torch.random.fork_rng(devices=[dev] if on_cuda else []):
            torch.manual_seed(0)
            tot, n = 0.0, 0
            for b in range(0, len(va_pos), 100):
                items = [ds[int(p)] for p in va_pos[b:b + 100]]
                d = BD.collate(items)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    l = diff(d["coords"].to(dev), classes=d["imgs"].to(dev))
                tot += float(l) * len(items); n += len(items)
        net.train()
        return tot / n

    t_last, steps_here = time.time(), 0
    ddp_net.train()
    while step < a.train_steps:
        for g in opt.param_groups:                              # per-group linear warm-up, constant after
            g["lr"] = g["base_lr"] * min(1.0, (step + 1) / g["warmup"]) if g["warmup"] else g["base_lr"]
        tot_den, tot_rep, n_tgt = 0.0, 0.0, 0
        for j in range(a.grad_accum):
            d = next(it)
            coords, imgs = d["coords"].to(dev, non_blocking=True), d["imgs"].to(dev, non_blocking=True)
            tgt = d["tgt"].to(dev, non_blocking=True) if T is not None else None
            mask = d["mask"].to(dev) if T is not None else None
            ctx = ddp_net.no_sync() if (ddp and j < a.grad_accum - 1) else _null()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    den, rep = ddp_net(coords, imgs, tgt, mask)
                loss = den + (a.repa_lambda * rep if rep is not None else 0.0)
                scaler.scale(loss / a.grad_accum).backward()
            tot_den += float(den) / a.grad_accum
            if rep is not None:
                tot_rep += float(rep) / a.grad_accum; n_tgt += int(mask.sum())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model_params, a.max_grad_norm)
        gp = torch.nn.utils.clip_grad_norm_(list(proj.parameters()), a.max_grad_norm) if proj is not None else None
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1; steps_here += 1
        if main0:
            ema.update()
        if main0 and (step % a.log_every == 0 or step <= 3):
            now = time.time()
            rec = dict(event="train", step=step, loss_den=tot_den, loss_repa=tot_rep if proj is not None else None, n_repa_units_rank0=n_tgt,
                       grad_norm=float(gn), proj_grad_norm=float(gp) if gp is not None else None, lr=opt.param_groups[0]["lr"], lr_enc=opt.param_groups[1]["lr"],
                       scale=scaler.get_scale() if scaler.is_enabled() else None,
                       sec_per_step=(now - t_last) / (a.log_every if step % a.log_every == 0 else 1),
                       finite=bool(np.isfinite(tot_den) and np.isfinite(tot_rep)))
            if on_cuda:
                rec["cuda_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log.log(**rec)
            print(f"step {step} den {tot_den:.4f} repa {tot_rep:+.4f} gn {float(gn):.2f} {rec['sec_per_step']:.3f}s/step", flush=True)
            t_last = now
        if a.val_every and step % a.val_every == 0:
            v = val_loss()
            if main0:
                log.log(event="val", step=step, val_loss=v)
            if ddp:
                dist.barrier()
        if step % a.save_every == 0:
            save_milestone(step)
        if step % a.resume_every == 0 or step == a.train_steps:
            save_latest(step)
        if ddp and (step % a.save_every == 0 or step % a.resume_every == 0):
            dist.barrier()
        if a.fm and step % a.verdict_every == 0 and step < a.train_steps:
            vrec = read_verdict(a.verdict_file, a.verdict_key) if main0 else None
            fail = bcast_flag(main0 and vrec["status"] == "FAIL")
            if main0:
                log.log(event="gate_check", step=step, verdict=vrec)
            if fail:
                if main0:
                    write_gate_fail(run_dir, a, step, vrec)
                P(f"GATE_FAIL at step {step}: {vrec}"); finish(0)
        if a.stop_after is not None and steps_here >= a.stop_after and step < a.train_steps:
            save_latest(step)
            P(f"stop-after {a.stop_after} reached at step {step} (smoke)"); finish(0)
    if main0:
        res = dict(status="complete", final_step=step, completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   arm_id=a.arm_id, milestones=sorted(int(p.stem[5:]) for p in run_dir.glob("bb_ms*.pt")), peak_rss_mb=U.peak_rss_mb(),
                   cuda_max_alloc_gb=torch.cuda.max_memory_allocated() / 1e9 if on_cuda else None, lane=a.lane, slurm=os.environ.get("SLURM_JOB_ID"))
        U.write_json_atomic(run_dir / "training_complete.json", res)
        print(json.dumps(res, default=str), flush=True)
    finish(0)


class TargetDS(torch.utils.data.Dataset):
    """Module-level (picklable for spawn workers, Windows/VPS lane): adds the REPA target tokens (float16) to every item."""

    def __init__(self, ds, targets_path=None):
        self.ds, self.targets_path, self._T = ds, targets_path, None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, key):
        it = self.ds[key]
        if self.targets_path is not None:
            if self._T is None:
                self._T = np.load(self.targets_path, mmap_mode="r")
            r = it["target_row"]
            it["tgt"] = torch.from_numpy(np.asarray(self._T[r], np.float16)) if r >= 0 else torch.zeros(256, self._T.shape[2], dtype=torch.float16)
        return it


def collate_targets(batch, with_targets=False):
    out = BD.collate(batch)
    if with_targets:
        out["tgt"] = torch.stack([b["tgt"] for b in batch]); out["mask"] = (out["target_row"] >= 0).float()
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *x):
        return False


if __name__ == "__main__":
    main()
