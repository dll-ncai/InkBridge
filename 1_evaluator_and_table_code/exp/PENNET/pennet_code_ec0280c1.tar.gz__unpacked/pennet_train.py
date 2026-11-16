"""PEN-Net training on OUHD-L v3 ligature units (E0 arrays), single GPU, auto-resume.

Loss (paper sec. 4.3): L = 0.5 * L1(x, y) + 1 * weighted CE(pen state; weights down 1, up 5, eos 1) + SDTW / 6000.
  * L1 term   : sum over valid rows (state != eos) of |dx| + |dy|, divided by the number of valid rows
                (released train.py: dx_loss + dy_loss, masked, .sum() / valid_mask.sum()).
  * CE term   : nn.CrossEntropyLoss(weight=[1,5,1], reduction='none') over all B*T padded rows, then .mean()
                (released code form; the released weights are [1,5,10], the paper says [1,5,1] -> paper used).
  * SDTW term : soft-DTW (gamma 0.1, squared Euclidean, released code) between label[:len_gt, :2] (incl. EOS row) and the
                teacher-forced prediction pred[:len_pred, :2], len_pred = first row whose arg-max state is EOS, + 1
                (else T); per-sample, averaged over the batch (released SDTW_BS=False path).
Optimiser: Adam lr 1e-3 constant (paper; the released MultiStepLR object is never attached/stepped), grad L2 clip 0.2
(released config).  Batch 512 (paper).  Steps: 100,000 (E0 budget rule; the paper's own schedule is 200k/500k steps).

Checkpointing: <run>/ckpt_latest.pt every --ckpt-every steps (atomic replace) with model, optimiser, step, data-order
position and RNG states; milestone weights <run>/milestones/ms_<k>.pt every --milestone-every steps; SIGTERM/SIGUSR1/
SIGINT -> save ckpt_latest at the end of the current step and exit 0 (Slurm requeue restarts and resumes).
Data order: epoch e uses np.random.default_rng(seed + e).permutation(N), drop_last (released DataLoader drop_last=True),
so a resumed run sees the same batches as an uninterrupted one.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pennet_data import build_label_tensor, image_tensor, load_split
from pennet_model import PENNet, SOS
from pennet_softdtw import soft_dtw_batch

STOP = {"flag": False, "sig": None}


def _handler(sig, frame):
    STOP["flag"] = True
    STOP["sig"] = sig


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def loss_fn(preds, labels, lens, ce_w, lam, gamma):
    xy_gt, st_gt = labels[:, :, :2], torch.argmax(labels[:, :, 2:], dim=-1)
    valid = (st_gt != 2).float()
    l1 = ((preds[:, :, 0] - labels[:, :, 0]).abs() + (preds[:, :, 1] - labels[:, :, 1]).abs()) * valid
    l1 = l1.sum() / valid.sum()
    B, T, _ = preds.shape
    ce = F.cross_entropy(preds[:, :, 2:].reshape(B * T, 3), st_gt.reshape(-1), weight=ce_w, reduction="none").mean()
    with torch.no_grad():
        pst = torch.argmax(preds[:, :, 2:], dim=-1)
        is_eos = pst == 2
        first = torch.where(is_eos.any(1), is_eos.float().argmax(1) + 1, torch.full_like(lens, T))
    sd = soft_dtw_batch(xy_gt, preds[:, :, :2], lens, first, gamma) / 6000.0
    sd = sd.mean()
    loss = lam[0] * l1 + lam[1] * ce + lam[2] * sd
    return loss, dict(l1=l1.item(), ce=ce.item(), sdtw6000=sd.item(), loss=loss.item(),
                      pred_len_mean=first.float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--condition", default="rendered")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ce-weights", type=float, nargs=3, default=[1.0, 5.0, 1.0])
    ap.add_argument("--lambdas", type=float, nargs=3, default=[0.5, 1.0, 1.0])   # SDTW is divided by 6000 in loss_fn
    ap.add_argument("--sdtw-gamma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--milestone-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-steps-this-process", type=int, default=0, help="stop (after saving) once this many steps ran")
    ap.add_argument("--max-units", type=int, default=0, help="smoke only: first N train units")
    a = ap.parse_args()

    for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(s, _handler)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    (run / "milestones").mkdir(parents=True, exist_ok=True)

    d = load_split(a.train_npz, a.condition)
    imgs = image_tensor(d["cond"])
    labels, lens = build_label_tensor(d)
    if a.max_units:
        imgs, labels, lens = imgs[:a.max_units], labels[:a.max_units], lens[:a.max_units]
    N = len(lens)
    imgs, labels, lens = imgs.to(dev), labels.to(dev), lens.to(dev)
    ce_w = torch.tensor(a.ce_weights, device=dev)
    sos = torch.tensor(SOS, device=dev).view(1, 1, 5)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    net = PENNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    step, epoch, pos = 0, 0, 0
    ck = run / "ckpt_latest.pt"
    resumed_from = None
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, epoch, pos = st["step"], st["epoch"], st["pos"]
        torch.set_rng_state(st["rng_torch"].cpu())
        if dev.type == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        resumed_from = step
    cfg = dict(arguments=vars(a), n_train_units=N, device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               capability=list(torch.cuda.get_device_capability(0)) if dev.type == "cuda" else None,
               torch=torch.__version__, cudnn=torch.backends.cudnn.version() if dev.type == "cuda" else None,
               n_params=sum(p.numel() for p in net.parameters()), resumed_from_step=resumed_from,
               gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "")) if dev.type == "cuda" else None,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
               cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
               slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_restart_count=os.environ.get("SLURM_RESTART_COUNT"),
               host=os.uname().nodename, start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    name = "run_config.json" if resumed_from is None else f"run_config_resume_{resumed_from}.json"
    json.dump(cfg, open(run / name, "w"), indent=1)
    print(json.dumps(cfg), flush=True)

    def save(tag):
        atomic_save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, epoch=epoch, pos=pos,
                         rng_torch=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                         args=vars(a)), ck)
        with open(run / "ckpt_events.txt", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\tstep={step}\t{tag}\tjob={os.environ.get('SLURM_JOB_ID')}\n")

    perm = np.random.default_rng(a.seed + epoch).permutation(N)
    nb = N // a.batch
    logf = open(run / "train_log.jsonl", "a")
    ran, t_last, steps_since = 0, time.time(), 0
    net.train()
    while step < a.steps:
        if pos >= nb:
            epoch += 1
            pos = 0
            perm = np.random.default_rng(a.seed + epoch).permutation(N)
        idx = torch.from_numpy(perm[pos * a.batch:(pos + 1) * a.batch]).to(dev)
        lb, ln = labels[idx], lens[idx]
        T = int(ln.max())
        lb = lb[:, :T]
        seq_in = torch.cat([sos.expand(len(idx), 1, 5), lb], dim=1)[:, :-1]
        preds = net(imgs[idx], seq_in)
        loss, parts = loss_fn(preds, lb, ln, ce_w, a.lambdas, a.sdtw_gamma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip) if a.clip > 0 else torch.tensor(0.0)
        opt.step()
        step += 1
        pos += 1
        ran += 1
        steps_since += 1
        if not np.isfinite(parts["loss"]):
            print(f"non-finite loss at step {step}: {parts}", flush=True)
            sys.exit(7)
        if step % a.log_every == 0 or step == 1:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            rec = dict(step=step, epoch=epoch, T=T, grad_norm=float(gn), s_per_step=(now - t_last) / max(steps_since, 1),
                       **parts)
            t_last, steps_since = now, 0
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
        if step % a.milestone_every == 0:
            k = step // a.milestone_every
            atomic_save(dict(model=net.state_dict(), step=step, args=vars(a)), run / "milestones" / f"ms_{k:03d}.pt")
        if step % a.ckpt_every == 0:
            save("periodic")
        if STOP["flag"]:
            save(f"signal_{STOP['sig']}")
            print(f"stopped on signal {STOP['sig']} at step {step}", flush=True)
            sys.exit(0)
        if a.max_steps_this_process and ran >= a.max_steps_this_process:
            save("max_steps_this_process")
            print(f"max-steps-this-process reached at step {step}", flush=True)
            return
    save("final")
    json.dump(dict(step=step, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   slurm_job_id=os.environ.get("SLURM_JOB_ID")), open(run / "training_complete.json", "w"))
    print("training complete", step, flush=True)


if __name__ == "__main__":
    main()
