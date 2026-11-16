"""E0: train the Nagamatsu image-conditioned diffusion model on OUHD-L ligature units v3.0.

Fork of src/standard_diffusion_train.py (repo commit 7c5cf623).  Model, diffusion, Trainer1D and
collate_fn are imported unchanged from the repo.  Changes, all in this file:
  1. dataset: OuhdlE0Dataset(--e0-train-npz, --condition) instead of PotDataset (CASIA .pot/LMDB);
  2. --run-dir: fixed run directory with automatic resume from its latest checkpoint
     (the original always creates a new timestamped directory); a resumed process writes
     run_config_resume_<step>.json and keeps the original run_config.json;
  3. on resume the RNG seed is (random_seed + resumed step) so a restart does not replay batches;
  4. --classes is unused (kept only because Trainer1D logs cls_num).
Everything else (arguments, defaults, optimiser, EMA, checkpoint format) is the repo's.
M5 change (ported from exp/M5 m5_train.py, 2026-10-20):
  5. --init-from CKPT: when the run dir has no checkpoint yet, load model + EMA weights of CKPT; optimiser, AMP
     scaler and step start fresh (step 0); writes init_from.json.  Auto-resume takes over once checkpoints exist.
Frame / replay / gate changes (2026-10-28, C1/C3 of uhwr_method_candidates_2026-10-27; all in this file plus
e0_frames.py, e0_replay.py, e0_dataset.py):
  6. --frame {identity,mirror_x,vt} [--frame-scripts ouhdl]: canonical frame applied by the dataset to units whose
     script tag is listed (the OUHD-L npz has no `scripts` key and is tagged 'ouhdl'); LTR replay units stay identity.
  7. --replay-npz m5_luo_train.npz [--replay-scripts ...] --urdu-share 0.25 [--urdu-share-end 0.5
     --curriculum-start-step 7000]: two-source batches (e0_replay.ReplayBatchSampler) replace Trainer1D's shuffled
     loader; Trainer1D is otherwise untouched (its random_split still defines the Urdu train/val-loss split, and
     the Urdu sampler draws only from that train part).  The fine-tune recipe (batch 128 x accum 4, lr 8e-5,
     10k steps, ckpt every 1k, val-loss every 1k) is unchanged.
  8. --gate-step 4000 --gate-gold-npz e0_gold.npz --gate-unipen-npz unipen_de/e0_test.npz: ReplayTrainer1D.train is
     a copy of Trainer1D.train with one hook: after the checkpoint at gate-step is saved, e0_replay.gate_eval samples
     gold 354 (frame + inverse) and 200 UNIPEN-de units (seed 0, DDIM-50, EMA), scores with uhwr_eval 0.3.1 and
     writes eval/gate_4k.json; on failure the job writes GATE_FAIL (json) in the run dir and exits 0.
     --gate-only runs the gate on an existing checkpoint (--gate-checkpoint or run dir + --gate-milestone) and exits.
"""

import argparse
import json
import os
import platform
import secrets
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "standard_diffusion" / "runs"


def parse_int_tuple(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--e0-train-npz", type=Path, required=True)
    parser.add_argument("--condition", default="rendered")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-milestone", type=int)
    parser.add_argument("--init-from", type=Path, help="M5: checkpoint whose model+EMA weights initialise a fresh run")
    # frame / replay / gate (2026-10-28)
    parser.add_argument("--frame", default="identity", choices=("identity", "mirror_x", "vt"))
    parser.add_argument("--frame-scripts", nargs="*", default=["ouhdl"], help="script tags the frame applies to")
    parser.add_argument("--replay-npz", type=Path, help="multi-script npz with a `scripts` key (m5_luo_train.npz)")
    parser.add_argument("--replay-scripts", nargs="*", help="replay script tags (default: all tags in --replay-npz)")
    parser.add_argument("--urdu-share", type=float, default=0.25, help="q_u: fixed share of --e0-train-npz units per batch")
    parser.add_argument("--urdu-share-end", type=float, help="curriculum: q_u rises linearly to this value by --train-steps")
    parser.add_argument("--curriculum-start-step", type=int, help="curriculum start (e.g. 7000 for 0.2->0.5 over the last 3k)")
    parser.add_argument("--gate-step", type=int, help="run the K3 milestone gate after the checkpoint at this step")
    parser.add_argument("--gate-gold-npz", type=Path); parser.add_argument("--gate-unipen-npz", type=Path)
    parser.add_argument("--gate-batch-size", type=int, default=256); parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument("--gate-max-units", type=int, help="CPU smoke only: cap both gate sets")
    parser.add_argument("--gate-label", help="label for the Route A preds files (default <run name>_ms<k>)")
    parser.add_argument("--gate-only", action="store_true", help="run the gate on an existing checkpoint and exit")
    parser.add_argument("--gate-checkpoint", type=Path); parser.add_argument("--gate-milestone", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--cv2-threads", type=int, default=0)
    parser.add_argument(
        "--content",
        choices=("kanji", "english"),
        default="kanji",
        help=(
            "Dataset subset. 'kanji' is the cache-compatible identifier for "
            "CASIA Chinese characters."
        ),
    )
    parser.add_argument("--classes", type=int, default=3000)
    parser.add_argument("--class-selection-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=320)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-thickness-min", type=int, default=1)
    parser.add_argument("--train-thickness-max", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=parse_int_tuple, default=(1, 2, 4, 8))
    parser.add_argument(
        "--encoder-hidden-dims",
        type=parse_int_tuple,
        default=(64, 128, 256, 512),
    )
    parser.add_argument("--attn-dim-head", type=int, default=32)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--cross-attn-dim-head", type=int, default=64)
    parser.add_argument("--cross-attn-heads", type=int, default=8)
    parser.add_argument("--linear-attn-dim-head", type=int, default=32)
    parser.add_argument("--linear-attn-heads", type=int, default=4)
    parser.add_argument("--attention-max-positions", type=int, default=5000)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--linear-beta-start", type=float, default=0.0001)
    parser.add_argument("--linear-beta-end", type=float, default=0.02)
    parser.add_argument("--cosine-schedule-s", type=float, default=0.008)
    parser.add_argument(
        "--objective",
        choices=("pred_noise", "pred_x0", "pred_v"),
        default="pred_noise",
    )
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--train-steps", type=int, default=500_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--validation-every", type=int, default=1_000)
    parser.add_argument("--gradient-accumulate-every", type=int, default=2)
    parser.add_argument("--ema-update-every", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.999)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sample-count", type=int, default=25)
    parser.add_argument("--sample-grid-rows", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mixed-precision-type",
        choices=("fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--split-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", default="diffusion")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="offline",
    )
    parser.add_argument("--wandb-log-freq", type=int, default=1_000)
    parser.add_argument("--notes", default="3000-class conditional diffusion training")
    return parser.parse_args()


def _make_trainer_class():
    """Subclass of the repo's Trainer1D whose train() is a copy of the original loop with the gate hook (change 8).
    Defined lazily so e0_train.py can be imported without torch/wandb on PYTHONPATH."""
    import torch
    import wandb
    from tqdm.auto import tqdm
    from diffusion_models.ddpm_conditional import Trainer1D
    from diffusion_models.conditional_1d.runtime_utils import write_json_atomic, output_timestamp

    class ReplayTrainer1D(Trainer1D):
        gate = None            # dict from main(); None -> behave exactly like Trainer1D.train
        gate_failed = False
        replay_sampler = None

        def _run_gate(self):
            import e0_replay as R
            g = self.gate
            self.ema.ema_model.eval()
            res = R.gate_eval(self.ema.ema_model, self.device, gold_npz=g["gold_npz"], unipen_npz=g["unipen_npz"], frame=g["frame"],
                              out_dir=g["out_dir"], label=g["label"], step=self.step, seed=g["seed"], batch_size=g["batch_size"],
                              sampling_timesteps=g["sampling_timesteps"], max_units=g["max_units"],
                              checkpoint=str(__import__("pathlib").Path(g["run_directory"]) / "checkpoints" / f"checkpoint_{self.step // g['save_every']:06d}.pt"),
                              extra_meta={"in_job_gate": True, "run_dir": g["run_directory"]})
            print(f"GATE step {self.step}: {json.dumps({k: res[k] for k in ('gold_aiou', 'gold_tau', 'unipen_direction', 'pass', 'checks')})}")
            if not res["pass"]:
                write_json_atomic(__import__("pathlib").Path(g["run_directory"]) / "GATE_FAIL",
                                  {"status": "gate_failed", "step": self.step, "gate": res, "at": output_timestamp()})
                self.gate_failed = True
            return res["pass"]

        def train(self):
            accelerator = self.accelerator
            wandb.watch(self.model, log="all", log_freq=self.wandb_log_freq)
            with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
                while self.step < self.train_num_steps:
                    self.model.train()
                    total_loss = 0.0
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.train_dl)
                        coords = data["coords"]
                        imgs = data["imgs"]
                        with self.accelerator.autocast():
                            loss = self.model(coords, classes=imgs)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        self.accelerator.backward(loss)
                    wandb.log({"loss/train": total_loss, "step": self.step})
                    pbar.set_description(f"loss: {total_loss:.4f}")
                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    accelerator.wait_for_everyone()
                    if self.step % self.validation_every == 0:
                        validation_loss = self._calculate_validation_loss()
                        wandb.log({"loss/val": validation_loss, "step": self.step})
                    self.step += 1
                    stop = False
                    if accelerator.is_main_process:
                        self.ema.update()
                        if self.step != 0 and self.step % self.save_and_sample_every == 0:
                            self._save_training_snapshot(imgs, coords)
                        # ---- change 8: milestone gate (after the checkpoint at gate step is on disk)
                        if self.gate is not None and self.step == int(self.gate["step"]):
                            stop = not self._run_gate()
                    if accelerator.num_processes > 1:
                        flag = torch.tensor([int(stop)], device=self.device)
                        flag = accelerator.reduce(flag, reduction="sum")
                        stop = bool(flag.item())
                    pbar.update(1)
                    if stop:
                        accelerator.print(f"gate failed at step {self.step}: stopping (checkpoint saved)")
                        break
            accelerator.print("training complete" if not self.gate_failed else "training stopped by gate")
            wandb.finish()

    return ReplayTrainer1D


def gate_only(args):
    """--gate-only: run the K3 gate on an existing checkpoint (e0_sample.py model construction) and exit."""
    import torch
    from ema_pytorch import EMA
    from diffusion_models.ddpm_conditional import GaussianDiffusion1D, Unet1D
    from diffusion_models.conditional_1d.runtime_utils import checkpoint_path
    from utils.casia_grid import TRAJECTORY_CHANNELS
    import e0_replay as R
    run_dir = args.run_dir.expanduser().resolve()
    cfg = json.load(open(run_dir / "run_config.json"))["arguments"]
    model = Unet1D(dim=int(cfg["model_dim"]), dim_mults=tuple(cfg["dim_mults"]), channels=TRAJECTORY_CHANNELS,
                   attn_dim_head=int(cfg["attn_dim_head"]), attn_heads=int(cfg["attn_heads"]),
                   cross_attn_dim_head=int(cfg["cross_attn_dim_head"]), cross_attn_heads=int(cfg["cross_attn_heads"]),
                   linear_attn_dim_head=int(cfg["linear_attn_dim_head"]), linear_attn_heads=int(cfg["linear_attn_heads"]),
                   attention_max_positions=int(cfg["attention_max_positions"]), condition_image_size=int(cfg["image_size"]),
                   encoder_hidden_dims=tuple(cfg["encoder_hidden_dims"]))
    diff = GaussianDiffusion1D(model, seq_length=int(cfg["sequence_length"]), timesteps=int(cfg["timesteps"]), sampling_timesteps=50,
                               objective=cfg["objective"], beta_schedule=cfg["beta_schedule"], ddim_sampling_eta=0.0,
                               linear_beta_start=float(cfg["linear_beta_start"]), linear_beta_end=float(cfg["linear_beta_end"]),
                               cosine_schedule_s=float(cfg["cosine_schedule_s"]))
    ckp = args.gate_checkpoint if args.gate_checkpoint is not None else checkpoint_path(run_dir, int(args.gate_milestone))
    ck = torch.load(str(ckp), map_location="cpu")
    ema = EMA(diff, beta=float(cfg["ema_decay"]), update_every=int(cfg["ema_update_every"]))
    ema.load_state_dict(ck["ema"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ema.ema_model.to(dev).eval()
    step = int(ck.get("step", -1)); del ck
    label = args.gate_label or f"{run_dir.name}_ms{step // max(args.save_every, 1)}"
    res = R.gate_eval(model, dev, gold_npz=str(args.gate_gold_npz), unipen_npz=str(args.gate_unipen_npz), frame=args.frame,
                      out_dir=str(run_dir / "eval"), label=label, step=step, seed=args.gate_seed, batch_size=args.gate_batch_size,
                      sampling_timesteps=50, max_units=args.gate_max_units, checkpoint=str(ckp),
                      extra_meta={"in_job_gate": False, "gate_only": True, "run_dir": str(run_dir)})
    print(json.dumps({k: res[k] for k in ("step", "gold_aiou", "gold_tau", "unipen_direction", "pass", "checks", "seconds", "files")}))
    return res


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import cv2
    import random
    import torch
    from accelerate import PartialState
    from accelerate.utils import broadcast_object_list

    from diffusion_models.ddpm_conditional import GaussianDiffusion1D, Trainer1D, Unet1D
    from diffusion_models.conditional_1d.runtime_utils import (
        create_unique_output_directory,
        output_timestamp,
        write_json_atomic,
    )
    from diffusion_models.conditional_1d.runtime_utils import latest_checkpoint_milestone
    from utils.casia_grid import TRAJECTORY_CHANNELS
    from e0_dataset import OuhdlE0Dataset
    import e0_frames as F
    import e0_replay as R

    if args.gate_only:
        return gate_only(args)

    distributed_state = PartialState()
    seed_holder = [args.random_seed]
    if distributed_state.is_main_process and seed_holder[0] is None:
        seed_holder[0] = secrets.randbits(32)
    broadcast_object_list(seed_holder)
    args.random_seed = seed_holder[0]

    # E0 change 2: fixed run directory + automatic resume from its latest checkpoint
    run_directory = args.run_dir.expanduser().resolve()
    run_name = args.run_name or run_directory.name
    resume_milestone = None
    if args.resume_milestone is None and (run_directory / "checkpoints").is_dir():
        resume_milestone = latest_checkpoint_milestone(run_directory)
    elif args.resume_milestone is not None:
        resume_milestone = args.resume_milestone
    resumed_step = 0 if resume_milestone is None else resume_milestone * args.save_every
    # E0 change 3: distinct RNG stream after a restart
    seed_now = (args.random_seed + resumed_step) % (2 ** 32)
    cv2.setNumThreads(args.cv2_threads)
    random.seed(seed_now)
    np.random.seed(seed_now)
    torch.manual_seed(seed_now)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_now)

    created_at = output_timestamp()
    run_directory.mkdir(parents=True, exist_ok=True)
    if distributed_state.is_main_process:
        cfg_name = ("run_config.json" if not (run_directory / "run_config.json").exists()
                    else f"run_config_resume_{resumed_step:07d}.json")
        write_json_atomic(
            run_directory / cfg_name,
            {
                "schema_version": 2,
                "kind": "training",
                "run_id": run_name,
                "created_at": created_at,
                "output_directory": run_directory,
                "arguments": vars(args),
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "distributed": {
                    "num_processes": distributed_state.num_processes,
                },
                "e0": {
                    "resume_milestone": resume_milestone,
                    "seed_used": seed_now,
                    "train_npz": str(args.e0_train_npz),
                    "condition": args.condition,
                    "init_from": None if args.init_from is None else str(args.init_from),
                    "frame": args.frame, "frame_scripts": args.frame_scripts,
                    "replay_npz": None if args.replay_npz is None else str(args.replay_npz),
                    "urdu_share": args.urdu_share, "urdu_share_end": args.urdu_share_end,
                    "curriculum_start_step": args.curriculum_start_step, "gate_step": args.gate_step,
                },
            },
        )
        print(f"Training output: {run_directory}")
    distributed_state.wait_for_everyone()

    # E0 change 1: OUHD-L v3 ligature units (fixed rendering, no thickness augmentation;
    # --train-thickness-* and --data-root are ignored)
    if args.image_size != 64:
        raise ValueError("E0 arrays are 64x64")
    if (run_directory / "GATE_FAIL").exists():
        print("GATE_FAIL present in run dir: not continuing training (exit 0)")
        return
    train_dataset = OuhdlE0Dataset(args.e0_train_npz, condition=args.condition, frame=args.frame,
                                   frame_scripts=set(args.frame_scripts), script_tag=R.URDU_TAG)

    model = Unet1D(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=TRAJECTORY_CHANNELS,
        attn_dim_head=args.attn_dim_head,
        attn_heads=args.attn_heads,
        cross_attn_dim_head=args.cross_attn_dim_head,
        cross_attn_heads=args.cross_attn_heads,
        linear_attn_dim_head=args.linear_attn_dim_head,
        linear_attn_heads=args.linear_attn_heads,
        attention_max_positions=args.attention_max_positions,
        condition_image_size=args.image_size,
        encoder_hidden_dims=args.encoder_hidden_dims,
    )
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=args.sequence_length,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
        ddim_sampling_eta=args.ddim_eta,
        linear_beta_start=args.linear_beta_start,
        linear_beta_end=args.linear_beta_end,
        cosine_schedule_s=args.cosine_schedule_s,
    )

    ReplayTrainer1D = _make_trainer_class()
    trainer = ReplayTrainer1D(
        diffusion,
        dataset=train_dataset,
        output_directory=run_directory,
        notes=args.notes,
        run_name=run_name,
        cls_num=args.classes,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_update_every=args.ema_update_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        split_batches=args.split_batches,
        max_grad_norm=args.max_grad_norm,
        save_and_sample_every=args.save_every,
        num_samples=args.sample_count,
        data_loader_workers=args.num_workers,
        validation_every=args.validation_every,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_log_freq=args.wandb_log_freq,
        sample_grid_rows=args.sample_grid_rows,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        pin_memory=args.pin_memory,
        resume_directory=args.resume_from or run_directory,
    )
    if resume_milestone is not None:
        trainer.load(resume_milestone)
        print(f"Resumed from milestone {resume_milestone} (step {trainer.step}); seed {seed_now}")
    elif args.init_from is not None:
        # M5 change 5: weights-only initialisation (fresh optimiser / scaler / step)
        import hashlib
        ck = torch.load(str(args.init_from), map_location=trainer.accelerator.device)
        trainer.accelerator.unwrap_model(trainer.model).load_state_dict(ck["model"])
        if hasattr(trainer, "ema"):
            trainer.ema.load_state_dict(ck["ema"])
        trainer.step = 0
        if distributed_state.is_main_process:
            write_json_atomic(run_directory / "init_from.json",
                              {"init_from": str(args.init_from), "sha256": R.sha256_file(args.init_from),
                               "source_step": int(ck.get("step", -1)), "loaded": ["model", "ema"],
                               "fresh": ["opt", "scaler", "step"], "at": output_timestamp()})
        print(f"Initialised model+EMA from {args.init_from} (source step {ck.get('step')}); step 0; seed {seed_now}")
        del ck

    # change 7: two-source replay loader (Urdu train split of --e0-train-npz + LTR shards of --replay-npz)
    if args.replay_npz is not None:
        from diffusion_models.conditional_1d.runtime_utils import cycle
        train_size = int(args.train_fraction * len(train_dataset))
        split = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size],
                                              generator=torch.Generator().manual_seed(args.split_seed))
        urdu_train_idx = np.sort(np.asarray(split[0].indices))        # identical to Trainer1D's own split
        urdu_ds, replay_ds, by_script, urdu_train_idx = R.build_replay_sources(
            args.e0_train_npz, args.replay_npz, args.condition, args.frame, args.frame_scripts, args.replay_scripts, urdu_train_idx)
        dl, sampler = R.make_replay_loader(
            urdu_ds, replay_ds, by_script, urdu_train_idx, batch_size=args.batch_size, seq_length=args.sequence_length,
            q_u=args.urdu_share, q_u_end=args.urdu_share_end, curriculum_start_step=args.curriculum_start_step,
            train_steps=args.train_steps, gradient_accumulate_every=args.gradient_accumulate_every, seed=seed_now,
            start_step=trainer.step, num_workers=args.num_workers, pin_memory=args.pin_memory,
            num_processes=distributed_state.num_processes)
        sampler.batch_size = sampler.B           # for accelerate's BatchSamplerShard bookkeeping
        trainer.train_dl = cycle(trainer.accelerator.prepare(dl))
        trainer.replay_sampler = sampler
        if distributed_state.is_main_process:
            write_json_atomic(run_directory / "replay_config.json", {
                "sampler": sampler.describe(), "frame": F.describe(args.frame), "frame_scripts": args.frame_scripts,
                "urdu_npz": str(args.e0_train_npz), "urdu_npz_sha256": R.sha256_file(args.e0_train_npz),
                "replay_npz": str(args.replay_npz), "replay_npz_sha256": R.sha256_file(args.replay_npz),
                "urdu_units_total": len(urdu_ds), "urdu_units_train_split": int(len(urdu_train_idx)),
                "replay_units_by_script": {k: int(len(v)) for k, v in by_script.items()},
                "note": "Trainer1D keeps its random_split for the val loss; its train DataLoader is replaced by this sampler",
                "at": output_timestamp()})
        print(f"Replay loader: {sampler.describe()}")

    # change 8: gate configuration
    if args.gate_step is not None:
        if args.gate_gold_npz is None or args.gate_unipen_npz is None:
            raise ValueError("--gate-step needs --gate-gold-npz and --gate-unipen-npz")
        if not diffusion.is_ddim_sampling or int(args.sampling_timesteps) != 50:
            raise ValueError("the gate protocol is DDIM-50: pass --sampling-timesteps 50")
        trainer.gate = dict(step=args.gate_step, gold_npz=str(args.gate_gold_npz), unipen_npz=str(args.gate_unipen_npz),
                            batch_size=args.gate_batch_size, seed=args.gate_seed, max_units=args.gate_max_units,
                            frame=args.frame, label=args.gate_label or f"{run_name}_ms{args.gate_step // args.save_every}",
                            out_dir=str(run_directory / "eval"), run_directory=str(run_directory), save_every=args.save_every,
                            sampling_timesteps=int(args.sampling_timesteps))

    trainer.train()
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        if getattr(trainer, "replay_sampler", None) is not None:
            write_json_atomic(run_directory / "replay_stats.json", {"batches": trainer.replay_sampler.batches_yielded,
                              "final_step": trainer.step, **trainer.replay_sampler.log, "at": output_timestamp()})
        if getattr(trainer, "gate_failed", False):
            print("GATE_FAIL written; exiting 0 without training_complete.json")
            return
        write_json_atomic(
            run_directory / "training_complete.json",
            {
                "status": "complete",
                "completed_at": output_timestamp(),
                "final_step": trainer.step,
            },
        )


if __name__ == "__main__":
    main()
