"""M5 (X6): train the Nagamatsu image-conditioned diffusion model on the multi-script set m5_multiscript_v1
(joint / leave-Urdu-out arms) or fine-tune it on OUHD-L subsets.  Fork of the E0 trainer e0_train.py
(e0/code_4cd40f57, itself a fork of src/standard_diffusion_train.py at repo commit 7c5cf623).
Only change against e0_train.py (M5 change 5):
  5. --init-from CKPT: when the run dir has no checkpoint yet, load the model ("model") and EMA ("ema")
     weights of CKPT before training; the optimiser, AMP scaler and step counter start fresh (step 0).
     Once the run has its own checkpoints, the normal E0 auto-resume takes over and --init-from is ignored.
     Writes init_from.json (path, sha256, source step) in the run dir.
--- original E0 docstring follows ---
E0: train the Nagamatsu image-conditioned diffusion model on OUHD-L ligature units v3.0.

Fork of src/standard_diffusion_train.py (repo commit 7c5cf623).  Model, diffusion, Trainer1D and
collate_fn are imported unchanged from the repo.  Changes, all in this file:
  1. dataset: OuhdlE0Dataset(--e0-train-npz, --condition) instead of PotDataset (CASIA .pot/LMDB);
  2. --run-dir: fixed run directory with automatic resume from its latest checkpoint
     (the original always creates a new timestamped directory); a resumed process writes
     run_config_resume_<step>.json and keeps the original run_config.json;
  3. on resume the RNG seed is (random_seed + resumed step) so a restart does not replay batches;
  4. --classes is unused (kept only because Trainer1D logs cls_num).
Everything else (arguments, defaults, optimiser, EMA, checkpoint format) is the repo's.
"""

import argparse
import os
import platform
import secrets
from pathlib import Path


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


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import cv2
    import numpy as np
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
                },
            },
        )
        print(f"Training output: {run_directory}")
    distributed_state.wait_for_everyone()

    # E0 change 1: OUHD-L v3 ligature units (fixed rendering, no thickness augmentation;
    # --train-thickness-* and --data-root are ignored)
    if args.image_size != 64:
        raise ValueError("E0 arrays are 64x64")
    train_dataset = OuhdlE0Dataset(args.e0_train_npz, condition=args.condition)

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

    trainer = Trainer1D(
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
            h = hashlib.sha256()
            with open(args.init_from, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            write_json_atomic(run_directory / "init_from.json",
                              {"init_from": str(args.init_from), "sha256": h.hexdigest(),
                               "source_step": int(ck.get("step", -1)), "loaded": ["model", "ema"],
                               "fresh": ["opt", "scaler", "step"], "at": output_timestamp()})
        print(f"Initialised model+EMA from {args.init_from} (source step {ck.get('step')}); step 0; seed {seed_now}")
        del ck
    trainer.train()
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
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
