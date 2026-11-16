"""RQF A2.3 one-model arms: VPS (Windows RTX 4090 runner) race copy of one chain, end to end, same code and recipe as the CLUSTER-A copy:
  smr_luo : frame_code_a44e78b0 e0_train.py (init LUO ms6, identity, --replay-npz m5_luo_train.npz --urdu-share 0.5, 128x4, lr 8e-5, 10k, ckpt 1k)
            -> c1_vps_post.py --kind frame (val1000 E0 rule) -> c1_sample.py 7 sets x seeds 0-2 (identity)
  smr_bb  : fmdx_backbone_v1 bb_train.py --regime ft (A4.3(e) recipe; 32x16 on <40 GB) + --pseudo-npz m5_luo_train --pseudo-render-npy luo_train_render224
            --pseudo-frac 0.5 -> bb_sample.py val1000 per milestone -> e0_score -> m5_select -> bb_sample 7 sets x seeds 0-2
  sma_luo / sma_bb : sm_v1 sma_train.py (D2 LoRA r16, n_experts 1, frozen backbone, 64x4, lr 1e-4, 10k) -> sma_sample.py val1000 (route all) ->
            e0_score -> m5_select -> sma_sample.py 7 sets x seeds 0-2 with the router manifests (p_rtl >= 0.5)
Run dir ROOT/runs/RQF/SM/<TAG>: STARTED.json (race marker read by the CLUSTER-A watcher), job_history.txt, eval/, preds/ (smr_luo) or ft/, SAMPLING_DONE."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(r"$VPS_ROOT\uhwr"); STG = ROOT / "stage"; PY = r"C:\ProgramData\miniconda3\envs\uhwr\python.exe"
NAGA = ROOT / "repos" / "nagamatsu" / "src"; SHIM = ROOT / "winshim"; FC = ROOT / "code" / "frame_code_a44e78b0"; C1 = ROOT / "code" / "c1_code_c7830637"
BB = ROOT / "code" / "fmdx_backbone_v1"; TIMM = STG / "exp" / "FMD" / "xs" / "code" / "pydeps_timm"; SMC = ROOT / "code" / "sm_v1"; D2 = ROOT / "code" / "d2moe_code_9a020b75"
D = STG / "data" / "m5_multiscript_v1"; OU = STG / "data" / "ouhdl_e0_v3"; DB = STG / "exp" / "FMD" / "xs" / "data" / "backbone"; MD = STG / "exp" / "FMD" / "xs" / "data" / "monitor_v1"
LUO = STG / "exp" / "M5" / "runs" / "luo" / "checkpoints" / "checkpoint_000006.pt"; ROUT = STG / "exp" / "D3_ROUTE" / "router"
GT = {"test_ouhdl": OU / "e0_test.npz", "test_casia_zh": STG / "exp/X4a/data/x4a_casia_zh.npz", "test_casia_en": STG / "exp/X4a/data/x4a_casia_en.npz",
      "test_hpl_tamil": STG / "exp/X4a/data/x4a_hpl_tamil.npz", "test_iam500": D / "m5_test_iam500.npz", "german_unipen_de_test": STG / "data/d2_v2/unipen_de/e0_test.npz",
      "german_onhw_all": STG / "data/d2_v2/onhw/e0_all.npz"}
R224 = {s: (STG / "exp/FMD/data/fmd_render224_test.npz" if s == "test_ouhdl" else MD / f"test_{s}_render224.npz") for s in GT}
SETS7 = list(GT)
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=("smr_luo", "smr_bb", "sma_luo", "sma_bb"))
    ap.add_argument("--tag", required=True); ap.add_argument("--frac", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backbone", default=""); ap.add_argument("--cmd", default=""); ap.add_argument("--smoke-steps", type=int); ap.add_argument("--smoke-maxu", type=int)
    a = ap.parse_args()
    RUN = ROOT / "runs" / "RQF" / "SM" / a.tag; EV = RUN / "eval"; EV.mkdir(parents=True, exist_ok=True)
    hist = lambda m: open(RUN / "job_history.txt", "a").write(f"{utc()} {m}\n")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if not (RUN / "STARTED.json").exists():
        json.dump(dict(tag=a.tag, lane="vps", cmd=a.cmd, gpu=gpu, started_utc=utc()), open(RUN / "STARTED.json", "w"))
    hist(f"sm_vps start kind={a.kind} tag={a.tag} frac={a.frac} seed={a.seed} backbone={a.backbone} gpu={gpu} cmd={a.cmd} smoke={a.smoke_steps}")
    steps = a.smoke_steps or 10000; save = a.smoke_steps or 1000; nms = steps // save
    npz = D / f"m5_ft_ouhdl_{a.frac}.npz"
    r224 = DB / "p01train_render224.npy" if a.frac == "p01" else STG / f"exp/RQF/data/render224/{a.frac}train_render224.npy"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", WANDB_MODE="disabled", PYTORCH_ALLOC_CONF="expandable_segments:True", PYTHONHASHSEED="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SMA_D2_DIR=str(D2), SMA_BB_DIR=str(BB))
    PP_FC = os.pathsep.join(map(str, [SHIM, FC / "lib", NAGA, FC, C1]))
    PP_BB = os.pathsep.join(map(str, [SHIM, NAGA, SMC, D2, BB, BB / "vendor", TIMM, FC]))
    def run(cmd, pp, log):
        e = dict(env, PYTHONPATH=pp)
        with open(RUN / log, "a") as f:
            f.write(f"\n## {utc()} {' '.join(map(str, cmd))}\n"); f.flush()
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=e)
        if r.returncode != 0:
            hist(f"FAILED rc={r.returncode} {log}"); sys.exit(r.returncode or 1)
    done = (RUN / "training_complete.json").exists()
    bbk = Path(a.backbone) if a.backbone else LUO
    if a.kind == "smr_luo":
        if not done:
            run([PY, FC / "e0_train.py", "--e0-train-npz", npz, "--condition", "rendered", "--run-dir", RUN, "--init-from", LUO, "--frame", "identity", "--frame-scripts", "ouhdl",
                 "--replay-npz", D / "m5_luo_train.npz", "--urdu-share", "0.5", "--batch-size", 128, "--gradient-accumulate-every", 4, "--learning-rate", "8e-5",
                 "--train-steps", steps, "--save-every", save, "--validation-every", save, "--amp", "--sequence-length", 320, "--image-size", 64, "--sampling-timesteps", 50,
                 "--train-fraction", "0.99", "--num-workers", 0, "--wandb-mode", "disabled", "--random-seed", a.seed, "--sample-count", 25, "--classes", 0, "--notes", f"SMR_{a.tag}"], PP_FC, "train.log")
        hold = RUN / "_hold"; hold.mkdir(exist_ok=True)
        if not (EV / "best_milestone.txt").exists():
            run([PY, C1 / "c1_vps_post.py", "--kind", "frame", "--run", RUN, "--arm", a.tag, "--frame", "identity", "--stage", STG, "--py", PY, "--frame-code", FC, "--c1-code", C1,
                 "--naga-src", NAGA, "--winshim", SHIM, "--procs", 16, "--queue-dir", hold, "--final-name", f"unused_{a.tag}"] + (["--smoke-units", a.smoke_maxu] if a.smoke_maxu else []), PP_FC, "post.log")
        best = int(open(EV / "best_milestone.txt").read().strip()); hist(f"selected ms{best}")
        if a.smoke_steps: hist("smoke: stop after selection"); return
        run([PY, C1 / "c1_sample.py", "--label", f"{a.tag}_ms{best}", "--run-dir", RUN, "--milestone", best, "--stage", STG, "--out-dir", RUN, "--batch-size", 256,
             "--sets"] + SETS7 + ["--frames", "identity", "--seeds", 0, 1, 2], PP_FC, "final.log")
    else:
        OUT = RUN / "ft"; OUT.mkdir(exist_ok=True); bbkind = "luo" if a.kind == "sma_luo" else "bb"
        if not done and a.kind == "smr_bb":
            run([PY, BB / "bb_train.py", "--arm-id", f"{a.tag}_FT", "--run-dir", RUN, "--regime", "ft", "--init-ckpt", bbk, "--train-npz", npz, "--render-npy", r224,
                 "--pseudo-npz", D / "m5_luo_train.npz", "--pseudo-render-npy", DB / "luo_train_render224.npy", "--pseudo-frac", "0.5",
                 "--train-steps", steps, "--save-every", save, "--batch-size", 32, "--grad-accum", 16, "--learning-rate", "8e-5", "--enc-lr", "8e-5", "--enc-warmup", 0,
                 "--seed", a.seed, "--num-workers", 0, "--lane", "vps"], PP_BB, "train.log")
        if not done and a.kind.startswith("sma"):
            run([PY, SMC / "sma_train.py", "--backbone-kind", bbkind] + (["--render-npy", r224] if bbkind == "bb" else []) + ["--stage", 1, "--run-dir", RUN, "--backbone", bbk,
                 "--backbone-weights", "ema", "--train-npz", npz, "--scripts", "ouhdl", "--script-override", "ouhdl", "--n-experts", 1, "--rank", 16, "--lora-alpha", 16,
                 "--lora-targets", "cross,attn,linear", "--gate-hidden", 128, "--gate-feat", "deepest", "--lb-coef", 0, "--batch-size", 64, "--grad-accum", 4,
                 "--train-steps", steps, "--save-every", save, "--val-every", steps * 10, "--learning-rate", "1e-4", "--keep-last", 0, "--num-workers", 0, "--seed", a.seed,
                 "--notes", f"RQF A2.3 SM-A {a.tag}"], PP_BB, "train.log")
        if not (RUN / "training_complete.json").exists(): hist("TRAINING_INCOMPLETE"); sys.exit(7)
        def ck(m): return RUN / f"bb_ms{m}.pt" if a.kind == "smr_bb" else RUN / "checkpoints" / f"checkpoint_{m:06d}.pt"
        def sample(m, gt, rd, setname, route, seed, out, extra=()):
            if a.kind == "smr_bb":
                cmd = [PY, BB / "bb_sample.py", "--ckpt", ck(m), "--render", rd, "--gt-npz", gt]
            else:
                cmd = [PY, SMC / "sma_sample.py", "--kind", bbkind, "--ckpt", ck(m), "--backbone", bbk, "--gt-npz", gt, "--set-name", setname, "--route", route] + (["--render", rd] if bbkind == "bb" else [])
            run(cmd + ["--seed", seed, "--sampling-timesteps", 50, "--ddim-eta", 0, "--batch-size", 250, "--out", out] + list(extra), PP_BB, "sample.log")
        vr = STG / "exp/FMD/data/fmd_render224_val1000.npz"
        if not (EV / "best_milestone.txt").exists():
            for m in range(1, nms + 1):
                o = EV / f"val__ouhdl1000__ms{m}_preds.npz"
                if not o.exists(): sample(m, OU / "e0_val.npz", vr, "val1000_ouhdl", "all", 0, o, ["--subset", 1000, "--subset-seed", 0])
                rr = EV / f"val__ouhdl1000__ms{m}_rows.csv"
                if not rr.exists():
                    run([PY, FC / "e0_score.py", "e0", "--npz", OU / "e0_val.npz", "--preds", o, "--method", f"{a.tag}_ms{m}", "--input", "rendered", "--out", rr, "--procs", 16], PP_FC, "score.log")
            r = subprocess.run([PY, str(FC / "m5_select.py"), "--eval-dir", str(EV), "--valsets", "ouhdl1000", "--out", str(EV / "selection.json"), "--criterion-note",
                                f"RQF A2.3 {a.kind} {a.tag} VPS: E0 rule on val1000 seed 0 DDIM-50 eta 0 EMA" + (" route=all" if a.kind.startswith("sma") else "")],
                               capture_output=True, text=True, env=dict(env, PYTHONPATH=PP_FC))
            best = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            if not best.isdigit(): hist(f"SELECTION_FAILED {r.stderr[-300:]}"); sys.exit(10)
            open(EV / "best_milestone.txt", "w").write(best + "\n")
        best = int(open(EV / "best_milestone.txt").read().strip()); hist(f"selected ms{best}")
        sets = SETS7 if not a.smoke_steps else ["test_ouhdl", "test_casia_zh"]
        for sd in ((0, 1, 2) if not a.smoke_steps else (0,)):
            for s in sets:
                o = OUT / f"{s}__{a.tag}_ft_ms{best}__seed{sd}_preds.npz"
                if (OUT / (o.stem + ".json")).exists(): continue
                sample(best, GT[s], R224[s], s, ROUT / f"manifest_{s}.csv", sd, o, (["--max-units", a.smoke_maxu] if a.smoke_maxu else []))
                open(OUT / "device_log.tsv", "a").write(f"{o.name}\t{gpu}\tvps\t{a.cmd}\t{utc()}\n")
    (RUN / "SAMPLING_DONE").write_text(utc()); hist("sampling_done")
if __name__ == "__main__":
    main()
