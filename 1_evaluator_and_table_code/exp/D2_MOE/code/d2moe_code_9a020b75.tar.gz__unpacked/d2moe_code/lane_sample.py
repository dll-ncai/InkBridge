"""VPS lane sampling driver: run d3b_sample.py / d2moe_sample.py over a list of sets x seeds, skip finished outputs,
write a sidecar json per preds file (GPU, torch, code sha256, checkpoint sha256, preds sha256).
python lane_sample.py --sampler d3b|d2moe --code CODE_DIR --run-dir RUN --out-dir OUT --sets SETGROUP|names --seeds 0 1 2 [--backbone PT] [--label D3B]
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
STAGE = Path(os.environ.get("UHWR_STAGE", r"$VPS_ROOT\uhwr\stage"))
SETS = {  # name -> (CLUSTER-A-relative npz, script tag)
    "val_ouhdl": ("data/m5_multiscript_v1/m5_val_ouhdl.npz", "ouhdl"),
    "val_casia_zh": ("data/m5_multiscript_v1/m5_val_casia_zh.npz", "casia_zh"),
    "val_casia_en": ("data/m5_multiscript_v1/m5_val_casia_en.npz", "casia_en"),
    "val_hpl_tamil": ("data/m5_multiscript_v1/m5_val_hpl_tamil.npz", "hpl_tamil"),
    "val_iam": ("data/m5_multiscript_v1/m5_val_iam.npz", "iam"),
    "test_ouhdl": ("data/ouhdl_e0_v3/e0_test.npz", "ouhdl"),
    "gold_ouhdl": ("data/ouhdl_e0_v3/e0_gold.npz", "ouhdl"),
    "test_casia_zh": ("exp/X4a/data/x4a_casia_zh.npz", "casia_zh"),
    "test_casia_en": ("exp/X4a/data/x4a_casia_en.npz", "casia_en"),
    "test_hpl_tamil": ("exp/X4a/data/x4a_hpl_tamil.npz", "hpl_tamil"),
    "test_iam500": ("data/m5_multiscript_v1/m5_test_iam500.npz", "iam"),
    "german_unipen_de_test": ("data/d2_v2/unipen_de/e0_test.npz", "unipen_de"),
    "german_onhw_all": ("data/d2_v2/onhw/e0_all.npz", "onhw"),
}
GROUPS = {
    "d3b_all": ["test_casia_zh", "test_casia_en", "test_hpl_tamil", "test_iam500", "german_unipen_de_test", "test_ouhdl", "gold_ouhdl", "german_onhw_all",
                "val_ouhdl", "val_casia_zh", "val_casia_en", "val_hpl_tamil", "val_iam"],
    "d2_s2": ["val_ouhdl", "gold_ouhdl", "test_ouhdl", "german_unipen_de_test", "german_onhw_all"],
    "d2_s1": ["val_casia_zh", "val_casia_en", "val_hpl_tamil", "val_iam", "val_ouhdl", "test_casia_zh", "test_casia_en", "test_hpl_tamil", "test_iam500"],
}

def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()

def latest_ckpt(run_dir):
    fs = sorted((Path(run_dir) / "checkpoints").glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return fs[-1] if fs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=("d3b", "d2moe"), required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--sets", nargs="+", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--backbone"); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--milestone", default="latest")
    ap.add_argument("--extra", default="", help="extra sampler args as one quoted string, e.g. \"--max-units 8\"")
    a = ap.parse_args()
    a.extra = a.extra.split()
    sets = []
    for s in a.sets:
        sets += GROUPS.get(s, [s])
    ck = latest_ckpt(a.run_dir) if a.milestone == "latest" else Path(a.run_dir) / "checkpoints" / f"checkpoint_{int(a.milestone):06d}.pt"
    if ck is None or not ck.is_file():
        print(json.dumps(dict(event="error", msg=f"no checkpoint in {a.run_dir}"))); sys.exit(3)
    ms = int(ck.stem.split("_")[1]); ck_sha = sha256_file(ck)
    code = Path(a.code); script = code / ("d3b_sample.py" if a.sampler == "d3b" else "d2moe_sample.py")
    code_sha = {p.name: sha256_file(p) for p in sorted(code.glob("*.py"))}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    log = open(out / f"{a.label}_ms{ms}_sample_log.jsonl", "a", encoding="utf-8"); rc_all = 0; t_all = time.time()
    for s in sets:
        rel, tag = SETS[s]; npz = STAGE / rel
        for sd in a.seeds:
            fn = out / f"{s}__{a.label}_ms{ms}__seed{sd}_preds.npz"
            if fn.is_file():
                continue
            tmp = out / f"{s}__{a.label}_ms{ms}__seed{sd}_preds.part.npz"
            cmd = [sys.executable, str(script), "--checkpoint", str(ck), "--npz", str(npz), "--script-override", tag, "--sampling-timesteps", "50",
                   "--ddim-eta", "0", "--batch-size", str(a.batch_size), "--seed", str(sd), "--out", str(tmp)] + a.extra
            if a.sampler == "d2moe":
                cmd += ["--backbone", a.backbone]
            t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            last = [l for l in p.stdout.strip().splitlines() if l.strip()]
            if p.returncode != 0 or not tmp.is_file():
                rec = dict(event="fail", set=s, seed=sd, rc=p.returncode, stderr_tail=p.stderr[-1500:], stdout_tail=p.stdout[-500:])
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True); rc_all = 5; continue
            os.replace(tmp, fn)
            side = dict(preds_file=fn.name, preds_sha256=sha256_file(fn), set=s, npz_clusterA=f"$U/{rel}", npz=str(npz), script_tag=tag, seed=sd,
                        label=a.label, sampler=script.name, code_dir=str(code), code_sha256=code_sha, checkpoint=str(ck), checkpoint_sha256=ck_sha, milestone=ms,
                        run_dir=a.run_dir, backbone=a.backbone, device=gpu, host=platform.node(), seconds=round(time.time() - t0, 1),
                        sampler_meta_line=last[-1] if last else None, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), extra=a.extra)
            if a.sampler == "d2moe":
                side["backbone_sha256"] = sha256_file(a.backbone)
            json.dump(side, open(str(fn)[:-4] + ".json", "w"), indent=1)
            log.write(json.dumps(dict(event="done", **{k: side[k] for k in ("set", "seed", "preds_file", "preds_sha256", "seconds")})) + "\n"); log.flush()
            print(json.dumps(dict(event="done", set=s, seed=sd, seconds=side["seconds"])), flush=True)
    log.write(json.dumps(dict(event="all_done", label=a.label, ms=ms, rc=rc_all, wall_seconds=round(time.time() - t_all, 1))) + "\n"); log.close()
    sys.exit(rc_all)

if __name__ == "__main__":
    main()
