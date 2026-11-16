#!/usr/bin/env python3
"""R2d in-job gate (stdlib only). Polls the training log every --every s while the training process group lives.
GATE_FAIL (json in --run-dir, then SIGTERM to the process group) if (a) any parsed loss is non-finite (nan/inf), or
(b) at step >= --min-step the median of the last 50 parsed losses exceeds --max-loss (divergence; LUO recipe loss is
~0.012 at 3k steps and ~0.002 at the end). At every new milestone checkpoint (checkpoints/checkpoint_*.pt or bb_ms*.pt)
one line 'gate_ok|gate_fail ms=<k> step=<s> median50=<v>' is appended to gate_log.txt.
--kind tqdm: m5_train / e0_train logs ('loss: X: ..| step/total'); --kind jsonl: bb_train train_log.jsonl (loss_den, finite)."""
import argparse, glob, json, math, os, re, signal, statistics, time
ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=("tqdm", "jsonl"), required=True); ap.add_argument("--log", required=True)
ap.add_argument("--run-dir", required=True); ap.add_argument("--min-step", type=int, required=True)
ap.add_argument("--max-loss", type=float, default=0.10); ap.add_argument("--pgid", type=int, required=True)
ap.add_argument("--every", type=float, default=60); ap.add_argument("--once", action="store_true", help="test: one pass, no kill")
a = ap.parse_args()
RX = re.compile(r"loss: ([^:\s]+):.*?\|\s*(\d+)/(\d+)")
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def parse():
    vals = []
    for f in sorted(glob.glob(a.log)):
        try:
            with open(f, "rb") as fh:
                fh.seek(0, 2); n = fh.tell(); fh.seek(max(0, n - 400000)); txt = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if a.kind == "tqdm":
            for line in re.split(r"[\r\n]", txt):
                m = RX.search(line)
                if m:
                    try: v = float(m.group(1))
                    except ValueError: v = float("nan")
                    vals.append((int(m.group(2)), v))
        else:
            for line in txt.splitlines():
                if '"event": "train"' not in line: continue
                try: r = json.loads(line)
                except ValueError: continue
                v = r.get("loss_den"); v = float("nan") if v is None or r.get("finite") is False else float(v)
                vals.append((int(r.get("step", 0)), v))
    return vals
def alive(pg):
    try: os.killpg(pg, 0); return True
    except OSError: return False
def ms_files():
    return sorted(glob.glob(os.path.join(a.run_dir, "checkpoints", "checkpoint_*.pt")) + glob.glob(os.path.join(a.run_dir, "bb_ms*.pt")))
seen = set(ms_files())
while True:
    vals = parse(); reason = None
    bad = [(s, v) for s, v in vals if not math.isfinite(v)]
    last = [v for _, v in vals[-50:] if math.isfinite(v)]; step = vals[-1][0] if vals else 0
    med = statistics.median(last) if last else float("nan")
    if bad: reason = f"non-finite loss at step {bad[0][0]}"
    elif step >= a.min_step and last and med > a.max_loss: reason = f"median50 loss {med:.5f} > {a.max_loss} at step {step}"
    new = [f for f in ms_files() if f not in seen]
    for f in new:
        seen.add(f)
        with open(os.path.join(a.run_dir, "gate_log.txt"), "a") as g:
            g.write(f"{utc()} {'gate_fail' if reason else 'gate_ok'} ms={os.path.basename(f)} step={step} median50={med:.5f}\n")
    if reason:
        rec = dict(gate="R2d in-job GATE (non-finite / divergence)", reason=reason, step=step, median50=med, max_loss=a.max_loss,
                   min_step=a.min_step, log=a.log, utc=utc(), slurm=os.environ.get("SLURM_JOB_ID"))
        with open(os.path.join(a.run_dir, "GATE_FAIL"), "w") as g: json.dump(rec, g, indent=1)
        print("GATE_FAIL", rec, flush=True)
        if not a.once:
            try: os.killpg(a.pgid, signal.SIGTERM); time.sleep(30); os.killpg(a.pgid, signal.SIGKILL)
            except OSError: pass
        break
    if a.once or not alive(a.pgid): break
    time.sleep(a.every)
