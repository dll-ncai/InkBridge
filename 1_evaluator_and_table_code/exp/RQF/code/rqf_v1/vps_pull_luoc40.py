#!/usr/bin/env python3
"""RQF R2b: pull LUOC40 VPS outputs to CLUSTER-A (key ~/.ssh/uhwr_staging_vps_ed25519, valid until 10 Nov 18:00Z, RQF pulls only), verify preds sha256 against the sidecars, patch sidecars
additively (source_npz_clusterA), hand the preds to the existing scoring chain (post_run.sh luoft) and the tables job, and relay the selected LUOC40 backbone to CLUSTER-A RQF/runs/LUOC40_bb
(opens gate 24964073) while cancelling the pending CLUSTER-A LUOC40 backbone copies (race rule: first to START wins). Every event: one UTC line in exp/RQF/queue/rqf_log.md.
Run: sbatch cpu1.sbatch vps_pull_luoc40.py  (poll 600 s, max 70 h; stop: touch exp/RQF/admin/STOP_VPS_PULL)"""
import glob, hashlib, io, json, os, subprocess, sys, tarfile, time
U = "$U"; Q = U + "/exp/RQF"; QR = Q + "/runs"; QP = Q + "/preds"; LOG = Q + "/queue/rqf_log.md"; ST = Q + "/admin/vps_pull_state.json"
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/uhwr_staging_vps_ed25519"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "Administrator@<ip>"]
VR = "$VPS_ROOT/uhwr/runs/RQF"; VQ = "$VPS_ROOT/uhwr/runner"
SEEDS = {"20261034": "526", "20261035": "527", "20261036": "528"}
GT = {"test_ouhdl": "data/ouhdl_e0_v3/e0_test.npz", "test_casia_zh": "exp/X4a/data/x4a_casia_zh.npz", "test_casia_en": "exp/X4a/data/x4a_casia_en.npz", "test_hpl_tamil": "exp/X4a/data/x4a_hpl_tamil.npz",
      "test_iam500": "data/m5_multiscript_v1/m5_test_iam500.npz", "german_unipen_de_test": "data/d2_v2/unipen_de/e0_test.npz", "german_onhw_all": "data/d2_v2/onhw/e0_all.npz"}
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def log(m): open(LOG, "a").write(f"{utc()} [rqf-s2 <agent> vps-pull] {m}\n")
def ssh(cmd, binary=False, timeout=900):
    r = subprocess.run(SSH + [cmd], capture_output=True, timeout=timeout)
    return r.stdout if binary else r.stdout.decode(errors="replace").replace("\r", "")
def exists(p): return "yes" in ssh(f'if exist "{p.replace("/", chr(92))}" (echo yes) else (echo no)')
def pull(path_rel, dest, excl=()):
    ex = " ".join(f"--exclude={e}" for e in excl)
    data = ssh(f"tar -cf - -C {VR} {ex} {path_rel}", binary=True)
    if len(data) < 100: return False
    os.makedirs(dest, exist_ok=True); tarfile.open(fileobj=io.BytesIO(data)).extractall(dest); return True
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def sb(args): return subprocess.run(["env", "-i", "HOME=" + os.environ["HOME"], "USER=" + os.environ["USER"], "PATH=/usr/bin:/bin:/usr/local/bin", "sbatch", "--parsable"] + args, capture_output=True, text=True).stdout.strip().split(";")[0]
st = json.load(open(ST)) if os.path.exists(ST) else {}
def save(): json.dump(st, open(ST, "w"), indent=1)
log("puller start (poll 600 s)"); import calendar; t_end = calendar.timegm((2026, 11, 14, 17, 30, 0))   # stops itself at 14 Nov 17:30Z (key approved to 14 Nov 18:00Z, lead)
while time.time() < t_end and not os.path.exists(Q + "/admin/STOP_VPS_PULL"):
    try:
        # (1) race: VPS backbone started -> cancel pending CLUSTER-A copies once
        if not st.get("race_cancel") and exists(f"{VR}/LUOC40_bb/run_config.json") and not os.path.exists(QR + "/LUOC40_bb/.local_lock"):
            ids = "24964074 24964380"; subprocess.run(["scancel"] + ids.split(), capture_output=True)
            log(f"VPS LUOC40_bb started first (CLUSTER-A copies not started): scancel {ids}; CLUSTER-B copy 3722157 must be cancelled on CLUSTER-B (scancel 3722157)"); st["race_cancel"] = True; save()
        # (2) backbone relay
        if not st.get("bb_relay") and exists(f"{VR}/LUOC40_bb/eval/best_milestone.txt") and exists(f"{VR}/LUOC40_bb/training_complete.json"):
            best = int(ssh(f'type {VR.replace("/", chr(92))}\\LUOC40_bb\\eval\\best_milestone.txt').split()[0]); dst = QR
            if not os.path.exists(QR + "/LUOC40_bb/.local_lock"):
                pull("LUOC40_bb", dst, excl=("checkpoints", "training_samples", "wandb", "*.pt"))
                ck = f"LUOC40_bb/checkpoints/checkpoint_{best:06d}.pt"; pull(ck, dst)
                p = f"{dst}/{ck}"; remote = ssh(f'powershell -NoProfile -Command "(Get-FileHash $VPS_ROOT\\uhwr\\runs\\RQF\\LUOC40_bb\\checkpoints\\checkpoint_{best:06d}.pt).Hash.ToLower()"').strip()
                ok = os.path.exists(p) and sha(p) == remote
                log(f"LUOC40_bb relayed from VPS: best ms{best}, checkpoint sha256 {remote[:16]} {'OK' if ok else 'MISMATCH'}; eval/best_milestone.txt present -> gate 24964073 opens")
                if ok: st["bb_relay"] = True; save()
            else: log("LUOC40_bb: CLUSTER-A copy started first; VPS backbone not relayed"); st["bb_relay"] = "skipped"; save()
        # (3) FT / backbone sampling outputs
        jobs = {f"LUOC40_p01_s{s}": f"{n}f_rqf_luoc40_LUOC40_p01_s{s}_final" for s, n in SEEDS.items()}
        jobs["LUOC40_bb"] = "529_rqf_luoc40_bb_sample"
        for arm, fin in jobs.items():
            if st.get(arm) or not exists(f"{VQ}/done/{fin}.cmd"): continue
            tmp = f"{Q}/admin/.vpspull_{arm}"; os.makedirs(tmp, exist_ok=True)
            pull(arm, tmp, excl=("checkpoints", "training_samples", "wandb", "*.pt"))
            tag = arm.replace("LUOC40_p01", "LUOC40VPS_p01"); src = f"{tmp}/{arm}"; pdst = f"{QP}/{tag}/preds"; os.makedirs(pdst, exist_ok=True); bad = []; n = 0
            for f in sorted(glob.glob(src + "/preds/*_preds.npz")):
                j = f[:-4] + ".json"; s = json.load(open(j)); d = sha(f)
                if d != s.get("preds_sha256"): bad.append(os.path.basename(f)); continue
                setname = os.path.basename(f).split("__")[0]
                side = dict(s); side["source_npz_clusterA"] = f"{U}/{GT[setname]}"; side["rqf_vps_pull"] = dict(utc=utc(), vps_run=f"{VR}/{arm}", device="VPS RTX 4090")
                nb = os.path.basename(f).replace(arm, tag)
                for k_ in ("label", "route_mode", "method"):
                    if isinstance(side.get(k_), str): side[k_] = side[k_].replace(arm, tag)
                side["rqf_vps_pull"]["evidence_label"] = "EXPLORATORY (VPS RTX 4090; e0_train.py, torch 2.9.1, backbone ms1)"
                os.replace(f, f"{pdst}/{nb}"); json.dump(side, open(f"{pdst}/{nb[:-4]}.json", "w"), indent=1); n += 1
            if bad: log(f"{arm}: sha mismatch {bad}; not marked pulled"); continue
            os.makedirs(f"{QR}/{tag}", exist_ok=True); subprocess.run(["cp", "-rp", src + "/.", f"{QR}/{tag}/"])
            pj = sb(["-J", f"rqfpost_{tag}", "-p", "cpuidle", "--time=1-00:00:00", "--cpus-per-task=8", "--mem=32G", "-o", f"{Q}/logs/%x_%j.out", f"{Q}/code/rqf_v1/cpu1.sbatch", "post_run.sh", tag, "luoft", "."])
            tj = sb(["-J", "rqf_tables_vps", "--dependency=afterany:" + pj, "-p", "cpuidle", "--time=1-00:00:00", "-o", f"{Q}/logs/%x_%j.out", f"{Q}/code/rqf_v1/cpu1.sbatch", "tables_rqf_interim.sh"])
            log(f"{arm}: pulled {n} preds (sha256 vs sidecar OK) from VPS final job {fin}; scoring job {pj}, tables job {tj}"); st[arm] = dict(post=pj, tables=tj); save()
    except Exception as e:
        log(f"puller error {type(e).__name__}: {str(e)[:150]}")
    if all(st.get(a) for a in ["LUOC40_p01_s20261034", "LUOC40_p01_s20261035", "LUOC40_p01_s20261036", "LUOC40_bb"]): break
    time.sleep(600)
log("puller end")
