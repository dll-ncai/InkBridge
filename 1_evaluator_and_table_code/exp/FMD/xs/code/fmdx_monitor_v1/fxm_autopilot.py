#!/usr/bin/env python3
"""FMDX warm$SCRATCH_ROOT monitor autopilot (CLUSTER-A, stdlib only; runs as a cpuidle Slurm job).
Replaces the control-plane loop (lead 8 Nov: one short CLUSTER-A command per hourly poll; no CLUSTER-A<->CLUSTER-B relays; sampling on CLUSTER-A only).
Per cycle: new finished arms (training_complete + eval/best_milestone.txt in X/runs/<arm>_clusterA) -> FT race copies (3 earliest
sbatch --test-only of FTC) + A9 bb sampling (3 earliest of GT9); FT selected -> scancel FT copies, A9 ft sampling (skipped when the
preds already exist); A9 race decided (3 seed locks or preds complete) -> scancel pending tasks of that job name; stale locks / missing
seeds with nothing queued -> resubmit seed (max 2); preds complete -> adapt+score (lane_score array); bb+ft complete -> compose (A4.4)
-> score cmp; new complete arm:kind set (no scoring queued) -> tables (tables_job_a9.sbatch, no verdicts).
Writes WARM_MONITOR_STATUS.md, state/warm_state.json, state/autopilot_summary.txt, state/autopilot_events.log, ACTIONS_LOG.md. Stops at --until."""
import json, os, glob, subprocess, time, re, sys, argparse, calendar
ap = argparse.ArgumentParser(); ap.add_argument("--until", default="2026-11-10T18:00:00Z"); ap.add_argument("--cycle", type=int, default=600)
A = ap.parse_args()
R = "$U"; X = R + "/exp/FMD/xs"; MR = X + "/runs/fmdx_monitor"; MC = X + "/code/fmdx_monitor_v1"
SF = MR + "/state/warm_state.json"; ACT = MR + "/ACTIONS_LOG.md"; SUM = MR + "/state/autopilot_summary.txt"; EVL = MR + "/state/autopilot_events.log"
PYCPU = f"apptainer exec -B $SCRATCH_ROOT {R}/images/torch27.sif {R}/venvs/cpu/bin/python"
SETS6 = {"test_casia_zh": "exp/X4a/data/x4a_casia_zh.npz", "test_casia_en": "exp/X4a/data/x4a_casia_en.npz", "test_hpl_tamil": "exp/X4a/data/x4a_hpl_tamil.npz",
         "test_iam500": "data/m5_multiscript_v1/m5_test_iam500.npz", "german_unipen_de_test": "data/d2_v2/unipen_de/e0_test.npz", "german_onhw_all": "data/d2_v2/onhw/e0_all.npz"}
GTMAP = ",".join(f"{k}={R}/{g}" for k, g in SETS6.items())
GTMAP_FT = f"test_ouhdl={R}/data/ouhdl_e0_v3/e0_test.npz,val1000_ouhdl={R}/data/ouhdl_e0_v3/e0_val.npz"
NEED = {"bb": 18, "ft": 24, "cmp": 18}; PER_SEED = {"bb": 6, "ft": 8, "cmp": 6}
GT9 = [("gpuidle", "gpu:a4000:1"), ("gpuidle", "gpu:a30:1"), ("gpuidle", "gpu:a100:1"), ("gpuidle", "gpu:h100:1"), ("gpuidle", "gpu:v100:1"),
       ("b200idle", "gpu:b200:1"), ("llms", "gpu:rtx6000:1"), ("llms", "gpu:b300:1")]
FTC = [("llms", "gpu:1"), ("b200idle", "gpu:1"), ("gpuidle", "gpu:a30:1"), ("gpuidle", "gpu:a100:1"), ("gpuidle", "gpu:h100:1"), ("gpuidle", "gpu:a4000:1")]
SKIP = {"SMOKEA"}
def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sh(cmd, t=600):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t).stdout
    except Exception: return ""
def log(msg, cand=False):
    line = f"- {utc()} [FMDX warm-monitor autopilot, <agent> track] {msg}\n"
    for p in [ACT, EVL] + ([R + "/exp/CAND/queue/cand_log.md", X + "/admin/cancellation_log.md"] if cand else []):
        with open(p, "a") as f: f.write(line)
def ld():
    try: return json.load(open(SF))
    except Exception: return {"arms": {}}
def sv(S): open(SF + ".tmp", "w").write(json.dumps(S, indent=1)); os.replace(SF + ".tmp", SF)
def npred(a, k): return len(glob.glob(f"{MR}/preds/{a}/{k}/*_preds.npz"))
def earliest(opts, t, mem, cpus):
    res = []
    for p, g in opts:
        o = sh(f"sbatch --test-only -p {p} --gres={g} -c {cpus} --mem={mem} -t {t} --exclude=<node> --wrap true 2>&1", 60)
        m = re.search(r"start at (\d{4}-\d\d-\d\dT[\d:]+)", o)
        if m: res.append((m.group(1), p, g))
    res.sort(); return res[:3]
def submit_a9(a, k, ms, arr="0-2"):
    ids = []
    for s, p, g in earliest(GT9, "06:00:00", "24G", 4):
        o = sh(f"cd {MR} && sbatch --parsable -J fxm_as9_{a}_{k} -p {p} --gres={g} --array={arr} {MC}/arm_sample_a9.sbatch {a} {k} clusterA {ms} 2>&1").strip().split("\n")[-1]
        ids.append(f"{p}:{g}={o} (est {s})")
    log(f"A9 sampling race copies (3 earliest test-only) {a} {k} ms{ms} array {arr}: {ids}"); return ids
def launch_ft(a, ms):
    src = f"{X}/runs/{a}_clusterA/bb_ms{ms}.pt"; ids = []
    for s, p, g in earliest(FTC, "2-00:00:00", "48G", 8):
        o = sh(f"cd {MR} && sbatch --parsable -J fxm_ft_{a} -p {p} --gres={g} {MC}/ft_clusterA.sbatch {a} {src} 2>&1").strip().split("\n")[-1]
        if o.isdigit(): ids.append(o)
        log(f"FT race copy {a} from {src}: {p}:{g} job {o} (est {s})")
    return ids
def meta_update(a):
    fm = a.rsplit("_", 1)[0]; mp = MR + "/tables/arm_meta.json"
    if fm in ("NOFM", "PEINIT"): st = "control"
    else:
        st = "unknown"
        for v in sorted(glob.glob(f"{X}/verdicts/verdict_{fm}.json") + glob.glob(f"{X}/hub/verdicts/verdict_{fm}.json"), key=os.path.getmtime, reverse=True):
            try: st = json.load(open(v)).get("status", "unknown"); break
            except Exception: pass
    m = json.load(open(mp)); m[a] = {"fm": "-" if st == "control" else fm, "status": st}
    open(mp + ".tmp", "w").write(json.dumps(m)); os.replace(mp + ".tmp", mp)
def score(a, k):
    sets = list(SETS6) + (["test_ouhdl", "val1000_ouhdl"] if k == "ft" else [])
    src = f"{MR}/preds/{a}/{k}"; dst = f"{MR}/score_in/{a}_{k}"; gm = GTMAP + ("," + GTMAP_FT if k == "ft" else "")
    o = sh(f"rm -rf {dst}; {PYCPU} {MC}/adapt_sidecars.py --src {src} --dst {dst} --gt-map '{gm}' 2>&1 | tail -1")
    open(f"{MR}/score_in/{a}_{k}.tasks", "w").write("\n".join(f"{s}__{a}_{k}_ms*_preds.npz" for s in sets) + "\n")
    j = sh(f"cd {MR} && PREDS={dst} OUT={MR}/score_out TASKS={MR}/score_in/{a}_{k}.tasks sbatch --parsable -p cpuidle --requeue -J fxm_sc_{a}_{k} --array=1-{len(sets)} --export=ALL,PREDS,OUT,TASKS {MC}/lane_score_array_mon.sh").strip()
    log(f"scoring {a} {k}: adapt={o.strip()[-40:]} job {j}"); return j or "submitted"
def compose(a):
    o = sh(f"mkdir -p {MR}/preds/{a}/cmp; {PYCPU} {MC}/compose_preds.py --arm {a} --bb-dir {MR}/preds/{a}/bb --ft-dir {MR}/preds/{a}/ft --out {MR}/preds/{a}/cmp 2>&1 | tail -2")
    log(f"composed preds (A4.4) {a}: {o.strip()[-120:]}"); return o.strip()[-80:] or "done"
def queued(name): return sh(f"squeue -u $USER -h -n {name} -o %i | wc -l").strip() not in ("", "0")
def cancel_pending(a, k, why):
    pend = sh(f"squeue -u $USER -h -t PD -r -o %i -n fxm_as9_{a}_{k}").split()
    if pend: sh("scancel " + " ".join(pend)); log(f"A9 race {a} {k}: {why}; cancelled pending tasks {pend}", cand=True)
def cycle(S):
    acts = []
    for d in sorted(glob.glob(f"{X}/runs/*_clusterA")):
        a = os.path.basename(d)[:-5]
        if a in S["arms"] or a in SKIP or not os.path.exists(d + "/training_complete.json"): continue
        try: ms = int(open(d + "/eval/best_milestone.txt").read().strip())
        except Exception: continue
        st = S["arms"][a] = dict(bb_ms=ms, ft_ms=None, rows=dict(bb=0, ft=0, cmp=0), since=utc())
        meta_update(a)
        bmf = f"{MR}/ft/{a}_FT_clusterA/eval/best_milestone.txt"
        st["ft_copies"] = [] if os.path.exists(bmf) or queued(f"fxm_ft_{a}") else launch_ft(a, ms)
        if npred(a, "bb") < 18 and not queued(f"fxm_as9_{a}_bb"): st["bb_sample"] = submit_a9(a, "bb", ms)
        acts.append(f"{a} training complete (best ms{ms}); FT copies {st['ft_copies']}")
    for a, st in S["arms"].items():
        st.setdefault("rows", dict(bb=0, ft=0, cmp=0))
        bm = f"{MR}/ft/{a}_FT_clusterA/eval/best_milestone.txt"
        if st.get("ft_ms") is None and os.path.exists(bm):
            st["ft_ms"] = int(open(bm).read().strip()); cps = st.get("ft_copies", [])
            if cps: sh("scancel " + " ".join(cps))
            log(f"{a} FT selected ms{st['ft_ms']} (val1000 E0); scancel issued for FT copies {cps} (winner in ft/{a}_FT_clusterA/job_history.txt)", cand=True)
            if npred(a, "ft") < 24 and not queued(f"fxm_as9_{a}_ft"): st["ft_sample"] = submit_a9(a, "ft", st["ft_ms"])
            acts.append(f"{a} FT done ms{st['ft_ms']}")
        for k in ("bb", "ft"):
            ms = st.get("bb_ms") if k == "bb" else st.get("ft_ms")
            if ms is None: continue
            pd_ = f"{MR}/preds/{a}/{k}"
            if npred(a, k) >= NEED[k]:
                if queued(f"fxm_as9_{a}_{k}"): cancel_pending(a, k, "preds complete")
                continue
            if len(glob.glob(pd_ + "/.lock_*")) == 3: cancel_pending(a, k, "3 seed locks taken")
            for s in (0, 1, 2):
                if len(glob.glob(f"{pd_}/*__seed{s}_preds.json")) >= PER_SEED[k]: continue
                lk = f"{pd_}/.lock_{a}_{k}_ms{ms}_seed{s}"; stale = False
                if os.path.isdir(lk):
                    ow = open(lk + "/owner").read().strip() if os.path.exists(lk + "/owner") else ""
                    stt = sh(f"sacct -n -X -j {ow} -o State 2>/dev/null | head -1").strip() if ow else ""
                    stale = bool(re.match(r"(CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_ME|PREEMPTED|BOOT_FAIL|DEADLINE|COMPLETED)", stt))
                else:
                    stale = not queued(f"fxm_as9_{a}_{k}")
                key = f"{a}|{k}|{s}"
                if stale and S.setdefault("repairs", {}).get(key, 0) < 2:
                    sh(f"rm -rf {lk}"); S["repairs"][key] = S["repairs"].get(key, 0) + 1
                    submit_a9(a, k, ms, arr=str(s)); acts.append(f"{a} {k} seed{s} resubmitted"); log(f"{a} {k} seed{s}: stale lock or nothing queued; resubmitted (repair {S['repairs'][key]})")
        n = {k: npred(a, k) for k in ("bb", "ft", "cmp")}
        rows = {k: len(glob.glob(f"{MR}/score_out/rows/*__{a}_{k}_ms*__rows031.csv")) for k in ("bb", "ft", "cmp")}
        for k in ("bb", "ft"):
            if n[k] >= NEED[k] and rows[k] < NEED[k] and not queued(f"fxm_sc_{a}_{k}") and S.setdefault("score_tries", {}).get(f"{a}|{k}", 0) < 2:
                S["score_tries"][f"{a}|{k}"] = S["score_tries"].get(f"{a}|{k}", 0) + 1; st[k + "_scored"] = score(a, k); acts.append(f"{a} {k} scoring")
        if n["bb"] >= 18 and n["ft"] >= 24 and n["cmp"] < 18:
            st["composed"] = compose(a); acts.append(f"{a} composed"); n["cmp"] = npred(a, "cmp")
        if n["cmp"] >= 18 and rows["cmp"] < 18 and not queued(f"fxm_sc_{a}_cmp") and S.setdefault("score_tries", {}).get(f"{a}|cmp", 0) < 2:
            S["score_tries"][f"{a}|cmp"] = S["score_tries"].get(f"{a}|cmp", 0) + 1; st["cmp_scored"] = score(a, "cmp"); acts.append(f"{a} cmp scoring")
        st["rows"] = rows; st["npreds"] = n
    full = ",".join(sorted(f"{a}:{k}" for a, st in S["arms"].items() for k in ("bb", "ft", "cmp") if st["rows"][k] >= NEED[k]))
    if full and full != S.get("tables_built_for") and not queued("fxm_tables") and not sh("squeue -u $USER -h -o %j | grep '^fxm_sc_'").strip():
        j = sh(f"sbatch --parsable {MC}/tables_job_a9.sbatch 'autopilot {utc()}'").strip()
        S["tables_built_for"] = full; S["tables_job"] = j; S.setdefault("tables_history", []).append([utc(), j, full])
        log(f"tables job {j} for {full}"); acts.append(f"tables job {j}")
    return acts
def write_status(S):
    L = [f"# WARM_MONITOR_STATUS (facts only) - autopilot {utc()}", "Track: FMDX warm monitor (replacement instance, <agent>; reason: <agent> API failing). Prereg: FMD_PREREG_2026-11-03.md incl. A9/A9.4 (sha in PREREG_FROZEN.flag).",
         "Autopilot: CLUSTER-A cpuidle job (code/fmdx_monitor_v1/fxm_autopilot.py); sampling on CLUSTER-A only (lead 8 Nov); no CLUSTER-A<->CLUSTER-B relays.", "",
         "| arm | bb ms | FT ms | preds bb/ft/cmp | rows bb/ft/cmp | since |", "|---|---|---|---|---|---|"]
    for a, st in sorted(S["arms"].items()):
        n = st.get("npreds", {}); r = st["rows"]
        L.append(f"| {a} | {st.get('bb_ms')} | {st.get('ft_ms')} | {n.get('bb','')}/{n.get('ft','')}/{n.get('cmp','')} | {r['bb']}/{r['ft']}/{r['cmp']} | {st.get('since','')} |")
    L += ["", "## Last table builds", *[f"- {t} job {j}: {k}" for t, j, k in S.get("tables_history", [])[-3:]], "", "## Running arms (fx_* on CLUSTER-A)", "```",
          sh("squeue -u $USER -h -t R -o '%i %j %M %P' | grep ' fx_' | head -30").strip(), "```", "", "## Autopilot events (last 40)", "```",
          "".join(open(EVL).readlines()[-40:]) if os.path.exists(EVL) else "", "```"]
    open(X + "/runs/WARM_MONITOR_STATUS.md.tmp", "w").write("\n".join(L)); os.replace(X + "/runs/WARM_MONITOR_STATUS.md.tmp", X + "/runs/WARM_MONITOR_STATUS.md")
    full = [a for a, st in S["arms"].items() if st["rows"]["ft"] >= 24 and st["rows"]["cmp"] >= 18 and st["rows"]["bb"] >= 18]
    tj = S.get("tables_job", ""); tst = sh(f"sacct -n -X -j {tj} -o State | head -1").strip() if tj else ""
    stage = {a: ("TABLED" if a in full else f"rows {st['rows']['bb']}/{st['rows']['ft']}/{st['rows']['cmp']} ftms={st.get('ft_ms')}") for a, st in S["arms"].items()}
    open(SUM, "w").write(json.dumps(dict(utc=utc(), full=sorted(full), tables_job=tj, tables_state=tst, tables_for_n=len(S.get("tables_built_for", "").split(",")), stage=stage,
                                         last_events=[l.strip()[:200] for l in (open(EVL).readlines()[-5:] if os.path.exists(EVL) else [])])))
until = calendar.timegm(time.strptime(A.until, "%Y-%m-%dT%H:%M:%SZ"))
log(f"autopilot start slurm={os.environ.get('SLURM_JOB_ID')} restart={os.environ.get('SLURM_RESTART_COUNT','0')} until {A.until}")
while time.time() < until:
    S = ld()
    try:
        cycle(S); sv(S); write_status(S)
    except Exception as e:
        log(f"cycle error: {repr(e)[:200]}")
    time.sleep(A.cycle)
S = ld(); write_status(S); log("autopilot reached --until; exiting")
