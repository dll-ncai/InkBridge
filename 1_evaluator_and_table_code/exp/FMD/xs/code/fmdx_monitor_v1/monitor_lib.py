# FMDX warm monitor control-plane library (runs in the `repl` kernel via exec(open(...).read())). Needs `host`.
import json, time, re, base64, hashlib, subprocess, sys, os
R='$U'; X=R+'/exp/FMD/xs'; N='$U_B'
MC=X+'/code/fmdx_monitor_v1'; MR=X+'/runs/fmdx_monitor'; MD=X+'/data/monitor_v1'; ACT=MR+'/ACTIONS_LOG.md'; EVF=MR+'/state/events.json'
c=host.compute.create('ssh:clusterA'); d=host.compute.create('ssh:clusterB')   # VPS handle removed 4 Nov (VPS off, <author> decision)
def shr(comp, cmd, t=90, tries=5, intent='cmd'):
    last=''
    for i in range(tries):
        try:
            r=comp.call_command(cmd, run_timeout_s=t, intent=intent); o=r.stdout
            if 'banner exchange' in o or 'UNKNOWN port' in o or (o.strip()=='' and not r.ok): last=o; time.sleep(30); continue
            return o
        except Exception as e:
            last=str(e); time.sleep(30)
    return 'FAILED: '+last[:300]
def put(path, text, mode='755'):
    b=base64.b64encode(text.encode()).decode()
    cmds=[f"mkdir -p $(dirname {path}); : > {path}.b64"]+[f"printf %s '{b[i:i+3000]}' >> {path}.b64" for i in range(0,len(b),3000)]
    cmds.append(f"base64 -d {path}.b64 > {path} && rm {path}.b64 && chmod {mode} {path}; sha256sum {path} | cut -c1-16")
    return shr(c, "; ".join(cmds), t=60, intent='Write '+path.split('/')[-1]).strip()
def vput(winpath, data: bytes):
    b=base64.b64encode(data).decode(); tmp=winpath+'.b64'; first=True
    for i in range(0,len(b),1800):
        op='Set-Content' if first else 'Add-Content'
        v.call_command(f"powershell -NoProfile -Command \"{op} -Path '{tmp}' -Value '{b[i:i+1800]}' -NoNewline\"", run_timeout_s=60, intent='Write chunk to VPS'); first=False
    ps=f"powershell -NoProfile -Command \"$b=[IO.File]::ReadAllText('{tmp}'); [IO.File]::WriteAllBytes('{winpath}',[Convert]::FromBase64String($b)); Remove-Item '{tmp}'; (Get-FileHash '{winpath}' -Algorithm SHA256).Hash\""
    return v.call_command(ps, run_timeout_s=60, intent='Decode file on VPS').stdout.strip()
def act(msg):
    ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line=f"- {ts} [warm-monitor, <agent>] {msg}".replace("'", "'\\''")
    shr(c, f"echo '{line}' >> {ACT}", t=40, tries=3)
def load_events():
    o=shr(c, f"cat {EVF}", t=40, tries=3)
    try: return json.loads(o)
    except Exception: return dict(decisions=[], events=[], seen=[])
def save_events(E): put(EVF, json.dumps(E, indent=1), mode='644')
def add_event(txt, E=None):
    E=E or load_events(); E['events'].append(txt); save_events(E); return E
def poll(write=True, verbose=False):
    out={}
    out['clusterA']=json.loads(shr(c, f"cd {MC} && python3 mon_state.py", intent='Poll CLUSTER-A arm state'))
    out['ctl']=shr(c, f"grep -h -i 'GATE_FAIL\\|RACE_LOST\\|requeue\\|restart' {X}/admin/controller_log.md {X}/admin/arm_log.md | tail -4 | cut -c1-240; tail -2 {X}/admin/cancellation_log.md | cut -c1-200", intent='Controller log tail')
    out['vps']='(VPS not polled: off by <author> decision 4 Nov)'
    out['clusterB']=shr(d, f"squeue -u <author> -h -t R -o '%i|%j|%T|%M|%R' | grep 'fx_\\|fxm' ; ls {N}/exp/FMD/xs/runs | grep -v CANCELLED | head -20; tail -2 {N}/exp/FMD/xs/admin/arm_log_clusterB.md | cut -c1-220", intent='Poll CLUSTER-B', tries=2)
    out['fxm']=shr(c, "squeue -u <user> -h -o '%i|%j|%T|%M|%r|%N' | grep 'fxm_' | head -30; for f in "+MR+"/comparators_a4000/*/job_history.txt; do echo $(basename $(dirname $f)) $(tail -1 $f | cut -c1-110); done 2>/dev/null", intent='Poll fxm jobs', tries=2)
    st=out['clusterA']; out['utc']=st['utc']; E=load_events(); out['E']=E
    L=[f"# WARM_MONITOR_STATUS (facts only) - poll {st['utc']}", "Track: FMDX warm monitor (<agent>; reason: monitoring with existing recipes). Prereg sha256 b5af7ee97706d4a7c8d5c880c7612b769712e9d1a41de4808ee4158e95fd04c8 (verified 2026-11-03T17:35Z).", "",
       "## CLUSTER-A warm arms (priority order; non-cancelled)","| rank | arm | queue | slurm | step | sec/step | train_complete | ms | val preds | best ms |","|---|---|---|---|---|---|---|---|---|---|"]
    for a in st['arms']:
        if a['rank']>40 and not a.get('run_dir'): continue
        sl=a.get('slurm') or {}
        L.append(f"| {a['rank']} | {a['arm']} | {a['queue_state']} | {sl.get('job','')} {sl.get('state','')} {sl.get('reason','')[:20]} | {a.get('step','')} | {a.get('sec_per_step','')} | {a.get('training_complete','')} | {a.get('n_ms','')} | {a.get('n_val_preds','')} | {a.get('best_milestone','')} |")
    L+=["","## CLUSTER-A monitor jobs (fxm_*) and comparator job history","```",out['fxm'].strip(),"```","","## VPS","```",out['vps'].replace('\r','').strip(),"```","","## CLUSTER-B","```",out['clusterB'].strip(),"```","","## Controller/race log tail","```",out['ctl'].strip(),"```",
        "","## Decisions recorded (lead directives; UTC)"]+[f"- {t}: {x}" for t,x in E['decisions']]+["","## Events (facts)"]+["- "+e for e in E['events']]
    out['md']="\n".join(L)
    if write: put(f"{X}/runs/WARM_MONITOR_STATUS.md", out['md'], mode='644')
    return out
def tick(write=True):
    P=poll(write=write); E=P['E']; ev=[]
    for a in P['clusterA']['arms']:
        if a.get('training_complete') and a.get('best_milestone') and a['arm'] not in E['seen']: ev.append(('clusterA_done', a['arm'], a['best_milestone']))
        if a.get('gate_fail') or a.get('race_lost'): ev.append(('flag', a['arm'], 'gate_fail' if a.get('gate_fail') else 'race_lost'))
    run=[(a['rank'],a['arm'],a.get('step'),a.get('sec_per_step') and round(a['sec_per_step'],2),(a.get('slurm') or {}).get('state')) for a in P['clusterA']['arms'] if (a.get('slurm') or {}).get('state')=='RUNNING' or a.get('step')]
    print(P['utc'], run); print('FXM:', P['fxm'].replace('\n',' || ')[:500]); print('VPS:', P['vps'].replace('\r','').replace('\n',' || ')[:420]); print('CLUSTER-B:', P['clusterB'].replace('\n',' || ')[:300]); print('events:', ev)
    return P, ev
POOL=[411,412,413,414,415,416,417,418,419,421,422,423,424,425,426,427,428,429,431,432,433,434,435,436,437,438,439]
PRIO_ORDER=['NOFM_W','PEINIT_W','c-radiov3-h_W','c-radiov3-g_W','c-radiov4-h_W','c-radiov3-l_W','c-radiov4-so400m_W','c-radiov3-b_W','dinov3-vitb16_W','dinov3-vitl16_W']
CMDNUM={a:(POOL[2*i],POOL[2*i+1]) for i,a in enumerate(PRIO_ORDER)}
SETS6 = {'test_casia_zh': ('exp/X4a/data/x4a_casia_zh.npz','test_test_casia_zh'),'test_casia_en': ('exp/X4a/data/x4a_casia_en.npz','test_test_casia_en'),
 'test_hpl_tamil': ('exp/X4a/data/x4a_hpl_tamil.npz','test_test_hpl_tamil'),'test_iam500': ('data/m5_multiscript_v1/m5_test_iam500.npz','test_test_iam500'),
 'german_unipen_de_test': ('data/d2_v2/unipen_de/e0_test.npz','test_german_unipen_de_test'),'german_onhw_all': ('data/d2_v2/onhw/e0_all.npz','test_german_onhw_all')}
def sets_spec(sub=None):
    return [dict(name=k, gt_rel=g, render_rel=f"exp/FMD/xs/data/monitor_v1/{r}_render224.npz", subset=sub) for k,(g,r) in SETS6.items()]
GTMAP=",".join(f"{k}={R}/{g}" for k,(g,_) in SETS6.items())
GTMAP_FT=f"test_ouhdl={R}/data/ouhdl_e0_v3/e0_test.npz,val1000_ouhdl={R}/data/ouhdl_e0_v3/e0_val.npz"
def bg(name, script, wait_marker=None, timeout=900):
    shr(c, f"cd {MR}; cat > {name}.sh <<'E'\n#!/bin/bash\n{script}\nE\nchmod +x {name}.sh; setsid nohup ./{name}.sh > {MR}/logs/{name}.log 2>&1 < /dev/null & echo started", t=60)
    t0=time.time()
    while wait_marker and time.time()-t0<timeout:
        time.sleep(20); o=shr(c, f"tail -5 {MR}/logs/{name}.log", t=60, tries=3)
        if wait_marker in o: return o
    return shr(c, f"tail -5 {MR}/logs/{name}.log", t=60)
def push_ckpt(rel, name):
    return bg(f"push_{name}", f"cd {R} && bash exp/FMD/xs/code/fmdx_backbone_v1/jobs/hub_push.sh fxm_{name} {rel}", wait_marker='pushed fxm_', timeout=900)
DISP=r'$VPS_ROOT\uhwr\stage\exp\FMD\xs\runs\fmdx_monitor\dispatch'
def ready_vps(arm, kind, spec):
    """put a sampling cmd into the dispatcher's ready\ dir (name order = priority.json order, bb before ft); written as .tmp then renamed"""
    idx=PRIO_ORDER.index(arm)*2+(0 if kind=='bb' else 1); num=900+idx; name=f"{idx:02d}_{arm}_{kind}"
    fn=f'handoff/spec_{num}.json'; json.dump(spec, open(fn,'w'))
    cb=subprocess.run([sys.executable,'fmdx_monitor_v1/vps_sample_cmd.py',fn,str(num)],capture_output=True).stdout
    tmp=rf'{DISP}\ready\{name}.tmp'; h=vput(tmp, cb)
    v.call_command(rf'cmd /c move /Y {tmp} {DISP}\ready\{name}.cmd', run_timeout_s=30, intent='Rename dispatcher job into ready')
    act(f"dispatch ready job {name}.cmd (sha256 {h.lower()}), tag {spec['tag']}, ckpt {spec['ckpt_rel']}, out {spec['out_rel']}")
    return h
def spec_bb(arm, lane, ms):
    return dict(tag=f"{arm}_bb_ms{ms}", ckpt_rel=f"exp/FMD/xs/runs/{arm}_{lane}/bb_ms{ms}.pt", out_rel=f"exp/FMD/xs/runs/fmdx_monitor/preds/{arm}/bb", seeds=[0,1,2], sets=sets_spec(None))
def spec_ft(arm, lane, ms):
    sets=[dict(name='test_ouhdl', gt_rel='data/ouhdl_e0_v3/e0_test.npz', render_rel='exp/FMD/data/fmd_render224_test.npz', subset=None),
          dict(name='val1000_ouhdl', gt_rel='data/ouhdl_e0_v3/e0_val.npz', render_rel='exp/FMD/data/fmd_render224_val1000.npz', subset=1000)]+sets_spec(None)
    return dict(tag=f"{arm}_ft_ms{ms}", ckpt_rel=f"exp/FMD/xs/runs/fmdx_monitor/ft/{arm}_FT_{lane}/bb_ms{ms}.pt", out_rel=f"exp/FMD/xs/runs/fmdx_monitor/preds/{arm}/ft", seeds=[0,1,2], sets=sets)
def launch_ft_clusterA(arm, src, gres_list=('a40','h100')):
    ids=[]
    for g in gres_list:
        o=shr(c, f"cd {MR} && sbatch --parsable -J fxm_ft_{arm} -p gpuidle --gres=gpu:{g}:1 {MC}/ft_clusterA.sbatch {arm} {src}", t=90); ids.append((g,o.strip()))
    act(f"FT launched for {arm} from {src}: CLUSTER-A copies {ids} (race lock race\\{arm}_FT on VPS hub; first to start wins)")
    return ids

def submit_arm_sample(arm, kind, lane, ms, nice=1000):
    o=shr(c, f"cd {MR} && sbatch --parsable -J fxm_as_{arm}_{kind} --nice={nice} --array=0-2 {MC}/arm_sample.sbatch {arm} {kind} {lane} {ms}", t=90).strip()
    act(f"arm sampling submitted on A4000 (prereg A8.1): {arm} {kind} lane={lane} ms={ms} array 0-2 (seed=task id) nice={nice} job {o}")
    return o

PYCPU=f"apptainer exec -B $SCRATCH_ROOT {R}/images/torch27.sif {R}/venvs/cpu/bin/python"
def score_arm(arm, kind, sets=None):
    """adapt sidecars of MR/preds/<arm>/<kind> (kind bb|ft|cmp) into MR/score_in/<arm>_<kind>, submit lane_score array (one task per set glob) on cpuidle"""
    sets = sets or (list(SETS6) + (['test_ouhdl','val1000_ouhdl'] if kind=='ft' else []))
    src=f"{MR}/preds/{arm}/{kind}"; dst=f"{MR}/score_in/{arm}_{kind}"
    gm=GTMAP+(","+GTMAP_FT if kind=='ft' else "")
    o=shr(c, f"rm -rf {dst}; {PYCPU} {MC}/adapt_sidecars.py --src {src} --dst {dst} --gt-map '{gm}'", t=120)
    tasks="\n".join(f"{st}__{arm}_{kind}_ms*_preds.npz" for st in sets)
    shr(c, f"printf '{tasks}\\n' > {MR}/score_in/{arm}_{kind}.tasks", t=40)
    n=len(sets)
    j=shr(c, f"cd {MR} && PREDS={dst} OUT={MR}/score_out TASKS={MR}/score_in/{arm}_{kind}.tasks sbatch --parsable -p cpuidle --requeue -J fxm_sc_{arm}_{kind} --array=1-{n} --export=ALL,PREDS,OUT,TASKS {MC}/lane_score_array_mon.sh", t=90).strip()
    act(f"scoring submitted {arm} {kind}: adapt={o.strip()[-30:]} job {j} ({n} set tasks) -> {MR}/score_out/rows")
    return j
def compose_arm(arm):
    o=shr(c, f"mkdir -p {MR}/preds/{arm}/cmp; {PYCPU} {MC}/compose_preds.py --arm {arm} --bb-dir {MR}/preds/{arm}/bb --ft-dir {MR}/preds/{arm}/ft --out {MR}/preds/{arm}/cmp | tail -3", t=170)
    act(f"composed preds A4.4 built for {arm}: {o.strip()[-120:]}")
    return o
