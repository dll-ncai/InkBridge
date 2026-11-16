def earliest(opts, script, t='06:00:00', mem='24G', cpus=4):
    """opts [(part, gres)] -> 3 with earliest sbatch --test-only start"""
    cmd="; ".join(f"echo \"{p} {g} $(sbatch --test-only -p {p} --gres={g} -c {cpus} --mem={mem} -t {t} --exclude=<node> --wrap true 2>&1 | grep -o 'start at [0-9T:-]*' | cut -d' ' -f3)\"" for p,g in opts)
    o=shr(c, cmd+"; echo .", t=120); res=[]
    for l in o.splitlines():
        q=l.split()
        if len(q)==3 and _re.match(r'\d{4}-', q[2]): res.append((q[2],q[0],q[1]))
    res.sort(); return [(p,g) for _,p,g in res[:3]], res

def submit_a9(arm, kind, lane, ms, arr='0-2'):
    pick,res=earliest(A9OPTS, 'a9'); ids=[]
    for p,g in pick:
        o=shr(c, f"cd {MR} && sbatch --parsable -J fxm_as9_{arm}_{kind} -p {p} --gres={g} --array={arr} {MC}/arm_sample_a9.sbatch {arm} {kind} {lane} {ms} 2>&1 | tail -1", t=90).strip(); ids.append(f"{p}:{g}={o}")
    act(f"A9 sampling race copies (3 earliest test-only: {res[:3]}) {arm} {kind} lane={lane} ms={ms} array={arr}: {ids}")
    return ids

def launch_ft(arm, ms):
    src=f"{X}/runs/{arm}_clusterA/bb_ms{ms}.pt"; pick,res=earliest(FTC,'ft',t='2-00:00:00',mem='48G',cpus=8); ids=[]
    for p,g in pick:
        o=shr(c, f"cd {MR} && sbatch --parsable -J fxm_ft_{arm} -p {p} --gres={g} {MC}/ft_clusterA.sbatch {arm} {src} 2>&1 | tail -1", t=90).strip()
        if o.isdigit(): ids.append(o)
    act(f"FT race copies {arm} from {src} (3 earliest test-only: {res[:3]}): {list(zip([p+':'+g for p,g in pick], ids))}")
    return ids

def watch_all():
    o=shr(c, f"cd {X}/runs; for d in *_clusterA; do [ -f $d/training_complete.json ] && [ -f $d/eval/best_milestone.txt ] && echo DONE ${{d%_clusterA}} $(cat $d/eval/best_milestone.txt); done; ls {X}/runs/*_clusterB/relay/RELAY.json 2>/dev/null | sed 's/^/RELAY /'; echo END", t=60)
    acts=[]; relays=[]
    for l in o.splitlines():
        p_=l.split()
        if len(p_)==3 and p_[0]=='DONE' and p_[1] not in S['arms'] and p_[1] not in SKIP and p_[2].isdigit():
            a,ms=p_[1],int(p_[2]); st=S['arms'].setdefault(a, dict(bb_ms=ms, ft_ms=None))
            st['ft_copies']=launch_ft(a, ms)
            bb=submit_a9(a,'bb','clusterA',ms); st.setdefault('a9',{})['bb']=[x.split('=')[1] for x in bb if x.split('=')[1].isdigit()]
            st['rows']=dict(bb=0,ft=0,cmp=0); acts.append(f"{a} training complete (CLUSTER-A), best ms{ms}; FT copies {st['ft_copies']}; A9 bb sampling {st['a9']['bb']}")
        if len(p_)==2 and p_[0]=='RELAY': relays.append(p_[1])
    new_rel=[r for r in relays if r not in S.setdefault('relays_seen',[])]
    if new_rel: S['relays_seen']+=new_rel; acts.append("RELAY.json seen: "+", ".join(new_rel))
    if acts:
        for x in acts: act(x)
        add_event(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())+" "+"; ".join(acts)); sv(S)
    return acts, new_rel
