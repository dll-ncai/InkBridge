
import json, time, os
SF=MR+'/state/warm_state.json'
def ld():
    try: return json.loads(shr(c, f'cat {SF}', t=40))
    except Exception: return {}
def sv(S):
    put(SF, json.dumps(S, indent=1), mode='644')
S=ld()
S.setdefault('arms', {
 'NOFM_W': dict(bb_ms=3, ft_ms=1, ft_sample='24900857', bb_scored=True),
 'c-radiov3-h_W': dict(bb_ms=4, ft_ms=4, ft_sample='24883662', bb_scored=True),
 'PEINIT_W': dict(bb_ms=4, ft_ms=None, ft_copies=['24879671','24878859','24882900','24882901'])})
S.setdefault('tables_built_for', ''); S.setdefault('last_status', 0)
def counts():
    cmd=f"cd {MR}; for a in " + " ".join(S['arms']) + "; do for k in bb ft cmp; do echo $a $k $(ls preds/$a/$k/*_preds.npz 2>/dev/null | wc -l) $(ls score_out/rows/*__${a}_${k}_ms*__rows031.csv 2>/dev/null | wc -l); done; echo $a best $(cat ft/${a}_FT_clusterA/eval/best_milestone.txt 2>/dev/null || echo none); done; squeue -u <user> -h -o '%i %T' | grep -c RUNNING"
    o=shr(c, cmd, t=90); C={}
    for l in o.splitlines():
        p=l.split()
        if len(p)==4: C[(p[0],p[1])]=(int(p[2]),int(p[3]))
        elif len(p)==3 and p[1]=='best': C[(p[0],'best')]=p[2]
    return C
NEED={'bb':18,'ft':24,'cmp':18}
def step():
    C=counts(); acts=[]
    for a,st in S['arms'].items():
        if st.get('ft_ms') is None and C.get((a,'best'),'none')!='none':
            st['ft_ms']=int(C[(a,'best')]); cps=[x for x in st.get('ft_copies',[])]
            shr(c, "scancel "+" ".join(cps), t=60)
            ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            l=f"- {ts} [FMDX warm-monitor, <agent>] {a} FT complete (val1000 best ms{st['ft_ms']}); scancel issued for remaining copies {cps} (winner = job in ft/{a}_FT_clusterA/job_history.txt)"
            shr(c, f"echo '{l}' >> {R}/exp/CAND/queue/cand_log.md; echo '{l}' >> {X}/admin/cancellation_log.md", t=40)
            st['ft_sample']=submit_a9(a,'ft','clusterA',st['ft_ms']); st.setdefault('a9',{})['ft']=[x.split('=')[1] for x in st['ft_sample'] if x.split('=')[1].isdigit()]; acts.append(f"{a} FT done ms{st['ft_ms']}; FT sampling {st['ft_sample']}")
        for k in ('bb','ft','cmp'):
            n,nr=C.get((a,k),(0,0))
            if n>=NEED[k] and nr<NEED[k] and not st.get(k+'_scored'):
                st[k+'_scored']=score_arm(a,k); acts.append(f"{a} {k} scoring {st[k+'_scored']}")
        if C.get((a,'bb'),(0,0))[0]>=18 and C.get((a,'ft'),(0,0))[0]>=24 and not st.get('composed'):
            st['composed']=compose_arm(a)[-100:]; acts.append(f"{a} composed: {st['composed'][-60:]}")
        st['rows']={k:C.get((a,k),(0,0))[1] for k in ('bb','ft','cmp')}
    for a,st in S['arms'].items():
        for k,ids in list(st.get('a9',{}).items()):
            nl=shr(c, f"ls -d {MR}/preds/{a}/{k}/.lock_* 2>/dev/null | wc -l", t=40).strip()
            if nl=='3':
                ids=[i for i in ids if str(i).isdigit()]
                pend=shr(c, "squeue -h -t PD -r -o %i -j "+",".join(ids)+" 2>/dev/null | sort -u | tr '\\n' ' '; echo", t=40).strip()
                if pend:
                    shr(c, f"scancel {pend}", t=40); ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    l=f"- {ts} [FMDX warm-monitor, <agent>] A9 sampling race {a} {k}: all 3 seed locks taken (owners in preds/{a}/{k}/.lock_*/owner); cancelled pending copies {pend}"
                    shr(c, f"echo '{l}' >> {R}/exp/CAND/queue/cand_log.md; echo '{l}' >> {X}/admin/cancellation_log.md", t=40); acts.append(f"{a} {k} A9 race decided; cancelled {pend}")
                del st['a9'][k]
    full=sorted(f"{a}:{k}" for a,st in S['arms'].items() for k in ('bb','ft','cmp') if st['rows'][k]>=NEED[k])
    key=",".join(full)
    if key and key!=S['tables_built_for']:
        busy=shr(c,"squeue -u <user> -h -n fxm_tables | wc -l",t=40).strip()
        if busy=='0':
            j=shr(c, f"sbatch --parsable {MC}/tables_job_a9.sbatch 'auto {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}: {key}'", t=60).strip()
            S['tables_built_for']=key; S['tables_job']=j; acts.append(f"tables job {j} for {key}")
    if acts:
        ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        for x in acts: act(x)
        add_event(f"{ts} "+"; ".join(acts))
    if time.time()-S['last_status']>3300:
        poll(write=True); S['last_status']=time.time()
    sv(S); return acts
