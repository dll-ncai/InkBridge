#!/usr/bin/env python3
# Stage 2 of SPEC_reviewer_archive_v11: column subset of scored per-unit rows (native protocol), no averaging, no rounding.
import os,sys,hashlib,json,time,collections
import numpy as np, pandas as pd
from multiprocessing import Pool
U='$U'; O=U+'/tmp/rv11_archive/stage2b'; os.makedirs(O,exist_ok=True)
MET=["aiou","penup_f1_v2","strict_order_acc","direction_acc","kendall_tau","ldtw"]
CONST=["route_mode","specialist","evaluator","uhwr_eval_file","preds_file","preds_sha256","gt_npz","gt_sha256"]
files=[l.rstrip('\n').split('\t') for l in open(U+'/tmp/rv11_archive/stage2b_files.tsv')]
def work(a):
    i,(d,n)=a; p=U+'/'+d+'/'+n
    try:
        raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest()
        import io
        df=pd.read_csv(io.BytesIO(raw),dtype={'unit_id':str,'writer_id':str,'sample_id':str,'error':str,'protocol':str},float_precision='round_trip',low_memory=False)
        nall=len(df); prot=df.protocol.value_counts().to_dict()
        cons={c:sorted(set(df[c].dropna().astype(str)))[:6] for c in CONST if c in df.columns}
        x=df[df.protocol=='native']
        fseed=int(n.split('__seed')[1].split('__')[0]); seedok=bool((x.seed==fseed).all())
        err=x.error.notna()&(x.error.astype(str).str.len()>0)
        rec=dict(i=i,dir=d,name=n,sha256=h,bytes=len(raw),mtime=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(os.path.getmtime(p))),n_rows=nall,protocols=prot,n_native=len(x),n_error=int(err.sum()),seed=fseed,seed_col_matches=seedok,const=cons,dup_units=int(x.unit_id.duplicated().sum()))
        return rec,x.unit_id.to_numpy(),x.writer_id.to_numpy(),err.to_numpy(),x[MET].to_numpy(np.float64)
    except Exception as e:
        return dict(i=i,dir=d,name=n,error=repr(e)),None,None,None,None
if __name__=='__main__':
    t0=time.time(); idx=[]; units={}; acc=collections.defaultdict(list); wconf=0
    with Pool(8) as pool:
        for rec,uid,wid,err,M in pool.imap(work,list(enumerate(files)),chunksize=4):
            idx.append(rec)
            if uid is None: continue
            s=rec['name'].split('__')[0]; um=units.setdefault(s,{})
            ui=np.empty(len(uid),np.int32)
            for k,(u,w) in enumerate(zip(uid,wid)):
                e=um.get(u)
                if e is None: e=um[u]=(len(um),w)
                elif e[1]!=w: wconf+=1
                ui[k]=e[0]
            key=(rec['dir'].replace('/','__'),s)
            acc[key].append((np.full(len(uid),rec['i'],np.int32),ui,err,M))
    for (dk,s),parts in acc.items():
        np.savez_compressed('%s/rows__%s__%s.npz'%(O,dk,s),file_idx=np.concatenate([p[0] for p in parts]),unit_idx=np.concatenate([p[1] for p in parts]),error=np.concatenate([p[2] for p in parts]),metrics=np.concatenate([p[3] for p in parts]),metric_names=np.array(MET))
    for s,um in units.items():
        with open('%s/UNITS_%s.tsv'%(O,s),'w') as o:
            o.write('unit_idx\tunit_id\twriter_id\n')
            for u,(k,w) in sorted(um.items(),key=lambda z:z[1][0]): o.write('%d\t%s\t%s\n'%(k,u,w))
    json.dump(dict(files=idx,writer_conflicts=wconf,seconds=time.time()-t0,script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),pandas=pd.__version__,numpy=np.__version__),open(O+'/INDEX.json','w'))
    os.system('cd %s/.. && tar cf stage2b.tar stage2b && split -b 10M -d stage2b.tar stage2b.part. && sha256sum stage2b.tar stage2b.part.* > stage2b.sha256 && ls -l stage2b.tar >> stage2b.sha256'%O)
    print('DONE files',len(idx),'errors',sum('error' in r for r in idx),'writer_conflicts',wconf,'sec',round(time.time()-t0))
