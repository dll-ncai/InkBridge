#!/usr/bin/env python3
"""RQF R3 step: apply frozen verifier (sealed_apply.py, verify.py 579c9b22, k=3) + random-of-3 control per unit, then stage scoring dir.
Usage: rq3_apply.py GROUP (A|C) [--only-train IDX,...]. Writes R3/rep/<base>/<set>/..., R3/score_in_<GROUP>/ (symlinks + normalised sidecars + tasks.txt)."""
import json,os,sys,subprocess,hashlib,glob
U='$U'; Q=U+'/exp/RQF/R3'; C=U+'/exp/RQF/code/rq3_v1'
VS='579c9b220434c3d80ce23ddb90923addf6081fd70acc9ae9461b1b8086de2e22'
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
g=sys.argv[1]; P=json.load(open(Q+'/plan.json')); S=Q+'/score_in_'+g; os.makedirs(S,exist_ok=True); tasks=[]; gsha={}
for u in P['units']:
    if u['group']!=g: continue
    if os.environ.get('RQ3_TRAIN') and str(u['train_idx']) not in os.environ['RQ3_TRAIN'].split(','): continue
    assert all(os.path.exists(p) for p in u['pool']),('missing pool',u['base'],u['set'])
    out=f"{Q}/rep/{u['base']}/{u['set']}"; os.makedirs(out,exist_ok=True)
    r=subprocess.run([sys.executable,C+'/sealed_apply.py','--verify',C+'/verify.py','--verify-sha',VS,'--pool',*u['pool'],'--inputs',u['gt'],'--set-index',str(u['set_index']),'--train-index',str(u['train_idx']),'--out-dir',out],capture_output=True,text=True)
    print(r.stdout.strip()[-200:],r.stderr.strip()[-300:],flush=True); assert r.returncode==0
    stem0=os.path.basename(u['pool'][0])[:-len('__seed0_preds.npz')]
    stage=[(out+f'/{stem0}_{t}__seed0_preds.npz') for t in ('rep3','rnd3')]
    if any(x is None for x in u['rows']): stage+=u['pool']   # base files without existing rows031 are scored too
    gt=u['gt']; gsha.setdefault(gt,h(gt))
    for f in stage:
        b=os.path.basename(f); m=b.split('__'); side=json.load(open(f[:-4]+'.json')); d=S+'/'+b
        if os.path.lexists(d): os.remove(d)
        os.symlink(os.path.realpath(f),d)
        json.dump(dict(set=u['set'],npz_clusterA=gt,source_sha256_expected=gsha[gt],route_mode=m[1],seed=int(b.split('__seed')[1][0]),preds_sha256=h(f),label=m[1],orig=side),open(d[:-4]+'.json','w'),indent=1)
        tasks.append(b)
open(S+'/tasks.txt','w').write('\n'.join(tasks)+'\n'); print('staged',len(tasks))
