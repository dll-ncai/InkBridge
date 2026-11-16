#!/usr/bin/env python3
"""RQF R3 read: per (base,set) spec with cells COMP_pool3 (existing/new base rows031, all training x sampling seeds), COMP_rep3, COMP_rnd3 (cell names kept
because sealed_read.py hard-codes the comparator name 'COMP_pool3'; label = base__set). Runs sealed_read.py (A1 ORDER-WIN, Holm over 3 order metrics) unchanged.
Usage: rq3_read.py OUT_DIR BASE[,BASE2..]|ALL [train_idx_max]"""
import json,os,sys,subprocess,glob,collections
U='$U'; Q=U+'/exp/RQF/R3'; C=U+'/exp/RQF/code/rq3_v1'
out=sys.argv[1]; want=sys.argv[2]; P=json.load(open(Q+'/plan.json')); os.makedirs(out,exist_ok=True)
fam=collections.defaultdict(list)
for u in P['units']:
    fb='COMP_p01' if u['base'].startswith('COMP_p01') else u['base']
    if want=='ALL' or fb in want.split(','): fam[(fb,u['set'])].append(u)
def rows_of(f): return glob.glob(f"{Q}/score_out/rows/{os.path.basename(f)[:-len('_preds.npz')]}__rows031.csv")
env=dict(os.environ,ORDER_CODE=C)
for (fb,s),us in sorted(fam.items()):
    cells={n:dict(device=us[0].get('device'),files={}) for n in ('COMP_pool3','COMP_rep3','COMP_rnd3')}
    for u in sorted(us,key=lambda x:x['train_idx']):
        t=u['train_idx']; stem0=os.path.basename(u['pool'][0])[:-len('__seed0_preds.npz')]; rd=f"{Q}/rep/{u['base']}/{u['set']}"
        for j in range(3):
            r=u['rows'][j] or (rows_of(u['pool'][j]) or [None])[0]
            if r is None: print('SKIP missing rows',u['base'],s,j); break
            cells['COMP_pool3']['files'][f'{t}{j}']=r
            for tag in ('rep3','rnd3'):
                rr=rows_of(f'{rd}/{stem0}_{tag}__seed0_preds.npz')
                if rr: cells['COMP_'+tag]['files'][f'{t}{j}']=rr[0]
    if not all(len(c['files'])>0 and len(c['files'])==len(cells['COMP_pool3']['files']) for c in cells.values()): print('INCOMPLETE',fb,s); continue
    lab=f'{fb}__{s}'; sp=f'{out}/spec_{lab}.json'
    json.dump(dict(set=s,cells=cells,contrasts=[['COMP_rep3','COMP_pool3'],['COMP_rnd3','COMP_pool3'],['COMP_rep3','COMP_rnd3']]),open(sp,'w'),indent=1)
    r=subprocess.run([sys.executable,C+'/sealed_read.py','--spec',sp,'--out',out,'--label',lab],capture_output=True,text=True,env=env)
    open(f'{out}/read_stdout_{lab}.txt','w').write(r.stdout+r.stderr); print(lab,'rc',r.returncode,'n_keys',len(cells['COMP_pool3']['files']),flush=True)
