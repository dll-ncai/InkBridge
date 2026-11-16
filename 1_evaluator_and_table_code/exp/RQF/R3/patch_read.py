
import json,hashlib,glob
R='$U/exp/RQF/'
f=R+'code/rq3_v1/rq3_read.py'; t=open(f).read()
if 'read_C3' not in t:
    t=t.replace("    fb='COMP_p01' if","    if u['base']=='COMP_p01_C34' and 'read_C3' not in out: continue   # interim 2-seed read (read_C) excludes C34; read_C3 = all 3 training seeds\n    fb='COMP_p01' if",1); open(f,'w').write(t)
print('patched','read_C3' in t)
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest(); bad=0
for p in glob.glob(R+'runs/R3/C34_4090/preds/*.npz'):
    sd=json.load(open(p[:-4]+'.json')); bad+= sd.get('preds_sha256') not in (None,h(p))
print('sidecar sha mismatches',bad)
