
import json, struct, glob, collections
tot=collections.Counter(); ex={}
for f in glob.glob("$U/exp/FMD/xs/weights/starvector-1b-im2svg/*.safetensors"):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
    for k,v in h.items():
        if k=="__metadata__": continue
        p=".".join(k.split(".")[:2]); tot[p]+=1; ex.setdefault(p,(k,v["shape"]))
for p,c in tot.items(): print(p,c,ex[p])
