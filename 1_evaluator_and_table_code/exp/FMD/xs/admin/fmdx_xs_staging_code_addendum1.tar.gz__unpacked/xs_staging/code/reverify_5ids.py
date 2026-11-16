import json,hashlib,os
W="$U/exp/FMD/xs/weights"; out={}
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<24),b""): h.update(b)
    return h.hexdigest()
for n in ["sam3","sam3.1","aimv2-large-patch14-224-distilled","aimv2-3b-patch14-224","pe-core-g14-448","pe-core-g14-448-timm"]:
    pj=json.load(open(f"{W}/{n}.manifest_part.json")); bad=[]
    for f in pj["files"]:
        if sha(f"{W}/{n}/"+f["path"])!=f["sha256"]: bad.append(f["path"])
    out[n]=dict(n_files=len(pj["files"]),n_sha_mismatch=len(bad),mismatch=bad); print(n,out[n],flush=True)
json.dump(out,open("$U/exp/FMD/xs/admin/reverify_5ids.json","w"),indent=1)
