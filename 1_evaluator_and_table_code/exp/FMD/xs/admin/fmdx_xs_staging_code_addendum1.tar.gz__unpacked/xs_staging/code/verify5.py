import json,os,glob
W="$U/exp/FMD/xs/weights"
M=json.load(open(W+"/MANIFEST_v2.json")); idx={m["local_name"]:m for m in M["models"]}
for n in ["sam3","sam3.1","aimv2-large-patch14-224-distilled","aimv2-3b-patch14-224","pe-core-g14-448"]:
    d=W+"/"+n; pj=json.load(open(d+".manifest_part.json")); m=idx.get(n)
    print("##",n,"| dir exists",os.path.isdir(d),"| repo",pj["repo_id"],"| rev",pj["revision"],"| part status",pj["status"],"| manifest entry",bool(m),"| mtime part",os.path.getmtime(d+".manifest_part.json"))
    mism=[];tot=0
    for f in pj["files"]:
        p=d+"/"+f["path"]; ok=os.path.exists(p) and os.path.getsize(p)==f["bytes"]; tot+=f["bytes"]
        if not ok: mism.append(f["path"])
    big=sorted(pj["files"],key=lambda f:-f["bytes"])[:2]
    print("   n_files",len(pj["files"]),"bytes",tot,"size mismatches",mism,"| largest",[(f["path"],f["bytes"],f["sha256"][:12]) for f in big])
    if m: print("   manifest total_bytes",m["total_bytes"],"load_test",[(t["test"],t["status"],t.get("token_map_hw_at_224")) for t in m["load_test"]])
