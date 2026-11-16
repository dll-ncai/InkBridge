
import json
X="$U/exp/FMD/xs/weights/"
want = [("c-radiov3-b",["*.py","*.json","*.safetensors","README.md"]),("c-radiov3-l",["*.py","*.json","*.safetensors","README.md"]),("c-radiov3-g",["*.py","*.json","*.safetensors","README.md"]),("c-radiov4-h",["*.py","*.json","*.safetensors","README.md"]),
        ("pe-spatial-l14-448-timm",["*.safetensors","*.json","*.md"]),("pe-spatial-g14-448-timm",["*.safetensors","*.json","*.md"]),("siglip2-giant-opt-patch16-256",None),("cleandift",["cleandift_sd21_unet.safetensors","README.md"])]
out=[]
for n,allow in want:
    p=json.load(open(X+n+".manifest_part.json"))
    out.append(dict(name=n, repo=p["repo_id"], revision=p["revision"], local_dir="exp/FMD/xs/weights/"+n, allow_patterns=allow, files={f["path"]:f["sha256"] for f in p["files"]}))
v1=json.load(open("$U/exp/FMD/weights/MANIFEST.json"))
m=next(m for m in v1["models"] if m.get("local_name")=="sd2.1")
out.append(dict(name="sd2.1", repo=m["repo_id"], revision=m["revision"], local_dir="exp/FMD/weights/sd2.1", allow_patterns=None, files={f["path"]:f["sha256"] for f in m["files"]}))
json.dump(out, open("$U/exp/FMD/xs/admin/clusterB_expected.json","w"), indent=1)
print([(o["name"],len(o["files"])) for o in out])
