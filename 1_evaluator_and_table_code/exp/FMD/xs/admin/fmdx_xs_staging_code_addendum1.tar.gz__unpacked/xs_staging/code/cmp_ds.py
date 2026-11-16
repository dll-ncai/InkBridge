
import json
W="$U/exp/FMD/xs/weights/"
a=json.load(open(W+"deepseek-ocr.manifest_part.json")); b=json.load(open(W+"unsloth-deepseek-ocr.manifest_part.json"))
fa={f["path"]:f["sha256"] for f in a["files"]}; fb={f["path"]:f["sha256"] for f in b["files"]}
for p in sorted(set(fa)|set(fb)):
    if p.endswith(".safetensors") or p.endswith(".json"): print(p, "same" if fa.get(p)==fb.get(p) else ("DIFF" if p in fa and p in fb else "only-"+("official" if p in fa else "unsloth")))
