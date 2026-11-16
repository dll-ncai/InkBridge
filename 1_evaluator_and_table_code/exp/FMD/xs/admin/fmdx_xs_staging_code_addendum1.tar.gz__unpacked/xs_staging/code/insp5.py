
import json, struct, collections, sys
W="$U/exp/FMD/xs/weights/"
for name in ["qaari-0.1-urdu-ocr","urdu-ocr-deepseek-nomy","urdu-ocr-deepseek-ocr2-kingabzpro"]:
    f=W+name+"/adapter_model.safetensors"
    with open(f,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; h=json.loads(fh.read(n))
    ks=[k for k in h if k!="__metadata__"]
    c=collections.Counter()
    for k in ks:
        parts=k.split(".")
        key="visual" if "visual" in k else ("sam" if "sam_model" in k else ("vision_model" if "vision_model" in k else ("projector" if "projector" in k else ("layers" if "layers" in k else "other"))))
        c[key]+=1
    print(name,len(ks),dict(c)); print("  ex:",ks[0],h[ks[0]]["shape"],"|",[k for k in ks if "visual" in k or "sam_model" in k or "vision_model" in k][:2])
    cfg=json.load(open(W+name+"/adapter_config.json")); print("  r",cfg.get("r"),"alpha",cfg.get("lora_alpha"),"target",str(cfg.get("target_modules"))[:200], "base", cfg.get("base_model_name_or_path"))
