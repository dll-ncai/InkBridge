
import torch, json, collections, hashlib
p="$U/exp/FMD/xs/weights/sam3.1/sam3.1_multiplex.pt"
ck=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
print(type(ck), list(ck)[:5] if isinstance(ck,dict) else "")
sd=ck.get("model",ck) if isinstance(ck,dict) else ck
keys=list(sd); print(len(keys))
pre=collections.Counter(".".join(k.split(".")[:3]) for k in keys)
out={"n_keys":len(keys),"top_prefix_counts":dict(pre.most_common(25))}
vk=[k for k in keys if "vision" in k.lower() or "backbone" in k.lower()]
out["vision_like_keys_n"]=len(vk); out["vision_samples"]=[(k,list(sd[k].shape)) for k in vk[:12]]
pe=[k for k in keys if "pos_embed" in k or "patch_embed" in k or "ln_pre" in k][:8]
out["pe_keys"]=[(k,list(sd[k].shape)) for k in pe]
json.dump(out,open("$U/exp/FMD/xs/admin/sam3.1_ckpt_keys.json","w"),indent=1)
print(json.dumps(out)[:3000])
