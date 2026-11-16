
import os, json, sys
os.environ["HF_HOME"]="$U/exp/FMD/xs/weights/.hf_home"
from huggingface_hub import HfApi
api=HfApi()
cands = """CompVis/cleandift
CompVis/CleanDIFT
nvidia/C-RADIOv3-B
nvidia/C-RADIOv3-L
nvidia/C-RADIOv3-H
nvidia/C-RADIOv3-g
nvidia/C-RADIOv4-B
nvidia/C-RADIOv4-L
nvidia/C-RADIOv4-H
nvidia/C-RADIOv4-g
nvidia/C-RADIOv4-SO400M
nvidia/C-RADIOv4-H
facebook/PE-Spatial-G14-448
facebook/PE-Spatial-L14-448
facebook/PE-Spatial-B16-512
facebook/PE-Spatial-S16-512
facebook/PE-Spatial-T16-512
facebook/PE-Core-G14-448
facebook/PE-Core-L14-336
stabilityai/stable-diffusion-3-medium
stabilityai/stable-diffusion-3-medium-diffusers
stabilityai/stable-diffusion-3.5-medium
black-forest-labs/FLUX.1-dev
black-forest-labs/FLUX.1-schnell
starvector/starvector-1b-im2svg
starvector/starvector-8b-im2svg
oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct
Qwen/Qwen2-VL-2B-Instruct
deepseek-ai/DeepSeek-OCR
deepseek-ai/DeepSeek-OCR-2
google/siglip2-giant-opt-patch16-256
google/siglip2-giant-opt-patch16-384
facebook/dinov3-convnext-tiny-pretrain-lvd1689m
facebook/dinov3-convnext-small-pretrain-lvd1689m
facebook/dinov3-convnext-base-pretrain-lvd1689m
facebook/dinov3-convnext-large-pretrain-lvd1689m
facebook/dinov3-vits16-pretrain-lvd1689m
facebook/dinov3-vitb16-pretrain-lvd1689m
facebook/dinov3-vith16plus-pretrain-lvd1689m
facebook/dinov3-vit7b16-pretrain-lvd1689m
google/inkfm
google/ink-fm
Derendering/InkSight-Small-p
Derendering/InkSight-Base-p""".split("\n")
out=[]
for r in dict.fromkeys(cands):
    try:
        i=api.model_info(r, files_metadata=True)
        tot=sum((s.size or 0) for s in i.siblings)
        out.append(dict(repo=r, ok=True, sha=i.sha, gated=str(i.gated), tags=[t for t in i.tags if t.startswith("license")], lic=(i.card_data.license if i.card_data else None), size=tot, nfiles=len(i.siblings), params=(i.safetensors.total if i.safetensors else None), created=str(i.created_at)[:10]))
    except Exception as e:
        out.append(dict(repo=r, ok=False, err=type(e).__name__+": "+str(e)[:200].replace("\n"," ")))
for o in out: print(json.dumps(o))
