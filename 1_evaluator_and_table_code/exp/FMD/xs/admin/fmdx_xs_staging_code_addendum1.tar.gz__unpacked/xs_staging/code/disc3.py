
import os, json, urllib.request
os.environ["HF_HOME"]="$U/exp/FMD/xs/weights/.hf_home"
from huggingface_hub import HfApi, hf_hub_download
api=HfApi()
for r in ["CompVis/cleandift","oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct","nomypython/urdu-ocr-deepseek","kingabzpro/deepseek-ocr-2-urdu-ocr-1m-lora","stabilityai/stable-diffusion-3-medium-diffusers","black-forest-labs/FLUX.1-dev","black-forest-labs/FLUX.1-schnell","nvidia/C-RADIOv4-H","facebook/PE-Spatial-L14-448","deepseek-ai/DeepSeek-OCR"]:
    try:
        i=api.model_info(r, files_metadata=True)
        print("##",r,i.sha[:8],"base:",(i.card_data.base_model if i.card_data else None),"lic:",(i.card_data.license if i.card_data else None),(i.card_data.license_name if i.card_data else None))
        for s in i.siblings[:40]: print("   ",s.rfilename,s.size)
    except Exception as e: print("ERR",r,type(e).__name__,str(e)[:200])
# gated access tests
for r,f in [("stabilityai/stable-diffusion-3-medium-diffusers","model_index.json"),("black-forest-labs/FLUX.1-dev","model_index.json"),("black-forest-labs/FLUX.1-schnell","model_index.json"),("facebook/dinov3-convnext-tiny-pretrain-lvd1689m","config.json")]:
    try:
        p=hf_hub_download(r,f); print("ACCESS OK",r,f)
    except Exception as e: print("ACCESS FAIL",r,type(e).__name__,str(e)[:300].replace("\n"," "))
for u in ["https://api.github.com/repos/CompVis/cleandift/contents","https://api.github.com/repos/CompVis/cleandift/commits/HEAD","https://api.github.com/repos/0xnomy/urdu-deepseek-ocr","https://api.github.com/repos/NVlabs/RADIO/commits/HEAD","https://api.github.com/search/repositories?q=inkfm+google","https://api.github.com/search/repositories?q=InkFM","https://api.github.com/search/repositories?q=ink+foundation+model+handwriting"]:
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"fmd-staging"}),timeout=20))
        if isinstance(d,dict) and "items" in d: print(u,[ (x["full_name"],x["stargazers_count"],x["description"]) for x in d["items"][:8]])
        elif isinstance(d,list): print(u,[x["name"] for x in d])
        else: print(u, d.get("sha") or {k:d.get(k) for k in ["full_name","description","pushed_at","license"]})
    except Exception as e: print("GH ERR",u,e)
