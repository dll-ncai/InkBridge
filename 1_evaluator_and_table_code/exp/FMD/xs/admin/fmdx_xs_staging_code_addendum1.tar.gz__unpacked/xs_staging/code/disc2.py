
import os, json
os.environ["HF_HOME"]="$U/exp/FMD/xs/weights/.hf_home"
from huggingface_hub import HfApi
api=HfApi()
seen={}
def s(q, author=None, limit=40, sort="downloads"):
    try:
        for m in api.list_models(search=q, author=author, limit=limit, sort=sort):
            if m.id not in seen:
                seen[m.id]=(q, m.downloads, str(m.created_at)[:10], m.pipeline_tag)
                print(q,"|",m.id,m.downloads,str(m.created_at)[:10],m.pipeline_tag)
    except Exception as e: print("ERR",q,e)
for q,a in [("RADIO","nvidia"),("radio",None),("PE-Spatial","facebook"),("PE-Core","facebook"),("perception_encoder",None),("urdu ocr",None),("urdu deepseek",None),("DeepSeek-OCR",None),("handwriting",None),("handwritten",None),("inkfm",None),("ink foundation",None),("online handwriting",None),("digital ink",None),("cleandift",None),("DIFT",None),("depth-anything-3",None),("Depth-Anything-V2-Large","depth-anything"),("dinov2-with-registers","facebook"),("TIPS","google"),("tips-",None),("siglip2","google"),("webssl","facebook"),("vit-mae-large","facebook"),("dinov3-vit","facebook"),("dust3r",None),("florence-2","microsoft"),("trocr","microsoft"),("GOT-OCR",None),("paddleocr-vl",None),("lighton",None),("dots.ocr",None),("chandra",None),("qari",None),("Qaari",None),("sd-turbo","stabilityai"),("stable-diffusion-xl-base-1.0","stabilityai"),("sam3","facebook"),("sam-vit","facebook"),("sketch",None),("stroke",None),("deepseek-ocr urdu",None),("aimv2","apple"),("theia",None),("SigLIP-so400m","google"),("jina-clip",None)]:
    s(q,a,limit=15)
