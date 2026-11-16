import os
from huggingface_hub import HfApi
api=HfApi()
for q,a in [("pe_spatial","facebook"),("vit_pe","facebook"),("sam3","facebook"),("tipsv2","google")]:
    for m in api.list_models(search=q,author=a,limit=30): print(q,m.id,m.gated)
for r in ["facebook/sam3","facebook/vit_pe_spatial_gigantic_patch14_448_timm","timm/vit_pe_spatial_large_patch14_448.fb","timm/vit_pe_spatial_base_patch16_512.fb","timm/vit_pe_spatial_gigantic_patch14_448.fb"]:
    try:
        i=api.model_info(r,files_metadata=True); print(r,i.gated,[(s.rfilename,s.size) for s in i.siblings][:6])
    except Exception as e: print(r,"ERR",type(e).__name__)
