#!/usr/bin/env python3
"""FMDX staging v2: stage FM weights on CLUSTER-A with pinned HF revisions and per-file sha256. Track: XS staging (<agent>-5).
usage: stage_xs.py <group> [local_name ...]"""
import os, sys, json, hashlib, datetime, time
ROOT = "$U/exp/FMD/xs/weights"
os.environ["HF_HOME"] = ROOT + "/.hf_home"
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
SD_ALLOW = ["model_index.json","scheduler/*","tokenizer/*","feature_extractor/*","text_encoder/config.json","text_encoder/model.safetensors",
            "unet/config.json","unet/diffusion_pytorch_model.safetensors","vae/config.json","vae/diffusion_pytorch_model.safetensors","README.md"]
DIFFUSERS_ALLOW = ["model_index.json","scheduler/*","text_encoder/*.json","text_encoder/model.safetensors","text_encoder_2/*.json","text_encoder_2/model*.safetensors","text_encoder_2/*.index.json",
                   "text_encoder_3/*.json","text_encoder_3/model-*.safetensors","text_encoder_3/model.safetensors.index.json","tokenizer*/*","transformer/*.json","transformer/diffusion_pytorch_model*.safetensors","vae/*.json","vae/diffusion_pytorch_model.safetensors","LICENSE*","README.md"]
NOBIN = ["*.safetensors","*.json","*.txt","*.py","*.md","*.model","*.tiktoken","LICENSE*","*.yaml","*.pt","*.pth","*.tar","*.jinja"]
G = {
 "cleandift": [("cleandift","CompVis/cleandift",None,"CleanDIFT SD1.5/SD2.1 feature-extractor checkpoints (full + unet-only + SD2.1 depth probe)"),
               ("sd1.5","stable-diffusion-v1-5/stable-diffusion-v1-5",SD_ALLOW,"SD1.5 fp32 unet/vae/text_encoder (ungated community copy; base for CleanDIFT sd15)")],
 "radio": [("c-radiov3-b","nvidia/C-RADIOv3-B",None,"C-RADIOv3-B"),("c-radiov3-l","nvidia/C-RADIOv3-L",None,"C-RADIOv3-L"),
           ("c-radiov3-h","nvidia/C-RADIOv3-H",None,"C-RADIOv3-H"),("c-radiov3-g","nvidia/C-RADIOv3-g",None,"C-RADIOv3-g"),
           ("c-radiov4-h","nvidia/C-RADIOv4-H",None,"C-RADIOv4-H"),("c-radiov4-so400m","nvidia/C-RADIOv4-SO400M",None,"C-RADIOv4-SO400M")],
 "pe": [("pe-spatial-g14-448","facebook/PE-Spatial-G14-448",None,"PE-Spatial G14 448 (official .pt)"),("pe-spatial-l14-448","facebook/PE-Spatial-L14-448",None,"PE-Spatial L14 448 (official .pt)"),
        ("pe-spatial-b16-512","facebook/PE-Spatial-B16-512",None,"PE-Spatial B16 512 (official .pt)"),("pe-spatial-s16-512","facebook/PE-Spatial-S16-512",None,"PE-Spatial S16 512 (official .pt)"),
        ("pe-spatial-t16-512","facebook/PE-Spatial-T16-512",None,"PE-Spatial T16 512 (official .pt)"),
        ("pe-spatial-l14-448-timm","timm/vit_pe_spatial_large_patch14_448.fb",["*.safetensors","*.json","*.md"],"PE-Spatial L14 timm port (load-test path)"),
        ("pe-spatial-b16-512-timm","timm/vit_pe_spatial_base_patch16_512.fb",["*.safetensors","*.json","*.md"],"PE-Spatial B16 timm port"),
        ("pe-spatial-g14-448-timm","timm/vit_pe_spatial_gigantic_patch14_448.fb",["*.safetensors","*.json","*.md"],"PE-Spatial G14 timm port"),
        ("pe-spatial-s16-512-timm","timm/vit_pe_spatial_small_patch16_512.fb",["*.safetensors","*.json","*.md"],"PE-Spatial S16 timm port (if exists)"),
        ("pe-spatial-t16-512-timm","timm/vit_pe_spatial_tiny_patch16_512.fb",["*.safetensors","*.json","*.md"],"PE-Spatial T16 timm port (if exists)"),
        ("pe-core-b16-224","facebook/PE-Core-B16-224",None,"PE-Core B16 224 (PE-init candidate, 14x14 tokens at 224)"),
        ("pe-core-l14-336","facebook/PE-Core-L14-336",None,"PE-Core L14 336")],
 "ocr": [("starvector-1b-im2svg","starvector/starvector-1b-im2svg",None,"StarVector-1B im2svg"),
         ("qwen2-vl-2b-instruct","Qwen/Qwen2-VL-2B-Instruct",None,"Qwen2-VL-2B-Instruct (base of Qaari)"),
         ("qaari-0.1-urdu-ocr","oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct",None,"Qaari 0.1 Urdu OCR (LoRA adapter on unsloth Qwen2-VL-2B bnb-4bit)"),
         ("deepseek-ocr","deepseek-ai/DeepSeek-OCR",["*.safetensors","*.json","*.py","LICENSE","README.md"],"DeepSeek-OCR"),
         ("deepseek-ocr-2","deepseek-ai/DeepSeek-OCR-2",None,"DeepSeek-OCR-2"),
         ("urdu-ocr-deepseek-nomy","nomypython/urdu-ocr-deepseek",None,"Urdu DeepSeek-OCR LoRA (github 0xnomy/urdu-deepseek-ocr; base unsloth/DeepSeek-OCR)"),
         ("urdu-ocr-deepseek-ocr2-kingabzpro","kingabzpro/deepseek-ocr-2-urdu-ocr-1m-lora",None,"Urdu DeepSeek-OCR-2 LoRA (base deepseek-ai/DeepSeek-OCR-2)"),
         ("unsloth-deepseek-ocr","unsloth/DeepSeek-OCR",["*.safetensors","*.json","*.py"],"unsloth/DeepSeek-OCR = declared base of nomy Urdu adapter")],
 "variants": [("siglip2-giant-opt-patch16-256","google/siglip2-giant-opt-patch16-256",None,"SigLIP2 giant-opt 256 (no 224 variant exists)"),
              ("siglip2-giant-opt-patch16-384","google/siglip2-giant-opt-patch16-384",None,"SigLIP2 giant-opt 384"),
              ("dinov3-convnext-tiny","facebook/dinov3-convnext-tiny-pretrain-lvd1689m",None,"DINOv3 ConvNeXt-T"),
              ("dinov3-convnext-small","facebook/dinov3-convnext-small-pretrain-lvd1689m",None,"DINOv3 ConvNeXt-S"),
              ("dinov3-convnext-base","facebook/dinov3-convnext-base-pretrain-lvd1689m",None,"DINOv3 ConvNeXt-B"),
              ("dinov3-convnext-large","facebook/dinov3-convnext-large-pretrain-lvd1689m",None,"DINOv3 ConvNeXt-L"),
              ("dinov3-vits16","facebook/dinov3-vits16-pretrain-lvd1689m",None,"DINOv3 ViT-S/16"),
              ("dinov3-vits16plus","facebook/dinov3-vits16plus-pretrain-lvd1689m",None,"DINOv3 ViT-S+/16"),
              ("dinov3-vitb16","facebook/dinov3-vitb16-pretrain-lvd1689m",None,"DINOv3 ViT-B/16"),
              ("dinov3-vith16plus","facebook/dinov3-vith16plus-pretrain-lvd1689m",None,"DINOv3 ViT-H+/16"),
              ("dinov3-vit7b16","facebook/dinov3-vit7b16-pretrain-lvd1689m",None,"DINOv3 ViT-7B/16")],
 "gated": [("sd3-medium-diffusers","stabilityai/stable-diffusion-3-medium-diffusers",DIFFUSERS_ALLOW,"SD3-medium (gated; fp32 diffusers layout, no fp16 duplicates)"),
           ("flux1-dev","black-forest-labs/FLUX.1-dev",DIFFUSERS_ALLOW,"FLUX.1-dev (gated, non-commercial)"),
           ("flux1-schnell","black-forest-labs/FLUX.1-schnell",DIFFUSERS_ALLOW,"FLUX.1-schnell (gated, apache-2.0)"),
           ("sam3","facebook/sam3",None,"SAM3 (gated manual)")],
 "pecore": [("pe-core-b16-224-timm","timm/vit_pe_core_base_patch16_224.fb",["*.safetensors","*.json","*.md"],"PE-Core B16 224 timm port"),("pe-core-l14-336-timm","timm/vit_pe_core_large_patch14_336.fb",["*.safetensors","*.json","*.md"],"PE-Core L14 336 timm port")],
 "extras": [("tipsv2-b14","google/tipsv2-b14",None,"TIPSv2 B/14"),("tipsv2-l14","google/tipsv2-l14",None,"TIPSv2 L/14"),
            ("tipsv2-so400m14","google/tipsv2-so400m14",None,"TIPSv2 so400m/14"),("tipsv2-g14","google/tipsv2-g14",None,"TIPSv2 g/14"),
            ("dinov2-reg-large","facebook/dinov2-with-registers-large",None,"DINOv2+registers L/14"),
            ("webssl-dino1b-224","facebook/webssl-dino1b-full2b-224",None,"Web-SSL DINO 1B 224"),
            ("webssl-mae300m-224","facebook/webssl-mae300m-full2b-224",None,"Web-SSL MAE 300M 224"),
            ("vit-mae-large","facebook/vit-mae-large",["*.safetensors","*.bin","*.json","*.md"],"MAE ViT-L/16 (pytorch_model.bin only)"),
            ("depth-anything-v2-large","depth-anything/Depth-Anything-V2-Large-hf",None,"Depth-Anything-V2-L (HF)"),
            ("depth-anything-3-large-1.1","depth-anything/DA3-LARGE-1.1",None,"Depth-Anything-3 Large 1.1"),
            ("aimv2-large-patch14-224","apple/aimv2-large-patch14-224",None,"AIMv2 L/14 224"),
            ("trocr-large-handwritten","microsoft/trocr-large-handwritten",["*.safetensors","*.bin","*.json","*.txt","*.md"],"TrOCR-large handwritten (pytorch_model.bin only)"),
            ("florence-2-large","microsoft/Florence-2-large",["*.safetensors","*.json","*.py","*.txt","*.md"],"Florence-2-large (DaViT)"),
            ("got-ocr2-hf","stepfun-ai/GOT-OCR-2.0-hf",["*.safetensors","*.json","*.txt","*.md"],"GOT-OCR2.0 (HF)"),
            ("paddleocr-vl-1.6","PaddlePaddle/PaddleOCR-VL-1.6",["*.safetensors","*.json","*.py","*.txt","*.md","*.model"],"PaddleOCR-VL 1.6"),
            ("lightonocr-2-1b","lightonai/LightOnOCR-2-1B",["*.safetensors","*.json","*.txt","*.md","*.py"],"LightOnOCR-2-1B")],
}
only = sys.argv[2:]
group = sys.argv[1]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
for name, repo, allow, role in G[group]:
    if only and name not in only: continue
    out = f"{ROOT}/{name}"; rec = {"local_name": name, "repo_id": repo, "role": role, "group": group, "allow_patterns": allow}
    try:
        info = api.model_info(repo, files_metadata=True)
        cd = info.card_data
        rec.update(revision=info.sha, gated=str(info.gated), license=(cd.license if cd else None) or [t for t in info.tags if t.startswith("license")],
                   license_name=getattr(cd, "license_name", None) if cd else None, license_link=getattr(cd, "license_link", None) if cd else None,
                   base_model=(cd.base_model if cd else None), last_modified=str(info.last_modified),
                   safetensors_total_params=(info.safetensors.total if info.safetensors else None))
        t0 = time.time()
        snapshot_download(repo, revision=info.sha, local_dir=out, allow_patterns=allow, max_workers=8)
        files = []
        for dp, dn, fn in os.walk(out):
            dn[:] = [d for d in dn if d != ".cache"]
            for f in sorted(fn):
                p = os.path.join(dp, f); files.append({"path": os.path.relpath(p, out), "bytes": os.path.getsize(p), "sha256": sha256(p)})
        files.sort(key=lambda x: x["path"])
        rec.update(status="ok", files=files, total_bytes=sum(x["bytes"] for x in files), n_files=len(files),
                   download_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), seconds=round(time.time()-t0),
                   command=f"snapshot_download('{repo}', revision='{info.sha}', local_dir='{out}', allow_patterns={allow})")
    except Exception as e:
        rec.update(status="error", error_type=type(e).__name__, error=str(e)[:1500])
    json.dump(rec, open(f"{ROOT}/{name}.manifest_part.json", "w"), indent=1)
    print(name, rec["status"], rec.get("total_bytes"), rec.get("error_type"), flush=True)
