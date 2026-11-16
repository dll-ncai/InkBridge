#!/usr/bin/env python3
"""FMDX staging addendum 1 (3 Nov 2026, lead follow-up 'fetch ALL candidates'): pinned HF snapshots + per-file sha256.  Track XS staging (<agent>).
usage: stage_xs_v3.py <group> [local_name ...]   (manifest parts go to xs/weights/<name>.manifest_part.json, group names start with add_)"""
import os, sys, json, hashlib, datetime, time
ROOT = "$U/exp/FMD/xs/weights"
os.environ["HF_HOME"] = ROOT + "/.hf_home"
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
SAFE = ["*.safetensors", "*.json", "*.txt", "*.md", "*.py", "*.model", "*.tiktoken", "*.jinja", "LICENSE*", "*.vocab", "*.yaml", "*.yml"]
SAFEBIN = SAFE + ["*.bin"]
DIFF = ["model_index.json","scheduler/*","text_encoder/*.json","text_encoder/model.safetensors","text_encoder_2/*.json","text_encoder_2/model*.safetensors","text_encoder_2/*.index.json",
        "text_encoder_3/*.json","text_encoder_3/model-*.safetensors","text_encoder_3/model.safetensors.index.json","tokenizer*/*","transformer/*.json","transformer/diffusion_pytorch_model*.safetensors","vae/*.json","vae/diffusion_pytorch_model.safetensors","LICENSE*","README.md"]
def e(name, repo, allow, role): return (name, repo, allow, role)
G = {
 "add_named": [
  e("starvector-8b-im2svg", "starvector/starvector-8b-im2svg", SAFE, "StarVector-8B im2svg"),
  e("pe-core-g14-448", "facebook/PE-Core-G14-448", None, "PE-Core G14 448 (official .pt)"),
  e("pe-core-g14-448-timm", "facebook/vit_pe_core_gigantic_patch14_448_timm", ["*.safetensors", "*.json", "*.md"], "PE-Core G14 448 timm port (7.5 GB; load-test path)"),
  e("pe-core-s16-384", "facebook/PE-Core-S16-384", None, "PE-Core S16 384 (official .pt)"),
  e("pe-core-t16-384", "facebook/PE-Core-T16-384", None, "PE-Core T16 384 (official .pt)"),
  e("pe-lang-l14-448-timm", "facebook/vit_pe_lang_large_patch14_448_timm", ["*.safetensors", "*.json", "*.md"], "PE-Lang L14 448 timm port"),
  e("pe-lang-g14-448-timm", "facebook/vit_pe_lang_gigantic_patch14_448_timm", ["*.safetensors", "*.json", "*.md"], "PE-Lang G14 448 timm port"),
  e("theia-tiny-cddsv", "theaiinstitute/theia-tiny-patch16-224-cddsv", SAFE, "Theia tiny cddsv"), e("theia-tiny-cdiv", "theaiinstitute/theia-tiny-patch16-224-cdiv", SAFE, "Theia tiny cdiv"),
  e("theia-small-cdiv", "theaiinstitute/theia-small-patch16-224-cdiv", SAFE, "Theia small cdiv"), e("theia-base-cdiv", "theaiinstitute/theia-base-patch16-224-cdiv", SAFE, "Theia base cdiv"),
  e("theia-base-cddsv", "theaiinstitute/theia-base-patch16-224-cddsv", SAFE, "Theia base cddsv"),
  e("sd3.5-medium-diffusers-layout", "stabilityai/stable-diffusion-3.5-medium", DIFF, "SD3.5-medium (gated)"), e("sd3.5-large-diffusers-layout", "stabilityai/stable-diffusion-3.5-large", DIFF, "SD3.5-large (gated)"),
  e("sd3.5-large-turbo-diffusers-layout", "stabilityai/stable-diffusion-3.5-large-turbo", DIFF, "SD3.5-large-turbo (gated)"), e("sam3.1", "facebook/sam3.1", None, "SAM3.1 (gated manual)")],
 "add_ocr1": [
  e("chandra-ocr-2", "datalab-to/chandra-ocr-2", SAFE, "Chandra OCR 2"), e("chandra", "datalab-to/chandra", SAFE, "Chandra OCR (v1)"),
  e("dots-ocr", "dots-studio/dots.ocr", SAFE, "dots.ocr"), e("dots-ocr-base", "dots-studio/dots.ocr.base", SAFE, "dots.ocr.base")],
 "add_ocr2": [
  e("lightonocr-3-4b", "lightonai/LightOnOCR-3-4B", SAFE, "LightOnOCR-3-4B"), e("lightonocr-3-0.8b", "lightonai/LightOnOCR-3-0.8B", SAFE, "LightOnOCR-3-0.8B"),
  e("paddleocr-vl-1.5", "PaddlePaddle/PaddleOCR-VL-1.5", SAFE, "PaddleOCR-VL-1.5"), e("paddleocr-vl-1.0", "PaddlePaddle/PaddleOCR-VL", SAFE, "PaddleOCR-VL (v1)"),
  e("florence-2-base", "microsoft/Florence-2-base", SAFE, "Florence-2-base"), e("florence-2-large-ft", "microsoft/Florence-2-large-ft", SAFE, "Florence-2-large-ft"),
  e("trocr-base-handwritten", "microsoft/trocr-base-handwritten", SAFE, "TrOCR base handwritten"), e("trocr-small-handwritten", "microsoft/trocr-small-handwritten", SAFEBIN, "TrOCR small handwritten (bin only)"),
  e("trocr-base-hw-hist-swe-2", "Riksarkivet/trocr-base-handwritten-hist-swe-2", SAFE, "TrOCR base hand. historical Swedish"),
  e("surya-ocr-2-persian-hw", "Reza2kn/surya-ocr-2-persian-handwriting-369k", SAFE, "Surya-OCR-2 Persian handwriting fine-tune"),
  e("arabic-ocr-deepseek-ocr-2", "loay/Arabic-OCR-DeepSeek-OCR-2", SAFE, "Arabic OCR DeepSeek-OCR-2 fine-tune"),
  e("qari-ocr-v0.3-vl-2b", "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct", SAFE, "Qari-OCR v0.3 (Arabic, Qwen2-VL-2B)"),
  e("qari-ocr-0.4.0-vl-4b", "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct", SAFE, "Qari-OCR 0.4.0 adapter (base Qwen3-VL-4B)"), e("qwen3-vl-4b-instruct", "Qwen/Qwen3-VL-4B-Instruct", SAFE, "Qwen3-VL-4B base of Qari 0.4.0"),
  e("arabic-english-hw-ocr-v3", "sherif1313/Arabic-English-handwritten-OCR-v3", SAFE, "Arabic-English handwritten OCR v3 (Qwen2.5-VL-3B)"),
  e("arabic-hw-ocr-4bit", "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2", SAFE, "Arabic handwritten OCR 4-bit (Qwen2.5-VL-3B)"), e("qwen2.5-vl-3b-instruct", "Qwen/Qwen2.5-VL-3B-Instruct", SAFE, "Qwen2.5-VL-3B base")],
 "add_ssl1": [
  e("webssl-dino300m-224", "facebook/webssl-dino300m-full2b-224", SAFE, "Web-SSL DINO 300M"), e("webssl-dino2b-224", "facebook/webssl-dino2b-full2b-224", SAFE, "Web-SSL DINO 2B"),
  e("webssl-dino3b-224", "facebook/webssl-dino3b-full2b-224", SAFE, "Web-SSL DINO 3B"), e("webssl-mae1b-224", "facebook/webssl-mae1b-full2b-224", SAFE, "Web-SSL MAE 1B"), e("webssl-mae3b-224", "facebook/webssl-mae3b-full2b-224", SAFE, "Web-SSL MAE 3B"),
  e("dinov2-reg-small", "facebook/dinov2-with-registers-small", SAFE, "DINOv2+reg S"), e("dinov2-reg-base", "facebook/dinov2-with-registers-base", SAFE, "DINOv2+reg B"), e("dinov2-reg-giant", "facebook/dinov2-with-registers-giant", SAFE, "DINOv2+reg g"),
  e("dinov2-giant", "facebook/dinov2-giant", SAFE, "DINOv2 g/14"), e("dinov2-base", "facebook/dinov2-base", SAFE, "DINOv2 B/14")],
 "add_ssl2": [
  e("webssl-dino5b-224", "facebook/webssl-dino5b-full2b-224", SAFE, "Web-SSL DINO 5B"), e("webssl-dino7b-224", "facebook/webssl-dino7b-full8b-224", SAFE, "Web-SSL DINO 7B (8B-image set)"),
  e("aimv2-huge-patch14-224", "apple/aimv2-huge-patch14-224", SAFE, "AIMv2 H/14"), e("aimv2-1b-patch14-224", "apple/aimv2-1B-patch14-224", SAFE, "AIMv2 1B/14"),
  e("aimv2-3b-patch14-224", "apple/aimv2-3B-patch14-224", SAFE, "AIMv2 3B/14"), e("aimv2-large-patch14-224-distilled", "apple/aimv2-large-patch14-224-distilled", SAFE, "AIMv2 L/14 distilled")],
 "add_misc": [
  e("tipsv1-s14", "google/tipsv1-s14", SAFE, "TIPS v1 S/14"), e("tipsv1-b14", "google/tipsv1-b14", SAFE, "TIPS v1 B/14"), e("tipsv1-l14", "google/tipsv1-l14", SAFE, "TIPS v1 L/14"),
  e("tipsv1-so400m14", "google/tipsv1-so400m14", SAFE, "TIPS v1 so400m/14"), e("tipsv1-g14", "google/tipsv1-g14", SAFE, "TIPS v1 g/14"),
  e("depth-anything-3-giant-1.1", "depth-anything/DA3-GIANT-1.1", SAFE, "Depth-Anything-3 Giant 1.1"), e("depth-anything-3-base", "depth-anything/DA3-BASE", SAFE, "Depth-Anything-3 Base"),
  e("dinov3-vitl16-sat493m", "facebook/dinov3-vitl16-pretrain-sat493m", SAFE, "DINOv3 ViT-L/16 satellite (gated manual; licence: stays on CLUSTER-A)"),
  e("siglip2-large-patch16-256", "google/siglip2-large-patch16-256", SAFE, "SigLIP2 L/16 256"), e("siglip2-so400m-patch16-256", "google/siglip2-so400m-patch16-256", SAFE, "SigLIP2 so400m/16 256")],
}
G["add_cfg"] = [e("siglip-large-patch16-384-cfg", "google/siglip-large-patch16-384", ["config.json", "preprocessor_config.json"], "config only (SigLIP-L/16-384 vision config used to rebuild the StarVector-8B image encoder)")]
G["add_sat7b"] = [e("dinov3-vit7b16-sat493m", "facebook/dinov3-vit7b16-pretrain-sat493m", SAFE, "DINOv3 ViT-7B/16 satellite (gated manual; licence: stays on CLUSTER-A)")]
only = sys.argv[2:]; group = sys.argv[1]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
for name, repo, allow, role in G[group]:
    if only and name not in only: continue
    out = f"{ROOT}/{name}"; rec = {"local_name": name, "repo_id": repo, "role": role, "group": group, "allow_patterns": allow}
    try:
        info = api.model_info(repo, files_metadata=True); cd = info.card_data
        rec.update(revision=info.sha, gated=str(info.gated), license=(cd.license if cd else None) or [t for t in info.tags if t.startswith("license")],
                   license_name=getattr(cd, "license_name", None) if cd else None, license_link=getattr(cd, "license_link", None) if cd else None,
                   base_model=(cd.base_model if cd else None), last_modified=str(info.last_modified), safetensors_total_params=(info.safetensors.total if info.safetensors else None))
        t0 = time.time()
        snapshot_download(repo, revision=info.sha, local_dir=out, allow_patterns=allow, max_workers=8)
        files = []
        for dp, dn, fn in os.walk(out):
            dn[:] = [d for d in dn if d != ".cache"]
            for f in sorted(fn):
                p = os.path.join(dp, f); files.append({"path": os.path.relpath(p, out), "bytes": os.path.getsize(p), "sha256": sha256(p)})
        files.sort(key=lambda x: x["path"])
        rec.update(status="ok", files=files, total_bytes=sum(x["bytes"] for x in files), n_files=len(files), download_date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), seconds=round(time.time()-t0),
                   command=f"snapshot_download('{repo}', revision='{info.sha}', local_dir='{out}', allow_patterns={allow})")
    except Exception as ex:
        rec.update(status="error", error_type=type(ex).__name__, error=str(ex)[:1500])
    json.dump(rec, open(f"{ROOT}/{name}.manifest_part.json", "w"), indent=1)
    print(name, rec["status"], rec.get("total_bytes"), rec.get("error_type"), flush=True)
