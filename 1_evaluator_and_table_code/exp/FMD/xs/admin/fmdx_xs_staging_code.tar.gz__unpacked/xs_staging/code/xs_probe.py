#!/usr/bin/env python3
"""XS staging: loaders + load test + extraction-rate timing for the FMDX candidate FMs.  Track 'XS staging' (<agent>).
NOT the pre-registered extractors (those are fixed by prereg amendment A3): conventions here are LOAD-TEST conventions
(which layer, which input size) chosen to obtain a token-map shape at 224 px and a throughput number.  Deviations of the input size
from 224 are reported per model in 'input_px'.
usage:  xs_probe.py load  <name> [<name>...]     CPU or GPU, 4 val1000 renders -> admin/loadtest/<name>.json
        xs_probe.py time  <tag> <name> [...]     GPU, 1000 val1000 renders -> admin/timing/<tag>__<name>.json (setup excluded)"""
import os, sys, json, time, traceback, hashlib, glob, platform, datetime
from pathlib import Path
sys.path.insert(0, "$U/exp/FMD/xs/pylibs")
if os.environ.get("XS_DS") == "1": sys.path.insert(0, "$U/exp/FMD/xs/pylibs_ds")   # transformers 4.46.3 stack (DeepSeek-OCR remote code)
sys.path.insert(0, "$U/exp/FMD/code/fmd_harness_v1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
torch.set_grad_enabled(False)
W = Path("$U/exp/FMD/xs/weights"); W1 = Path("$U/exp/FMD/weights")
ADM = Path("$U/exp/FMD/xs/admin")
VAL = Path("$U/exp/FMD/data/fmd_render224_val1000.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); HALF = ((0.5,) * 3, (0.5,) * 3); NONE = ((0.0,) * 3, (1.0,) * 3)

def pp_norm(wd, default=IMNET):
    p = Path(wd) / "preprocessor_config.json"
    if p.exists():
        j = json.load(open(p))
        if "image_mean" in j: return (tuple(j["image_mean"]), tuple(j["image_std"]))
    return default

def to_nchw(x):          # tokens or maps -> [N,H,W,C] float32 numpy
    return x

def grid_tokens(t, n_prefix=None):
    """[B,T,C] -> [B,g,g,C] dropping prefix tokens so that T-prefix is a square."""
    T = t.shape[1]
    g = int(round((T - (n_prefix or 0)) ** 0.5))
    if n_prefix is None:
        for npre in range(0, 9):
            g = int(round((T - npre) ** 0.5))
            if g * g == T - npre: n_prefix = npre; break
    t = t[:, n_prefix:, :]; g = int(round(t.shape[1] ** 0.5)); assert g * g == t.shape[1], t.shape
    return t.reshape(t.shape[0], g, g, -1)

class Wrap:
    """fwd(x01: float [N,3,224,224] in [0,1] on dev) -> {layer: [N,H,W,C] tensor}"""
    def __init__(self, fwd, info): self.fwd, self.info = fwd, info

def _norm(x, ms):
    m = torch.tensor(ms[0], device=x.device, dtype=x.dtype).view(1, 3, 1, 1); s = torch.tensor(ms[1], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s

def _resize(x, px):
    return x if x.shape[-1] == px else torch.nn.functional.interpolate(x, size=(px, px), mode="bilinear", align_corners=False, antialias=px < x.shape[-1])

def dt_for(n_params):
    return torch.float32 if n_params < 1.6e9 else torch.bfloat16

# ------------------------------------------------------------------ loaders
def load_hf_vit(wd, dtype=None, **kw):
    """HF AutoModel vision encoder (dinov2/dinov3/webssl/mae/siglip2/aimv2/...). Tries 224 directly, then interpolate_pos_encoding, then native size."""
    import transformers
    wd = Path(wd); cfg = json.load(open(wd / "config.json"))
    mt = cfg.get("model_type", "")
    ms = pp_norm(wd)
    try:
        full = transformers.AutoModel.from_pretrained(str(wd), torch_dtype=dtype or torch.float32, trust_remote_code=False)
        trc = False
    except Exception:
        full = transformers.AutoModel.from_pretrained(str(wd), torch_dtype=dtype or torch.float32, trust_remote_code=True)
        trc = True
    m = full
    if hasattr(full, "vision_model"): m = full.vision_model
    elif hasattr(full, "vision_tower"): m = full.vision_tower
    m = m.eval().to(DEV)
    nparam = sum(p.numel() for p in m.parameters())
    native = cfg.get("vision_config", cfg).get("image_size", 224)
    state = {"mode": None}
    is_mae = mt == "vit_mae"
    if is_mae: m.config.mask_ratio = 0.0; m.embeddings.config.mask_ratio = 0.0
    def run(x, px, interp):
        extra = {}
        if is_mae:
            L = (px // m.config.patch_size) ** 2; extra["noise"] = torch.arange(L, device=x.device, dtype=torch.float32)[None].expand(x.shape[0], -1)   # identity order, mask_ratio 0
        o = m(pixel_values=_norm(_resize(x, px), ms).to(next(m.parameters()).dtype), output_hidden_states=True, **({"interpolate_pos_encoding": True} if interp else {}), **extra)
        return o
    def fwd(x):
        modes = [(224, False), (224, True), (native, False)] if state["mode"] is None else [state["mode"]]
        err = None
        for px, interp in modes:
            try:
                o = run(x, px, interp); state["mode"] = (px, interp); break
            except Exception as e:
                err = e; o = None
        if o is None: raise err
        hs = o.hidden_states; last = o.last_hidden_state
        outs = {}
        for nm, t in (("last", last), ("mid", hs[len(hs) // 2])):
            if t.ndim == 4:
                t = t.permute(0, 2, 3, 1) if t.shape[1] != t.shape[2] or t.shape[1] > 4096 and False else t
                if t.shape[1] > t.shape[-1]: pass
                outs[nm] = t
            else: outs[nm] = grid_tokens(t)
        return outs
    info = dict(cls=type(m).__name__, params=nparam, dtype=str(next(m.parameters()).dtype), model_type=mt, native_image_size=native, norm=str(ms), trust_remote_code_used=trc, mae_no_mask=is_mae)
    _f = fwd
    def fwd2(x):
        o = _f(x); info["input_mode_(px,interpolate_pos_encoding)"] = str(state["mode"]); return o
    return Wrap(fwd2, info)

def load_dinov3_convnext(wd, dtype=None, **kw):
    import transformers
    m = transformers.AutoModel.from_pretrained(str(wd), torch_dtype=dtype or torch.float32).eval().to(DEV)
    ms = pp_norm(wd)
    def fwd(x):
        o = m(pixel_values=_norm(x, ms), output_hidden_states=True)
        hs = o.hidden_states
        mp = lambda t: t.permute(0, 2, 3, 1) if t.ndim == 4 and t.shape[1] > t.shape[-1] else t
        return {"last": mp(hs[-1]), "mid": mp(hs[-2])}
    return Wrap(fwd, dict(cls=type(m).__name__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), norm=str(ms)))

def load_radio(wd, dtype=None, **kw):
    import transformers
    m = transformers.AutoModel.from_pretrained(str(wd), trust_remote_code=True, torch_dtype=dtype or torch.float32).eval().to(DEV)
    ps = int(getattr(m, "patch_size", 16))
    def fwd(x):
        o = m(x.to(next(m.parameters()).dtype))
        if hasattr(o, "features"): f = o.features
        elif isinstance(o, (tuple, list)): f = o[1]
        elif isinstance(o, dict): f = o["backbone"][1] if "backbone" in o else list(o.values())[-1]
        else: f = o
        if isinstance(f, (tuple, list)): f = f[-1]
        if f.ndim == 3: f = grid_tokens(f, 0)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(m).__name__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), patch_size=ps, norm="none ([0,1] RGB; model has built-in input conditioner)"))

PE_TIMM = {"pe-spatial-g14-448": ("pe-spatial-g14-448-timm", "vit_pe_spatial_gigantic_patch14_448.fb"), "pe-spatial-l14-448": ("pe-spatial-l14-448-timm", "vit_pe_spatial_large_patch14_448.fb"),
           "pe-spatial-b16-512": ("pe-spatial-b16-512-timm", "vit_pe_spatial_base_patch16_512.fb"), "pe-spatial-s16-512": ("pe-spatial-s16-512-timm", "vit_pe_spatial_small_patch16_512.fb"),
           "pe-spatial-t16-512": ("pe-spatial-t16-512-timm", "vit_pe_spatial_tiny_patch16_512.fb"), "pe-core-b16-224": ("pe-core-b16-224-timm", "vit_pe_core_base_patch16_224.fb"), "pe-core-l14-336": ("pe-core-l14-336-timm", "vit_pe_core_large_patch14_336.fb")}
def load_pe(name, dtype=None, **kw):
    import timm
    tdir, tname = PE_TIMM[name]
    f = None
    if tdir:
        fs = sorted((W / tdir).glob("*.safetensors")); f = str(fs[0]) if fs else None
    if f is None: raise FileNotFoundError(f"timm port dir for {name} not staged ({tdir})")
    err = None
    for ia in (dict(img_size=224), dict(img_size=224, dynamic_img_size=True), dict(dynamic_img_size=True)):
        try:
            m = timm.create_model(tname, pretrained=True, pretrained_cfg_overlay=dict(file=f), num_classes=0, **ia); break
        except Exception as e: err = e; m = None
    if m is None: raise err
    m = m.eval().to(dtype or torch.float32).to(DEV)
    cfgd = m.pretrained_cfg; ms = (tuple(cfgd.get("mean", (0.5,) * 3)), tuple(cfgd.get("std", (0.5,) * 3)))
    def fwd(x):
        t = m.forward_features(_norm(x, ms).to(next(m.parameters()).dtype))
        if t.ndim == 3: t = grid_tokens(t)
        else: t = t.permute(0, 2, 3, 1)
        return {"last": t}
    return Wrap(fwd, dict(cls=type(m).__name__, timm_name=tname, timm=__import__("timm").__version__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), init_kwargs=str(ia), norm=str(ms), weight_file=f))

def load_cleandift(variant, dtype=None, **kw):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel
    from safetensors.torch import load_file
    base = W1 / "sd2.1" if variant == "sd21" else W / "sd1.5"
    ck = W / "cleandift" / f"cleandift_{variant if variant=='sd21' else 'sd15'}_unet.safetensors"
    res = 768 if variant == "sd21" else 512
    dt = torch.float32
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dt).to(DEV).eval()
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dt)
    sd = load_file(str(ck)); r = unet.load_state_dict(sd, strict=True); unet = unet.to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer"); te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dt).to(DEV).eval()
    emb = te(tok([""], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEV))[0]; del te
    feats = {}
    unet.up_blocks[1].register_forward_hook(lambda m, i, o: feats.__setitem__("up1", o))
    class Stop(Exception): pass
    unet.up_blocks[2].register_forward_hook(lambda m, i, o: feats.__setitem__("up2", o))
    def fwd(x):
        xi = _resize(x, res) * 2 - 1
        z = vae.encode(xi).latent_dist.mean * vae.config.scaling_factor
        try:
            unet(z, torch.zeros(z.shape[0], dtype=torch.long, device=z.device), encoder_hidden_states=emb.expand(z.shape[0], -1, -1))
        except Stop: pass
        return {"up1": feats["up1"].permute(0, 2, 3, 1), "up2": feats["up2"].permute(0, 2, 3, 1)}
    return Wrap(fwd, dict(cls="UNet2DConditionModel", params=sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in vae.parameters()), dtype="float32", input_px=res,
                          base=str(base), cleandift_ckpt=str(ck), strict_load=str(r), convention="LOAD-TEST ONLY: clean latent (posterior mean), t=0, empty prompt, up_block outputs (full UNet forward; no early stop)"))

def load_qwen2vl_vision(wd, dtype=None, adapter=None, **kw):
    import transformers
    from PIL import Image
    wd = Path(wd)
    full = transformers.Qwen2VLForConditionalGeneration.from_pretrained(str(wd), torch_dtype=dtype or torch.bfloat16)
    vis = full.model.visual if hasattr(full, "model") and hasattr(full.model, "visual") else full.visual
    merged_info = None
    if adapter:
        from safetensors.torch import load_file
        ad = load_file(str(Path(adapter) / "adapter_model.safetensors")); acfg = json.load(open(Path(adapter) / "adapter_config.json"))
        sc = acfg["lora_alpha"] / acfg["r"]; n = 0
        for k in list(ad):
            if ".visual." in k and k.endswith("lora_A.weight"):
                mod = k.split("visual.", 1)[1][:-len(".lora_A.weight")]
                B = ad[k.replace("lora_A", "lora_B")].float(); A = ad[k].float()
                tgt = vis.get_submodule(mod); tgt.weight.data = (tgt.weight.data.float() + sc * (B @ A)).to(tgt.weight.dtype); n += 1
        merged_info = dict(n_visual_lora_modules_merged=n, scale=sc, adapter=str(adapter))
    vis = vis.eval().to(DEV); del full
    ip = transformers.AutoImageProcessor.from_pretrained(str(wd))
    store = {}
    vis.blocks[-1].register_forward_hook(lambda m, i, o: store.__setitem__("pre", o[0] if isinstance(o, tuple) else o))
    def fwd(x):
        arr = (x.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
        imgs = [Image.fromarray(a) for a in arr]
        pi = ip(images=imgs, return_tensors="pt", min_pixels=224 * 224, max_pixels=224 * 224)
        pv, thw = pi["pixel_values"].to(DEV, next(vis.parameters()).dtype), pi["image_grid_thw"].to(DEV)
        vo = vis(pv, grid_thw=thw)
        n = x.shape[0]; h, w = int(thw[0, 1]), int(thw[0, 2])
        pre_t = getattr(vo, "last_hidden_state", None); merged = getattr(vo, "pooler_output", None) if hasattr(vo, "pooler_output") else vo
        pre = pre_t.reshape(n, h, w, -1) if pre_t is not None and pre_t.shape[0] == n * h * w else store["pre"].reshape(n, h, w, -1)
        mg = merged.reshape(n, h // 2, w // 2, -1)
        return {"pre_merge": pre, "last": mg}
    return Wrap(fwd, dict(cls=type(vis).__name__, params=sum(p.numel() for p in vis.parameters()), dtype=str(next(vis.parameters()).dtype), input_px=224, note="vision tower of Qwen2-VL; 224 px -> 16x16 patches (pre-merge), 8x8 after 2x2 merger", lora_merge=merged_info))

def load_deepseek_ocr(wd, dtype=None, **kw):
    sys.path.insert(0, "$U/exp/FMD/xs/pylibs_ds")   # transformers 4.46.3 stack required by DeepSeek-OCR remote code
    import transformers
    m = transformers.AutoModel.from_pretrained(str(wd), trust_remote_code=True, torch_dtype=dtype or torch.bfloat16, use_safetensors=True).eval()
    inner = m.model if hasattr(m, "model") else m
    if not hasattr(inner, "vision_model"):
        raise NotImplementedError("DeepSeek-OCR-2 loaded (class %s, encoder modules %s); its encoder is SAM + Qwen2 decoder-as-encoder (no CLIP vision_model) - extractor not written" % (type(inner).__name__, [n for n, _ in inner.named_children()][:8]))
    sam, clip = inner.sam_model.to(DEV).eval(), inner.vision_model.to(DEV).eval()
    dt = next(sam.parameters()).dtype
    ms = ((0.5,) * 3, (0.5,) * 3)
    state = {"px": None}
    def run(x, px):
        xx = _norm(_resize(x, px), ms).to(dt)
        s = sam(xx)
        c = clip(xx, s)
        return s, c
    def fwd(x):
        err = None
        for px in ([224, 512] if state["px"] is None else [state["px"]]):
            try:
                s, c = run(x, px); state["px"] = px; break
            except Exception as e: err = e; s = None
        if s is None: raise err
        out = {"sam_last": s.permute(0, 2, 3, 1)}
        out["clip_last"] = grid_tokens(c)
        return out
    return Wrap(fwd, dict(cls="SAM-ViT-B + CLIP-L (DeepEncoder)", params=sum(p.numel() for p in sam.parameters()) + sum(p.numel() for p in clip.parameters()), dtype=str(dt), note="DeepSeek-OCR DeepEncoder; first tries 224 px direct, then 512 px (Tiny mode native)"))

def load_generic_extra(wd, dtype=None, **kw):
    return load_hf_vit(wd, dtype=dtype)

def load_trocr_encoder(wd, dtype=None, **kw):
    import transformers
    m = transformers.VisionEncoderDecoderModel.from_pretrained(str(wd), torch_dtype=dtype or torch.float32).encoder.eval().to(DEV)
    ms = pp_norm(wd, HALF); size = m.config.image_size
    def fwd(x):
        o = m(pixel_values=_norm(_resize(x, size), ms).to(next(m.parameters()).dtype), output_hidden_states=True)
        return {"last": grid_tokens(o.last_hidden_state)}
    return Wrap(fwd, dict(cls=type(m).__name__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), input_px=size, norm=str(ms)))

def load_tips(wd, dtype=None, **kw):
    import transformers
    m = transformers.AutoModel.from_pretrained(str(wd), trust_remote_code=True, torch_dtype=dtype or torch.float32).eval().to(DEV)
    def fwd(x):
        o = m.encode_image(x.to(next(m.parameters()).dtype)) if hasattr(m, "encode_image") else m(x)
        # TIPS image encoder returns (cls, registers, patch tokens) or similar
        f = o.patch_tokens if hasattr(o, "patch_tokens") else (o[-1] if isinstance(o, (tuple, list)) else o)
        if f.ndim == 3: f = grid_tokens(f, 0) if int(round(f.shape[1] ** 0.5)) ** 2 == f.shape[1] else grid_tokens(f)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(m).__name__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), norm="none ([0,1])"))


def load_starvector(wd, dtype=None, **kw):
    import types, importlib.util
    from safetensors import safe_open
    fa = types.ModuleType("fairscale"); fn = types.ModuleType("fairscale.nn"); fc = types.ModuleType("fairscale.nn.checkpoint"); fcc = types.ModuleType("fairscale.nn.checkpoint.checkpoint_activations")
    fcc.checkpoint_wrapper = lambda m, *a, **k: m
    for n, m in (("fairscale", fa), ("fairscale.nn", fn), ("fairscale.nn.checkpoint", fc), ("fairscale.nn.checkpoint.checkpoint_activations", fcc)): sys.modules[n] = m
    spec = importlib.util.spec_from_file_location("sv_clip_model", "$U/exp/FMD/xs/weights/github/star-vector/starvector/model/image_encoder/clip_model.py")
    cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
    vit = cm.VisionTransformer(input_resolution=224, patch_size=14, width=1024, layers=23, heads=16, use_grad_checkpointing=False)
    ln = cm.LayerNorm(1024)
    sd = {}
    for f in sorted(Path(wd).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith("model.image_encoder."): sd[k[len("model.image_encoder."):]] = sf.get_tensor(k).float()
    vs = {k[len("visual_encoder."):]: v for k, v in sd.items() if k.startswith("visual_encoder.")}
    ls = {k[len("ln_vision."):]: v for k, v in sd.items() if k.startswith("ln_vision.")}
    r1 = vit.load_state_dict(vs, strict=True); r2 = ln.load_state_dict(ls, strict=True)
    vit, ln = vit.eval().to(DEV), ln.eval().to(DEV)
    ms = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    def fwd(x):
        o = ln(vit(_norm(x, ms)))
        return {"last": grid_tokens(o, 1)}
    return Wrap(fwd, dict(cls="StarVector CLIP-ViT (width 1024, 23 layers, patch 14) + ln_vision", params=sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in ln.parameters()), dtype="float32", input_px=224,
                          norm=str(ms), strict_load=str(r1) + " " + str(r2), note="encoder re-built from starvector repo clip_model.py (fairscale checkpoint_wrapper stubbed; commit in MANIFEST github list); weights = model.image_encoder.* of starvector-1b-im2svg"))

def load_instantiate_only(wd, dtype=None, **kw):
    import transformers
    wd = Path(wd); err = []
    for cls in ("AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            m = getattr(transformers, cls).from_pretrained(str(wd), trust_remote_code=True, torch_dtype=dtype or torch.bfloat16)
            n = sum(p.numel() for p in m.parameters())
            def fwd(x): raise NotImplementedError(f"instantiate-only: {type(m).__name__} ({cls}) loaded, {n} params; no feature-extraction loader written (extras tier)")
            return Wrap(fwd, dict(cls=type(m).__name__, auto_class=cls, params=n, dtype=str(next(m.parameters()).dtype), load_only=True))
        except Exception as e:
            err.append(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(" | ".join(err))


def load_sd21_dift_gate(variant="t261_up1", dtype=None, **kw):
    """Cost proxy of the FMD DIFT gate extractor: 768-px, 4 noise draws, t=261, UNet stopped after up_block 1 (fp32).  Same as exp/FMD/code/fmd_ext_diffusion/sd21_dift.py (gate layer only)."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel
    base = W1 / "sd2.1"; dt = torch.float32
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dt).to(DEV).eval()
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dt).to(DEV).eval()
    sched = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
    tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer"); te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dt).to(DEV).eval()
    emb = te(tok([""], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEV))[0]; del te
    g = torch.Generator("cpu").manual_seed(0); eps = torch.randn((4, 4, 96, 96), generator=g).to(DEV)
    acp = sched.alphas_cumprod.to(DEV); feats = {}
    class Stop(Exception): pass
    def hk(m, i, o): feats["f"] = o; raise Stop()
    unet.up_blocks[1].register_forward_hook(hk)
    def fwd(x):
        B = x.shape[0]; xi = _resize(x, 768) * 2 - 1; z = vae.encode(xi).latent_dist.mean * vae.config.scaling_factor
        a = acp[261]; zn = (a.sqrt() * z[:, None] + (1 - a).sqrt() * eps[None]).reshape(B * 4, 4, 96, 96)
        try: unet(zn, torch.full((B * 4,), 261, device=DEV, dtype=torch.long), encoder_hidden_states=emb.expand(B * 4, -1, -1))
        except Stop: pass
        f = feats["f"].reshape(B, 4, *feats["f"].shape[1:]).mean(1)
        return {"t261_up1": f.permute(0, 2, 3, 1)}
    return Wrap(fwd, dict(cls="SD2.1 UNet+VAE (DIFT gate cost proxy)", params=sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in vae.parameters()), dtype="float32", input_px=768, draws=4, t=261))


def load_da2(wd, dtype=None, **kw):
    import transformers
    m = transformers.AutoModelForDepthEstimation.from_pretrained(str(wd), torch_dtype=dtype or torch.float32).eval().to(DEV)
    ms = pp_norm(wd)
    def fwd(x):
        fm = m.backbone.forward_with_filtered_kwargs(_norm(x, ms)).feature_maps     # 4 maps [B,C,H,W] (DINOv2-L, patch 14)
        f = [grid_tokens(t, 1) if t.ndim == 3 else t.permute(0, 2, 3, 1) for t in fm]
        return {"last": f[-1], "mid": f[1]}
    return Wrap(fwd, dict(cls=type(m).__name__ + ".backbone", params=sum(p.numel() for p in m.backbone.parameters()), dtype=str(next(m.parameters()).dtype), norm=str(ms)))

def registry():
    R = {}
    def reg(name, fn, *a, **k): R[name] = (fn, a, k)
    for n in ["c-radiov3-b", "c-radiov3-l", "c-radiov3-h", "c-radiov3-g", "c-radiov4-h", "c-radiov4-so400m"]: reg(n, load_radio, W / n)
    for n in PE_TIMM: reg(n, load_pe, n)
    reg("cleandift-sd21", load_cleandift, "sd21"); reg("cleandift-sd15", load_cleandift, "sd15")
    for n in ["siglip2-giant-opt-patch16-256", "siglip2-giant-opt-patch16-384", "dinov3-vits16", "dinov3-vits16plus", "dinov3-vitb16", "dinov3-vith16plus"]: reg(n, load_hf_vit, W / n)
    reg("dinov3-vit7b16", load_hf_vit, W / "dinov3-vit7b16", dtype=torch.bfloat16)
    for n in ["dinov3-convnext-tiny", "dinov3-convnext-small", "dinov3-convnext-base", "dinov3-convnext-large"]: reg(n, load_dinov3_convnext, W / n)
    reg("qwen2-vl-2b-instruct", load_qwen2vl_vision, W / "qwen2-vl-2b-instruct")
    reg("deepseek-ocr", load_deepseek_ocr, W / "deepseek-ocr"); reg("deepseek-ocr-2", load_deepseek_ocr, W / "deepseek-ocr-2")
    for n in ["dinov2-reg-large", "webssl-dino1b-224", "webssl-mae300m-224", "vit-mae-large", "aimv2-large-patch14-224"]: reg(n, load_generic_extra, W / n)
    for n in ["tipsv2-b14", "tipsv2-l14", "tipsv2-so400m14", "tipsv2-g14"]: reg(n, load_tips, W / n)
    reg("trocr-large-handwritten", load_trocr_encoder, W / "trocr-large-handwritten")
    reg("dinov3-vitl16", load_hf_vit, W1 / "dinov3-vitl16"); reg("dinov2-large", load_hf_vit, W1 / "dinov2-large"); reg("siglip2-so400m-patch14-224", load_hf_vit, W1 / "siglip2-so400m-patch14-224")
    reg("sd21-dift-gate", load_sd21_dift_gate); reg("depth-anything-v2-large", load_da2, W / "depth-anything-v2-large")
    reg("qaari-urdu-vision", load_qwen2vl_vision, W / "qwen2-vl-2b-instruct", adapter=str(W / "qaari-0.1-urdu-ocr"))
    reg("starvector-1b-im2svg", load_starvector, W / "starvector-1b-im2svg")
    for n in ["florence-2-large", "got-ocr2-hf", "paddleocr-vl-1.6", "lightonocr-2-1b", "unsloth-deepseek-ocr", "depth-anything-3-large-1.1"]: reg(n, load_instantiate_only, W / n)
    reg("tipsv2-g14", load_tips, W / "tipsv2-g14", dtype=torch.float32)
    return R

def _imgs(n, start=0):
    Z = np.load(VAL); a = Z["img224"][start:start + n]
    x = torch.from_numpy(a.astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
    return x

def env_rec():
    r = dict(code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), date_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), host=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"), torch=torch.__version__, device=DEV,
             gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None, python=platform.python_version())
    try:
        import transformers; r["transformers"] = transformers.__version__
    except Exception: pass
    return r

def cmd_load(names):
    R = registry(); (ADM / "loadtest").mkdir(parents=True, exist_ok=True)
    for n in names:
        rec = dict(name=n, env=env_rec()); t0 = time.time()
        try:
            fn, a, k = R[n]
            dt = None
            w = fn(*a, **k)
            rec["setup_s"] = round(time.time() - t0, 1)
            x = _imgs(4).to(DEV)
            o = w.fwd(x)
            rec["shapes_N4"] = {L: list(v.shape) for L, v in o.items()}
            rec["token_map_hw_at_224"] = {L: list(v.shape[1:3]) for L, v in o.items()}
            rec["finite"] = {L: bool(torch.isfinite(v.float()).all()) for L, v in o.items()}
            rec["info"] = w.info; rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error_type"] = type(e).__name__; rec["error"] = str(e)[:1500]; rec["tb_tail"] = traceback.format_exc()[-1800:]
        json.dump(rec, open(ADM / "loadtest" / f"{n}.json", "w"), indent=1, default=str)
        print(n, rec["status"], rec.get("token_map_hw_at_224") or rec.get("error_type"), flush=True)
        import gc; gc.collect()
        if DEV == "cuda": torch.cuda.empty_cache()

def cmd_time(tag, names, n_units=1000):
    R = registry(); (ADM / "timing").mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False     # strict fp32 as in the prereg extractors
    AC = os.environ.get("XS_AUTOCAST", "")
    X = _imgs(n_units)
    for n in names:
        out = ADM / "timing" / f"{tag}__{n}.json"
        if out.exists() and json.load(open(out)).get("status") == "ok": print("skip", n); continue
        rec = dict(name=n, tag=tag, env=env_rec(), n_units=n_units, autocast=os.environ.get("XS_AUTOCAST", "none"), tf32=False); t0 = time.time()
        try:
            fn, a, k = R[n]; w = fn(*a, **k); rec["setup_s"] = round(time.time() - t0, 1)
            bs = 32; done = False
            while not done:
                try:
                    _o = w.fwd(X[:min(bs, 4)].to(DEV)); torch.cuda.synchronize() if DEV == "cuda" else None
                    rec["token_map_hw_at_224"] = {L: list(v.shape[1:3]) for L, v in _o.items()}; rec["channels"] = {L: int(v.shape[-1]) for L, v in _o.items()}
                    t1 = time.time(); peak0 = 0
                    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
                    for b in range(0, n_units, bs):
                        if AC == "bf16":
                            with torch.autocast("cuda", dtype=torch.bfloat16): o = w.fwd(X[b:b + bs].to(DEV))
                        else: o = w.fwd(X[b:b + bs].to(DEV))
                        [v.float().cpu() for v in o.values()]
                    torch.cuda.synchronize() if DEV == "cuda" else None
                    el = time.time() - t1; done = True
                except torch.cuda.OutOfMemoryError:
                    bs //= 2; torch.cuda.empty_cache()
                    if bs < 1: raise
            rec["info"] = w.info
            rec.update(status="ok", batch_size=bs, extract_s=round(el, 2), units_per_s=round(n_units / el, 3),
                       peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else None, includes="forward + fp32 CPU copy of all returned layers; excludes weight load")
        except Exception as e:
            rec["status"] = "error"; rec["error_type"] = type(e).__name__; rec["error"] = str(e)[:1500]
        json.dump(rec, open(out, "w"), indent=1, default=str)
        print(n, rec["status"], rec.get("units_per_s") or rec.get("error_type"), flush=True)
        import gc; gc.collect()
        if DEV == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "load": cmd_load(sys.argv[2:])
    elif c == "time": cmd_time(sys.argv[2], sys.argv[3:])
    elif c == "list": print(" ".join(registry()))
