# xs_probe_add.py : executed inside xs_probe.py globals (addendum 1, 3 Nov 2026).  LOAD-TEST loaders only; not the pre-registered extractors.
PE_TIMM.update({"pe-core-g14-448": ("pe-core-g14-448-timm", "vit_pe_core_gigantic_patch14_448.fb"), "pe-lang-l14-448": ("pe-lang-l14-448-timm", "vit_pe_lang_large_patch14_448.fb"),
                "pe-lang-g14-448": ("pe-lang-g14-448-timm", "vit_pe_lang_gigantic_patch14_448.fb")})

def _qwenvl_vis(wd, dtype, adapter):
    import transformers
    full = None; errs = []
    for cls in ("AutoModelForImageTextToText", "AutoModel"):
        try: full = getattr(transformers, cls).from_pretrained(str(wd), torch_dtype=dtype or torch.bfloat16); break
        except Exception as e: errs.append(f"{cls}: {type(e).__name__}: {str(e)[:150]}")
    if full is None: raise RuntimeError(" | ".join(errs))
    inner = getattr(full, "model", full)
    vis = getattr(inner, "visual", None) or getattr(full, "visual", None)
    if vis is None: raise AttributeError("no .visual in " + type(full).__name__ + " / " + type(inner).__name__)
    merged_info = None
    if adapter:
        from safetensors.torch import load_file
        ad = load_file(str(Path(adapter) / "adapter_model.safetensors")); acfg = json.load(open(Path(adapter) / "adapter_config.json")); sc = acfg["lora_alpha"] / acfg["r"]; n = 0
        for k in list(ad):
            if ".visual." in k and k.endswith("lora_A.weight"):
                mod = k.split("visual.", 1)[1][:-len(".lora_A.weight")]; B = ad[k.replace("lora_A", "lora_B")].float(); A = ad[k].float(); tgt = vis.get_submodule(mod)
                tgt.weight.data = (tgt.weight.data.float() + sc * (B @ A)).to(tgt.weight.dtype); n += 1
        merged_info = dict(n_visual_lora_modules_merged=n, scale=sc, adapter=str(adapter))
    return full, vis, merged_info

def load_qwenvl_vision(wd, dtype=None, adapter=None, **kw):
    """Qwen2-VL / Qwen3-VL / Qwen3.5-VL family vision tower (+ optional visual-LoRA merge).  224 px -> patch grid, merged grid."""
    import transformers
    from PIL import Image
    wd = Path(wd); full, vis, merged_info = _qwenvl_vis(wd, dtype, adapter); vis = vis.eval().to(DEV); cfgj = json.load(open(wd / "config.json")); del full
    ip = None; ierr = []
    for src in (wd, kw.get("proc_dir")):
        if src is None: continue
        for mk in (lambda s: transformers.AutoImageProcessor.from_pretrained(str(s)), lambda s: transformers.AutoProcessor.from_pretrained(str(s)).image_processor):
            try: ip = mk(src); break
            except Exception as e: ierr.append(str(e)[:120])
        if ip is not None: break
    if ip is None: raise RuntimeError("no image processor: " + " | ".join(ierr))
    store = {}
    def fwd(x):
        arr = (x.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8); imgs = [Image.fromarray(a) for a in arr]
        try: pi = ip(images=imgs, return_tensors="pt", size={"shortest_edge": 224 * 224, "longest_edge": 224 * 224})
        except Exception: pi = ip(images=imgs, return_tensors="pt", min_pixels=224 * 224, max_pixels=224 * 224)
        pv, thw = pi["pixel_values"].to(DEV, next(vis.parameters()).dtype), pi["image_grid_thw"].to(DEV)
        vo = vis(pv, grid_thw=thw); n = x.shape[0]; h, w = int(thw[0, 1]), int(thw[0, 2])
        pre = getattr(vo, "last_hidden_state", None); mg = getattr(vo, "pooler_output", None) if hasattr(vo, "pooler_output") else vo
        if isinstance(vo, (tuple, list)): mg = vo[0]
        out = {}
        if pre is not None and pre.shape[0] == n * h * w: out["pre_merge"] = pre.reshape(n, h, w, -1)
        out["last"] = mg.reshape(n, h // 2, w // 2, -1)
        return out
    return Wrap(fwd, dict(cls=type(vis).__name__, params=sum(p.numel() for p in vis.parameters()), dtype=str(next(vis.parameters()).dtype), input_px=224, patch_size=cfgj.get("vision_config", {}).get("patch_size"), lora_merge=merged_info,
                          note="vision tower only; 224 px -> patch grid (pre_merge) and 2x2-merged grid (last)"))

def load_starvector8b(wd, dtype=None, **kw):
    import transformers
    from safetensors import safe_open
    cfgd = Path("$U/exp/FMD/xs/weights/siglip-large-patch16-384-cfg")
    vc = transformers.SiglipVisionConfig.from_pretrained(str(cfgd)); m = transformers.SiglipVisionModel(vc)
    sd = {}
    for f in sorted(Path(wd).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith("model.image_encoder.visual_encoder."): sd[k[len("model.image_encoder.visual_encoder."):]] = sf.get_tensor(k)
    sd = {(k if k.startswith("vision_model.") else "vision_model." + k): v for k, v in sd.items()}
    r = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); m = m.eval().to(DEV)
    ms = ((0.5,) * 3, (0.5,) * 3); st = {}
    def fwd(x):
        for px, ip in ((224, True), (384, False)):
            try:
                o = m(pixel_values=_norm(_resize(x, px), ms), interpolate_pos_encoding=ip); st["m"] = (px, ip); break
            except Exception as e: err = e; o = None
        if o is None: raise err
        return {"last": grid_tokens(o.last_hidden_state, 0)}
    return Wrap(fwd, dict(cls="SiglipVisionModel (google/siglip-large-patch16-384 config) with starvector-8b weights", params=sum(p.numel() for p in m.parameters()), dtype="float32", missing=len(r.missing_keys), unexpected=len(r.unexpected_keys), note="weights = model.image_encoder.visual_encoder.* of starvector-8b-im2svg; post_layernorm/head keys may be absent"))

def load_instantiate2(wd, dtype=None, **kw):
    import transformers
    wd = Path(wd); err = []
    for trc in (False, True):
        for cls in ("AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                m = getattr(transformers, cls).from_pretrained(str(wd), trust_remote_code=trc, torch_dtype=dtype or torch.bfloat16); n = sum(p.numel() for p in m.parameters())
                def fwd(x): raise NotImplementedError(f"instantiate-only: {type(m).__name__} ({cls}, trust_remote_code={trc}) loaded, {n} params; no feature-extraction loader written")
                return Wrap(fwd, dict(cls=type(m).__name__, auto_class=cls, trust_remote_code=trc, params=n, dtype=str(next(m.parameters()).dtype), load_only=True))
            except Exception as e: err.append(f"{cls}/trc={trc}: {type(e).__name__}: {str(e)[:140]}")
    raise RuntimeError(" | ".join(err))

def _vae_enc(vae, x, res):
    z = vae.encode(_resize(x, res) * 2 - 1).latent_dist.mean
    return (z - (vae.config.shift_factor or 0.0)) * vae.config.scaling_factor

def load_sd3_mmdit(wd, dtype=None, sigma=0.25, res=512, **kw):
    """SD3 / SD3.5 MMDiT: VAE-encode the 224 render resized to res, flow-matching noise at sigma, empty prompt (3 text encoders on CPU), image-token output of block mid and last.  LOAD-TEST convention."""
    from diffusers import StableDiffusion3Pipeline
    dt = dtype or torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(str(wd), torch_dtype=dt)
    with torch.no_grad():
        pe, npe, ppe, nppe = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False)
    tr, vae = pipe.transformer.to(DEV).eval(), pipe.vae.to(DEV).eval(); pe, ppe = pe.to(DEV, dt), ppe.to(DEV, dt)
    for a in ("text_encoder", "text_encoder_2", "text_encoder_3"): setattr(pipe, a, None)
    import gc; gc.collect()
    nb = len(tr.transformer_blocks); feats = {}
    tr.transformer_blocks[nb // 2].register_forward_hook(lambda m, i, o: feats.__setitem__("mid", o[1] if isinstance(o, tuple) and o[1].ndim == 3 and o[1].shape[1] > 100 else o[0]))
    tr.transformer_blocks[nb - 1].register_forward_hook(lambda m, i, o: feats.__setitem__("last", o[1] if isinstance(o, tuple) and o[1].ndim == 3 and o[1].shape[1] > 100 else o[0]))
    g = torch.Generator("cpu").manual_seed(0); st = {}
    def fwd(x):
        B = x.shape[0]; z = _vae_enc(vae, x.to(dt), res).to(dt)
        if "eps" not in st: st["eps"] = torch.randn((1,) + z.shape[1:], generator=g).to(DEV, dt)
        zt = (1 - sigma) * z + sigma * st["eps"]
        tr(hidden_states=zt, timestep=torch.full((B,), sigma * 1000, device=DEV, dtype=dt), encoder_hidden_states=pe.expand(B, -1, -1), pooled_projections=ppe.expand(B, -1), return_dict=False)
        n = int((feats["mid"].shape[1]) ** 0.5)
        return {k: v.reshape(B, n, n, -1) for k, v in feats.items()}
    return Wrap(fwd, dict(cls=type(tr).__name__, params=sum(p.numel() for p in tr.parameters()) + sum(p.numel() for p in vae.parameters()), dtype=str(dt), input_px=res, sigma=sigma, n_blocks=nb, convention="LOAD-TEST ONLY: VAE latent (shift/scale), flow noise sigma=0.25 fixed eps, empty prompt, full transformer forward; mid = block nb//2 image tokens, last = last block image tokens"))

def load_flux_dit(wd, dtype=None, sigma=0.25, res=512, **kw):
    from diffusers import FluxPipeline
    dt = dtype or torch.bfloat16
    pipe = FluxPipeline.from_pretrained(str(wd), torch_dtype=dt)
    with torch.no_grad(): pe, ppe, tids = pipe.encode_prompt(prompt="", prompt_2=None, device="cpu", num_images_per_prompt=1, max_sequence_length=64)
    tr, vae = pipe.transformer.to(DEV).eval(), pipe.vae.to(DEV).eval(); pe, ppe, tids = pe.to(DEV, dt), ppe.to(DEV, dt), tids.to(DEV)
    for a in ("text_encoder", "text_encoder_2"): setattr(pipe, a, None)
    import gc; gc.collect()
    nb = len(tr.transformer_blocks); feats = {}
    def grab(name):
        def h(m, i, o): feats[name] = o[1] if isinstance(o, tuple) and o[1].shape[1] != pe.shape[1] else o[0]
        return h
    tr.transformer_blocks[nb // 2].register_forward_hook(grab("mid")); tr.transformer_blocks[nb - 1].register_forward_hook(grab("last_double"))
    g = torch.Generator("cpu").manual_seed(0); st = {}
    guid = bool(getattr(tr.config, "guidance_embeds", False))
    def fwd(x):
        B = x.shape[0]; z = _vae_enc(vae, x.to(dt), res).to(dt); C, h, w = z.shape[1:]
        if "eps" not in st: st["eps"] = torch.randn((1,) + z.shape[1:], generator=g).to(DEV, dt)
        zt = (1 - sigma) * z + sigma * st["eps"]
        pk = zt.reshape(B, C, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (h // 2) * (w // 2), C * 4)
        ids = torch.zeros(h // 2, w // 2, 3, device=DEV, dtype=dt); ids[..., 1] = torch.arange(h // 2, device=DEV)[:, None]; ids[..., 2] = torch.arange(w // 2, device=DEV)[None]; ids = ids.reshape(-1, 3)
        kwargs = dict(hidden_states=pk, timestep=torch.full((B,), sigma, device=DEV, dtype=dt), encoder_hidden_states=pe.expand(B, -1, -1), pooled_projections=ppe.expand(B, -1), txt_ids=tids, img_ids=ids, return_dict=False)
        if guid: kwargs["guidance"] = torch.full((B,), 3.5, device=DEV, dtype=dt)
        tr(**kwargs); n = h // 2
        return {k: v.reshape(B, n, n, -1) for k, v in feats.items()}
    return Wrap(fwd, dict(cls=type(tr).__name__, params=sum(p.numel() for p in tr.parameters()) + sum(p.numel() for p in vae.parameters()), dtype=str(dt), input_px=res, sigma=sigma, n_double_blocks=nb, guidance_embeds=guid,
                          convention="LOAD-TEST ONLY: VAE latent, rectified-flow noise sigma=0.25, empty prompt (T5 max 64 tok), full forward; mid = double-stream block nb//2 image tokens"))

def load_sam3_vision(wd, dtype=None, **kw):
    import transformers
    wd = Path(wd); err = []; m = None
    for cls in ("Sam3Model", "AutoModel"):
        try: m = getattr(transformers, cls).from_pretrained(str(wd), torch_dtype=dtype or torch.float32); break
        except Exception as e: err.append(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
    if m is None: raise RuntimeError(" | ".join(err))
    enc = getattr(m, "vision_encoder", None) or getattr(m, "vision_model", None)
    if enc is None: raise AttributeError("no vision encoder attr; children=" + str([n for n, _ in m.named_children()]))
    enc = enc.eval().to(DEV); ms = pp_norm(wd, HALF); st = {}
    def fwd(x):
        for px in ([224, 1008] if "px" not in st else [st["px"]]):
            try:
                o = enc(pixel_values=_norm(_resize(x, px), ms).to(next(enc.parameters()).dtype)); st["px"] = px; break
            except Exception as e: err_ = e; o = None
        if o is None: raise err_
        f = getattr(o, "last_hidden_state", o[0] if isinstance(o, (tuple, list)) else o)
        if isinstance(f, (tuple, list)): f = f[-1]
        if f.ndim == 3: f = grid_tokens(f)
        elif f.shape[1] > f.shape[-1]: f = f
        else: f = f.permute(0, 2, 3, 1)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(enc).__name__, params=sum(p.numel() for p in enc.parameters()), dtype=str(next(enc.parameters()).dtype), input_mode=str(st), norm=str(ms)))

def load_theia(wd, dtype=None, **kw):
    """Theia (DeiT backbone, remote code).  needs the DeiT config in the HF cache (HF_HOME=xs/weights/.hf_home, prefetched by prefetch_deit.py)."""
    import transformers
    m = transformers.AutoModel.from_pretrained(str(wd), trust_remote_code=True, torch_dtype=dtype or torch.float32).eval().to(DEV)
    def fwd(x):
        u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
        f = m.forward_feature(u8, do_resize=False, interpolate_pos_encoding=False)
        if f.ndim == 3: f = grid_tokens(f)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(m).__name__, params=sum(p.numel() for p in m.parameters()), dtype=str(next(m.parameters()).dtype), backbone=getattr(m.config, "backbone", None), norm="HF preprocessor of the DeiT backbone (do_normalize) inside forward_feature"))

def load_florence(wd, dtype=None, **kw):
    """Florence-2 DaViT vision tower (pre-projection tokens).  LOAD-TEST convention: 224-px input (native 768), ImageNet mean/std, DaViT forward_features_unpool if present."""
    import transformers
    errs = []; m = None
    for trc in (False, True):
        for cls in (transformers.AutoModel, transformers.AutoModelForCausalLM):
            try: m = cls.from_pretrained(str(wd), trust_remote_code=trc, torch_dtype=dtype or torch.float32); break
            except Exception as e: errs.append(f"{cls.__name__}/trc={trc}: {type(e).__name__}: {str(e)[:150]}")
        if m is not None: break
    if m is None: raise RuntimeError(" | ".join(errs))
    m = m.eval().to(DEV)
    vt = getattr(m, "vision_tower", None) or getattr(getattr(m, "model", None), "vision_tower", None)
    if vt is None: raise RuntimeError("no vision_tower attr on " + type(m).__name__)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    def fwd(x):
        x = ((x - mean) / std).to(next(vt.parameters()).dtype)
        f = vt.forward_features_unpool(x) if hasattr(vt, "forward_features_unpool") else vt(x)
        if isinstance(f, (tuple, list)): f = f[0]
        if hasattr(f, "last_hidden_state"): f = f.last_hidden_state
        if f.ndim == 4 and f.shape[1] > f.shape[-1]: f = f.permute(0, 2, 3, 1)  # NCHW (native HF Florence2VisionBackbone) -> NHWC
        if f.ndim == 3: f = grid_tokens(f)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(m).__name__, vision_tower=type(vt).__name__, params=sum(p.numel() for p in vt.parameters()), dtype=str(next(vt.parameters()).dtype), norm="ImageNet mean/std", note="LOAD-TEST ONLY: DaViT tokens at 224 px input (native 768 px), pre-projection"))

def add_registry(R, reg):
    A = W
    for n in ["pe-core-g14-448", "pe-lang-l14-448", "pe-lang-g14-448"]: reg(n, load_pe, n)
    for n in ["florence-2-base", "florence-2-large-ft"]: reg(n, load_florence, A / n)
    reg("starvector-8b-im2svg", load_starvector8b, A / "starvector-8b-im2svg")
    for n, kw in [("webssl-dino300m-224", {}), ("webssl-dino2b-224", {}), ("webssl-dino3b-224", dict(dtype=torch.bfloat16)), ("webssl-dino5b-224", dict(dtype=torch.bfloat16)), ("webssl-dino7b-224", dict(dtype=torch.bfloat16)),
                  ("webssl-mae1b-224", {}), ("webssl-mae3b-224", dict(dtype=torch.bfloat16)), ("dinov2-reg-small", {}), ("dinov2-reg-base", {}), ("dinov2-reg-giant", {}), ("dinov2-giant", {}), ("dinov2-base", {}),
                  ("aimv2-huge-patch14-224", {}), ("aimv2-1b-patch14-224", {}), ("aimv2-3b-patch14-224", dict(dtype=torch.bfloat16)), ("aimv2-large-patch14-224-distilled", {}),
                  ("dinov3-vitl16-sat493m", {}), ("dinov3-vit7b16-sat493m", dict(dtype=torch.bfloat16)), ("siglip2-large-patch16-256", {}), ("siglip2-so400m-patch16-256", {})]: reg(n, load_hf_vit, A / n, **kw)
    for n in ["tipsv1-s14", "tipsv1-b14", "tipsv1-l14", "tipsv1-so400m14", "tipsv1-g14"]: reg(n, load_tips, A / n)
    for n in ["trocr-base-handwritten", "trocr-small-handwritten", "trocr-base-hw-hist-swe-2"]: reg(n, load_trocr_encoder, A / n)
    for n in ["chandra-ocr-2", "chandra", "lightonocr-3-4b", "lightonocr-3-0.8b", "qari-ocr-v0.3-vl-2b", "arabic-english-hw-ocr-v3", "arabic-hw-ocr-4bit", "surya-ocr-2-persian-hw"]: reg(n, load_qwenvl_vision, A / n)
    reg("qari-ocr-0.4.0-vl-4b-vision", load_qwenvl_vision, A / "qwen3-vl-4b-instruct", adapter=str(A / "qari-ocr-0.4.0-vl-4b"))
    reg("qwen3-vl-4b-instruct", load_qwenvl_vision, A / "qwen3-vl-4b-instruct"); reg("qwen2.5-vl-3b-instruct", load_qwenvl_vision, A / "qwen2.5-vl-3b-instruct")
    for n in ["dots-ocr", "dots-ocr-base", "paddleocr-vl-1.5", "paddleocr-vl-1.0", "arabic-ocr-deepseek-ocr-2", "depth-anything-3-giant-1.1", "depth-anything-3-base"]: reg(n, load_instantiate2, A / n)
    for n in ["sd3-medium-diffusers", "sd3.5-medium-diffusers-layout", "sd3.5-large-diffusers-layout", "sd3.5-large-turbo-diffusers-layout"]: reg(n, load_sd3_mmdit, A / n)
    reg("flux1-dev", load_flux_dit, A / "flux1-dev"); reg("flux1-schnell", load_flux_dit, A / "flux1-schnell")
    for n in ["theia-tiny-cddsv", "theia-tiny-cdiv", "theia-small-cdiv", "theia-base-cdiv", "theia-base-cddsv"]: reg(n, load_theia, A / n)
    reg("sam3", load_sam3_vision, A / "sam3"); reg("sam3.1", load_sam3_vision, A / "sam3.1")
