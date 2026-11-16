
p="$U/exp/FMD/xs/code/xs_probe_add.py"
s=open(p).read()
new = '''def load_florence(wd, dtype=None, **kw):
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
        if f.ndim == 3: f = grid_tokens(f)
        return {"last": f}
    return Wrap(fwd, dict(cls=type(m).__name__, vision_tower=type(vt).__name__, params=sum(p.numel() for p in vt.parameters()), dtype=str(next(vt.parameters()).dtype), norm="ImageNet mean/std", note="LOAD-TEST ONLY: DaViT tokens at 224 px input (native 768 px), pre-projection"))

def add_registry(R, reg):'''
s=s.replace("def add_registry(R, reg):", new, 1)
s=s.replace('    reg("starvector-8b-im2svg"', '    for n in ["florence-2-base", "florence-2-large-ft"]: reg(n, load_florence, A / n)\n    reg("starvector-8b-im2svg"', 1)
open(p,"w").write(s); print(s.count("load_florence"))
