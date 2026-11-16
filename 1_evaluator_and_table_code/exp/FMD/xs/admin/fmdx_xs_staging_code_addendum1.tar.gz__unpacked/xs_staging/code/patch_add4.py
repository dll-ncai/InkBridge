
p="$U/exp/FMD/xs/code/xs_probe_add.py"
s=open(p).read()
s=s.replace('        if f.ndim == 3: f = grid_tokens(f)\n        return {"last": f}\n    return Wrap(fwd, dict(cls=type(m).__name__, vision_tower','        if f.ndim == 4 and f.shape[1] > f.shape[-1]: f = f.permute(0, 2, 3, 1)  # NCHW (native HF Florence2VisionBackbone) -> NHWC\n        if f.ndim == 3: f = grid_tokens(f)\n        return {"last": f}\n    return Wrap(fwd, dict(cls=type(m).__name__, vision_tower',1)
open(p,"w").write(s); print(s.count("NCHW (native"))
