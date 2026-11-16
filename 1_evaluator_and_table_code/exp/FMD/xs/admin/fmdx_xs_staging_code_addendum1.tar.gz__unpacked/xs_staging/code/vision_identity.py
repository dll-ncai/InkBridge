#!/usr/bin/env python3
"""Does the vision encoder of a fine-tuned VLM equal its base model's?  Tensor-by-tensor comparison of the vision-tower tensors (safetensors, mmap).  XS staging addendum."""
import json, glob, os, sys
from pathlib import Path
import torch
from safetensors import safe_open
W = Path("$U/exp/FMD/xs/weights"); W1 = Path("$U/exp/FMD/weights")
def tensors(d, marks):
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                for mk in marks:
                    if mk in k: out[k.split(mk, 1)[1] if mk.endswith(".") else k] = (str(f), k); break
    return out
PAIRS = [("qari-ocr-v0.3-vl-2b", W/"qari-ocr-v0.3-vl-2b", "qwen2-vl-2b-instruct", W/"qwen2-vl-2b-instruct", ["visual."]),
         ("arabic-english-hw-ocr-v3", W/"arabic-english-hw-ocr-v3", "qwen2.5-vl-3b-instruct", W/"qwen2.5-vl-3b-instruct", ["visual."]),
         ("arabic-hw-ocr-4bit", W/"arabic-hw-ocr-4bit", "qwen2.5-vl-3b-instruct", W/"qwen2.5-vl-3b-instruct", ["visual."]),
         ("chandra-ocr-2", W/"chandra-ocr-2", "qwen3.5-4b (v1)", W1/"qwen3.5-4b", ["visual."]),
         ("lightonocr-3-4b", W/"lightonocr-3-4b", "qwen3.5-4b (v1)", W1/"qwen3.5-4b", ["visual."]),
         ("chandra", W/"chandra", "qwen3-vl-8b-instruct (v1)", W1/"qwen3-vl-8b-instruct", ["visual."]),
         ("arabic-ocr-deepseek-ocr-2", W/"arabic-ocr-deepseek-ocr-2", "deepseek-ocr-2", W/"deepseek-ocr-2", ["sam_model.", "qwen2_model.", "projector."]),
         ("dots-ocr", W/"dots-ocr", "dots-ocr-base", W/"dots-ocr-base", ["vision_tower."])]
res = []
for a, da, b, db, marks in PAIRS:
    r = dict(model=a, base=b, marks=marks)
    try:
        ta, tb = tensors(da, marks), tensors(db, marks); common = sorted(set(ta) & set(tb)); r.update(n_a=len(ta), n_b=len(tb), n_common=len(common)); same = 0; maxd = 0.0; first = None; shape_mm = 0
        for k in common:
            with safe_open(ta[k][0], "pt") as fa, safe_open(tb[k][0], "pt") as fb:
                x, y = fa.get_tensor(ta[k][1]), fb.get_tensor(tb[k][1])
            if x.shape != y.shape: shape_mm += 1; continue
            if torch.equal(x, y): same += 1
            else:
                d = float((x.float() - y.float()).abs().max()); maxd = max(maxd, d); first = first or k
        r.update(n_identical=same, n_different=len(common) - same - shape_mm, n_shape_mismatch=shape_mm, max_abs_diff=maxd, first_different=first, status="ok")
    except Exception as e: r.update(status="error", error=type(e).__name__ + ": " + str(e)[:200])
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open("$U/exp/FMD/xs/admin/vision_identity_addendum1.json", "w"), indent=1)
