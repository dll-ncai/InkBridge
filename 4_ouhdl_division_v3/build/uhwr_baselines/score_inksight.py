"""Score InkSight outputs (inksight_runner.py jsonl) on the declared ligature-unit subset.

Crops are unit canvases at f = 4 (256 x 256), so crop-pixel strokes / 4 = unit-canvas strokes.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "uhwr_eval_pkg")]
from uhwr_eval import score  # noqa: E402


def run(pkl, crop_dir, subsets=(("subset.json", "out_{c}.jsonl"), ("subset2.json", "out2_{c}.jsonl"))):
    lines = pickle.load(open(pkl, "rb"))
    rows = []
    for subf, outf in subsets:
        sub = json.load(open(os.path.join(crop_dir, subf)))
        draw = "seed0_n40" if subf == "subset.json" else "ext_seed1_n80"
        for c in ("rendered", "dataset"):
            p = os.path.join(crop_dir, outf.format(c=c))
            if not os.path.exists(p):
                continue
            out = {d["id"]: d for d in map(json.loads, open(p))}
            for s in sub:
                if s["unit_id"] not in out:
                    continue
                d = lines[s["line"]]
                u = d["line"].units[s["unit"]]
                o = out[s["unit_id"]]
                pred = [np.asarray(st, float) / 4.0 for st in o["strokes"] if len(st)]
                base = dict(sample_id=d["sample_id"], writer_id=d["writer_id"], unit="ligature",
                            unit_id=u["unit_id"], method="inksight_small_p", input=c, canvas_w=64,
                            runtime_s=o["secs"], prompt_used=o["prompt"],
                            inksight_recognized=o["recognized"], subset_draw=draw)
                for prot in ("native", "standardised"):
                    try:
                        r = score(u["ink"], pred, u["canvas"], protocol=prot)
                        rows.append({**base, "protocol": prot, **r, "error": ""})
                    except Exception as e:
                        rows.append({**base, "protocol": prot, "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)
