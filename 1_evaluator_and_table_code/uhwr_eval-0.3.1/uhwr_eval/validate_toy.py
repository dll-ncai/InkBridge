"""Toy validation: reproduce metric_sanity_check.csv (DTW/LDTW/AIoU) through the uhwr_eval
wrapper and show that the order/direction metrics detect reversed/swapped strokes.
Toy glyph and perturbations are copied from the script that produced the CSV."""
import sys
import numpy as np
import pandas as pd
from uhwr_eval import score, from_normalized
from uhwr_eval.metrics import _sha_ok

CANVAS = (64, 64)

def circle(n=80, cw=False, r=0.45, c=(0, -0.2)):
    t = np.linspace(0, 2 * np.pi, n); t = -t if cw else t
    return np.c_[c[0] + r * np.sin(t), c[1] - r * np.cos(t)]
def bar(n=40, rtl=False):
    x = np.linspace(-0.5, 0.5, n); x = x[::-1] if rtl else x
    return np.c_[x, np.full(n, 0.5)]
def resample(s, k):
    t = np.linspace(0, 1, len(s)); tt = np.linspace(0, 1, len(s) * k)
    return np.c_[np.interp(tt, t, s[:, 0]), np.interp(tt, t, s[:, 1])]

def variants():
    rng = np.random.default_rng(0)
    C, B = circle(), bar()
    return [C, B], {
        "identical": [C, B],
        "loop direction reversed (CW)": [circle(cw=True), B],
        "stroke order swapped": [B, C],
        "bar direction reversed (RTL)": [C, bar(rtl=True)],
        "same path, 2x point density": [resample(C, 2), resample(B, 2)],
        "same path, 0.5x point density": [C[::2], B[::2]],
        "jitter sigma=1px": [C + rng.normal(0, 1 / 32, C.shape), B + rng.normal(0, 1 / 32, B.shape)],
        "shape drift (loop scaled 0.85)": [circle(r=0.45 * 0.85), B],
    }

def run(reference_csv=None, out_csv="toy_validation.csv"):
    assert _sha_ok(), "vendored Nagamatsu metrics.py does not match the recorded sha256"
    gt_n, V = variants()
    gt = from_normalized(gt_n, CANVAS)
    rows = []
    for name, v in V.items():
        pr = from_normalized(v, CANVAS)
        for prot in ("native", "standardised"):
            r = score(gt, pr, CANVAS, protocol=prot)
            rows.append({"variant": name, "protocol": prot, **r})
    df = pd.DataFrame(rows)
    if reference_csv:
        ref = pd.read_csv(reference_csv).set_index("variant")
        nat = df[df.protocol == "native"].set_index("variant")
        for m in ("dtw", "ldtw", "aiou"):
            diff = (nat[m].round(3) - ref[m]).abs()
            df.loc[df.protocol == "native", f"ref_{m}"] = df.loc[df.protocol == "native", "variant"].map(ref[m])
            assert (diff <= 1e-3 + 1e-9).all(), (m, diff.to_dict())
    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    df = run(sys.argv[1] if len(sys.argv) > 1 else None)
    cols = ["variant", "protocol", "dtw", "ldtw", "aiou", "strict_order_acc", "kendall_tau",
            "direction_acc", "loop_orient_acc", "penup_f1", "frechet", "chamfer", "lpips"]
    print(df[cols].round(3).to_string(index=False))
