"""PEN-Net inference: greedy autoregressive decoding (deterministic; no sampling) for the units of an e0_<split>.npz,
written in the E0 prediction format (preds (N, 320, 4) Nagamatsu space, unit_ids, meta) so that
experiments/e0/e0_score.py `e0` scores it unchanged.

python pennet_decode.py --weights RUN/milestones/ms_008.pt --npz e0_test.npz --condition rendered|dataset \
    [--subset 1000 --subset-seed 0] --out preds.npz
"""
import argparse
import json
import time

import numpy as np
import torch

from pennet_data import MAX_ROWS, image_tensor, load_split, rows_to_nagamatsu
from pennet_model import PENNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--condition", default="rendered")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-units", type=int)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-step", type=int, default=MAX_ROWS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.weights, map_location=dev, weights_only=False)
    net = PENNet().to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    d = load_split(a.npz, a.condition)
    n_all = len(d["unit_ids"])
    if a.subset is not None:   # identical subset rule to experiments/e0/e0_sample.py
        sel = np.sort(np.random.default_rng(a.subset_seed).choice(n_all, min(a.subset, n_all), replace=False))
    elif a.max_units is not None:
        sel = np.arange(min(a.max_units, n_all))
    else:
        sel = np.arange(n_all)
    imgs = image_tensor(d["cond"][sel])
    out, flags, lens, t0 = [], [], [], time.time()
    for b in range(0, len(sel), a.batch_size):
        x = imgs[b:b + a.batch_size].to(dev)
        rows, ln = net.decode_greedy(x, a.max_step)
        rows, ln = rows.cpu().numpy(), ln.cpu().numpy()
        for r, l in zip(rows, ln):
            arr, f = rows_to_nagamatsu(r, l)
            out.append(arr)
            flags.append(f)
            lens.append(int(l))
    preds = np.stack(out).astype(np.float32)
    meta = dict(weights=a.weights, step=int(ck.get("step", -1)), npz=a.npz, condition=a.condition, decoder="greedy",
                max_step=a.max_step, n_units=len(sel), subset=a.subset, subset_seed=a.subset_seed if a.subset else None,
                max_units=a.max_units, batch_size=a.batch_size,
                n_no_eos=int(sum(f["no_eos"] for f in flags)), n_empty=int(sum(f["empty"] for f in flags)),
                n_truncated=int(sum(f["truncated"] for f in flags)), mean_rows=float(np.mean(lens)),
                seconds=round(time.time() - t0, 1),
                device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    np.savez_compressed(a.out, preds=preds, unit_ids=d["unit_ids"][sel], emitted_rows=np.array(lens, np.int32),
                        meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
