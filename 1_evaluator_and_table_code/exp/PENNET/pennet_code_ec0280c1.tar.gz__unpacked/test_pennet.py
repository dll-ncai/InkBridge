"""Unit tests for the PEN-Net re-implementation (run in the smoke gate; exit code != 0 on failure).
python test_pennet.py --npz-dir DATA --out tests.json
 1. soft-DTW batch vs naive float64 reference: values and gradients, variable lengths, gamma 0.1 and 1.
 2. model shapes: CRNN_x -> (n,256,1,33), CRNN_y -> (n,256,33,1), attention over 66 features, output (n,T,5).
    Parameter count recorded.
 3. label round trip on every unit of val/test/gold: E0 seq -> PEN-Net rows -> Nagamatsu array -> e0_adapter.decode()
    equals e0_adapter.decode(seq) (stroke count and points, atol 1e-5 px).
 4. image convention: max of the input tensor sits on ink pixels (white ink on black).
"""
import argparse
import json
import sys

import numpy as np
import torch

from pennet_data import build_label_tensor, image_tensor, load_split, nagamatsu_to_rows, rows_to_nagamatsu
from pennet_model import PENNet
from pennet_softdtw import soft_dtw_batch, soft_dtw_reference


def t_softdtw(dev):
    res = []
    g = torch.Generator().manual_seed(0)
    for scale, gamma in [(1.0, 1.0), (64.0, 0.1)]:
        B, N, M = 6, 12, 10
        x = (torch.rand(B, N, 2, generator=g, dtype=torch.float64) * scale)
        y = (torch.rand(B, M, 2, generator=g, dtype=torch.float64) * scale)
        n = torch.tensor([12, 3, 7, 1, 5, 12])
        m = torch.tensor([10, 4, 2, 6, 10, 1])
        xd, yd = x.clone().to(dev).requires_grad_(), y.clone().to(dev).requires_grad_()
        w = torch.randn(B, generator=g, dtype=torch.float64)
        v = soft_dtw_batch(xd, yd, n.to(dev), m.to(dev), gamma)
        (v * w.to(dev)).sum().backward()
        ref, gx, gy = [], torch.zeros_like(x), torch.zeros_like(y)
        for b in range(B):
            xb = x[b, :n[b]].clone().requires_grad_()
            yb = y[b, :m[b]].clone().requires_grad_()
            r = soft_dtw_reference(xb, yb, gamma)
            (r * w[b]).backward()
            ref.append(r.item())
            gx[b, :n[b]], gy[b, :m[b]] = xb.grad, yb.grad
        ref = torch.tensor(ref, dtype=torch.float64)
        dv = float((v.detach().cpu() - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
        dgx = float((xd.grad.cpu() - gx).abs().max() / gx.abs().max().clamp_min(1e-12))
        dgy = float((yd.grad.cpu() - gy).abs().max() / gy.abs().max().clamp_min(1e-12))
        res.append(dict(scale=scale, gamma=gamma, rel_val=dv, rel_grad_x=dgx, rel_grad_y=dgy,
                        ok=bool(dv < 1e-9 and dgx < 1e-7 and dgy < 1e-7)))
    # float32 on the training scale: finite values/gradients for full-length sequences
    x = (torch.rand(8, 321, 2, generator=g) * 64).to(dev)
    y = (torch.rand(8, 321, 2, generator=g) * 64).to(dev).requires_grad_()
    v = soft_dtw_batch(x, y, torch.full((8,), 321, device=dev), torch.full((8,), 300, device=dev), 0.1)
    v.sum().backward()
    res.append(dict(float32_321_finite=bool(torch.isfinite(v).all() and torch.isfinite(y.grad).all()),
                    ok=bool(torch.isfinite(v).all() and torch.isfinite(y.grad).all())))
    return res


def t_model(dev):
    net = PENNet().to(dev)
    img = torch.zeros(4, 1, 64, 64, device=dev)
    s1, s2 = tuple(net.cnn1(img).shape), tuple(net.cnn2(img).shape)
    z, hn, cn, w = net.encode(img)
    out = net(img, torch.zeros(4, 17, 5, device=dev))
    net.eval()
    rows, ln = net.decode_greedy(img, 5)
    ok = s1 == (4, 256, 1, 33) and s2 == (4, 256, 33, 1) and tuple(w.shape) == (4, 66, 1) and tuple(out.shape) == (4, 17, 5) \
        and tuple(hn.shape) == (3, 4, 256) and tuple(rows.shape) == (4, 5, 5)
    return dict(cnn1=s1, cnn2=s2, attn=tuple(w.shape), out=tuple(out.shape), n_params=sum(p.numel() for p in net.parameters()),
                ok=bool(ok))


def t_roundtrip(npz_dir):
    sys.path.insert(0, ".")
    import e0_adapter as A
    res = {}
    for split in ("val", "test", "gold"):
        d = load_split(f"{npz_dir}/e0_{split}.npz", "rendered")
        off = d["off"]
        bad, maxerr = 0, 0.0
        lab, lens = build_label_tensor(d)
        for k in range(len(off) - 1):
            seq = d["seq"][off[k]:off[k + 1]]
            arr, f = rows_to_nagamatsu(lab[k].numpy(), int(lens[k]))
            g, p = A.decode(seq), A.decode(arr)
            if len(g) != len(p) or any(len(a) != len(b) for a, b in zip(g, p)):
                bad += 1
                continue
            e = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(g, p))
            maxerr = max(maxerr, e)
            if e > 1e-5 or f["no_eos"] or f["empty"] or f["truncated"]:
                bad += 1
        img = image_tensor(d["cond"][:50])
        ink = d["cond"][:50] < 128
        ok_img = bool((img.numpy()[:, 0][ink] > 0).mean() > 0.99 and (img.numpy()[:, 0][~ink] < 0).mean() > 0.9)
        res[split] = dict(n=len(off) - 1, bad=bad, max_abs_err_px=maxerr, image_white_on_black=ok_img,
                          ok=bool(bad == 0 and ok_img))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = dict(device=str(dev), softdtw=t_softdtw(dev), model=t_model(dev), roundtrip=t_roundtrip(a.npz_dir))
    ok = all(x["ok"] for x in r["softdtw"]) and r["model"]["ok"] and all(v["ok"] for v in r["roundtrip"].values())
    r["all_ok"] = bool(ok)
    json.dump(r, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(r, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
