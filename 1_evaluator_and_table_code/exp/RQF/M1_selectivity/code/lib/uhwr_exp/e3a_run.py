"""E3a: recovered OUHD-L test-line trajectories fed to Momna's fixed BiLSTM-CTC recogniser.

python -m uhwr_exp.e3a_run --bundle bundle --pred out/pred --momna ext/momna --out out/e3a

Recognisers (dll-<site>/Synthetic-Stroke-Trajectory checkpoints, unchanged):
  finetuned_3way        recognizer_finetuned_3way.pt (synthetic pretrain + OUHD-L fine-tune;
                        trained with STROKE_SIMPLIFY on)
  pretrained_synthetic  recognizer_pretrained_synthetic.pt (synthetic UPTI only; never saw
                        OUHD-L; STROKE_SIMPLIFY on)
  realonly_baseline     recognizer_realonly_baseline.pt (OUHD-L only; trained on RAW
                        trajectories, so STROKE_SIMPLIFY off)
CONTAMINATION: finetuned_3way and realonly_baseline were trained on the repo's own random
line-level 80/10/10 split (seed 0) of all 2,403 OUHD-L lines, not on the Zenodo writer-disjoint
split; that split file is not shipped, so roughly 80% of our 253 test lines are expected to be
in their training data.  pretrained_synthetic is the only uncontaminated recogniser.
Inputs per line:
  GT_exact     tokens built from the raw CSV by the repo's OnlineRecogDataset (bundle)
               = the verify_results.py preprocessing (distance thinning, z-score, simplify, cap)
  GT_strokes   the GT pen-down strokes passed through the Path-B tokenizer route below (checks
               that the route itself costs nothing)
  <method>|<input>  recovered strokes (eval px) for geo_rtl, geo_rtl_local, momna_s123,
               momna_s123_subpaths at the reference input and at dataset img/
Path-B route (as the repo applies to recovered trajectories): path_b_polylines_to_tokens
(per-line z-score, pen_state = 1 on the last point of each stroke) -> simplify_trajectory
(target_step 0.117, sigma 0.6; skipped for realonly) -> drop_near_duplicates -> cap 1024.
Greedy CTC decoding; CER = edit distance over vocabulary ids / reference length, references
encoded with the repo's encode_text normalisation.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

CKPTS = {"finetuned_3way": ("recognizer_finetuned_3way.pt", True),
         "pretrained_synthetic": ("recognizer_pretrained_synthetic.pt", True),
         "realonly_baseline": ("recognizer_realonly_baseline.pt", False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--momna", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = os.path.abspath(a.momna)
    for sub in ["", "stage4_recognizer"]:
        sys.path.insert(0, os.path.join(M, sub))
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("UHWR_TORCH_THREADS", "8")))
    from vocab_loader import load_vocab, encode_text
    from recognizer import TrajRecognizer
    from traj_tokenizer import (path_b_polylines_to_tokens, simplify_trajectory,
                                drop_near_duplicates, cap_sequence_length)
    from uhwr_exp.common import load_lines, line_sample
    from uhwr_exp.e2_score import REF
    import pandas as pd
    c2i = load_vocab(os.path.join(M, "checkpoints", "vocab_minimal.json"))
    V = len(c2i) + 1
    lines = load_lines(a.bundle)
    gtok = pickle.load(open(os.path.join(a.bundle, "momna_gt_tokens.pkl"), "rb"))
    gtok_raw = pickle.load(open(os.path.join(a.bundle, "momna_gt_tokens_raw.pkl"), "rb"))
    sids = sorted(f[5:-4] for f in os.listdir(a.pred) if f.startswith("pred_"))
    if a.limit:
        sids = sids[: a.limit]

    def toks(strokes, simplify):
        t = path_b_polylines_to_tokens([[np.asarray(s, np.float32)] for s in strokes if len(s)])
        if t.shape[0] == 0:
            return t
        if simplify:
            t = simplify_trajectory(t, target_step=0.117, smooth_sigma=0.6)
        return cap_sequence_length(drop_near_duplicates(t), max_T=1024)

    def ed(x, y):
        if not x: return len(y)
        if not y: return len(x)
        prev = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i]
            for j, cy in enumerate(y, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
            prev = cur
        return prev[-1]

    t0 = time.time()
    items = []  # (sid, source, tokens per simplify flag)
    for sid in sids:
        d = lines[sid]
        L = line_sample(d, sid)
        P = pickle.load(open(os.path.join(a.pred, f"pred_{sid}.pkl"), "rb"))
        srcs = {"GT_strokes": L.ink}
        for meth in ("geo_rtl", "geo_rtl_local", "momna_s123", "momna_s123_subpaths"):
            for cond in (REF[meth], "dataset"):
                k = ("line", meth, cond, sid)
                if k in P:
                    srcs[f"{meth}|{cond}"] = P[k]
        for name, st in srcs.items():
            items.append((sid, name, toks(st, True), toks(st, False)))
        items.append((sid, "GT_exact", gtok[sid][0], gtok_raw[sid][0]))
    rows = []
    for rec, (fn, simp) in CKPTS.items():
        model = TrajRecognizer(vocab_size_with_blank=V, input_dim=3)
        ck = torch.load(os.path.join(M, "checkpoints", fn), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for sid, name, ts, tr in items:
            x = ts if simp else tr
            ref = encode_text(str(lines[sid]["text"]).strip(), c2i)
            if x.shape[0] == 0:
                hyp = []
            else:
                with torch.no_grad():
                    lg = model(torch.from_numpy(x).float()[None], torch.tensor([x.shape[0]]))
                    am = F.log_softmax(lg, -1)[0].argmax(-1).tolist()
                hyp, prev = [], None
                for tk in am:
                    if tk != prev and tk != 0:
                        hyp.append(tk)
                    prev = tk
            e = ed(hyp, ref)
            meth, _, cond = name.partition("|")
            rows.append(dict(sample_id=sid, writer_id=lines[sid]["writer_id"], recogniser=rec,
                             source=meth, input=cond or "-", n_tokens=int(x.shape[0]),
                             edits=e, ref_len=len(ref), cer=e / max(len(ref), 1)))
        print(rec, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "e3a_cer_per_line.csv"), index=False)
    s = df.groupby(["recogniser", "source", "input"]).apply(
        lambda g: pd.Series(dict(n_lines=len(g), cer_micro=g.edits.sum() / g.ref_len.sum(),
                                 cer_macro=g.cer.mean()))).reset_index()
    s.to_csv(os.path.join(a.out, "e3a_cer_summary.csv"), index=False)
    print(s.to_string())


if __name__ == "__main__":
    main()
