"""ORDER loop, lane Repair: GT-free per-unit verifier (minimum-Bayes-risk selection) among candidate samples.
For each unit, candidates = the same unit in every --pool preds file (e.g. the 3 sampling seeds of one arm). Utility between two
candidates = F1u + F1o on the 64 px canvas: F1u = pen-up endpoint agreement (orientation/order-invariant stroke end-point pairs; cost =
min over orientation of the max end-point distance; Hungarian; agree if <= 3 px; F1 = 2 TP / (n_a + n_b)); F1o = same with the
oriented cost max(|start-start|, |end-end|) (direction agreement). Selected = argmax over candidates
of the mean utility vs the other candidates; ties -> higher input-ink F (pred points on input ink, 1 px dilation, vs input ink pixels
covered by pred points dilated 1 px); then the lower pool index. No GT is read (input image = model condition).
Every --target file is written with each unit's row array replaced by the selected candidate (rows copied bit-exact).
--dir-vote: after selection, reverse each selected stroke when a strict majority of the other candidates' end-point-matched
strokes run the opposite way (direction consensus; coordinates bit-exact, only point order inside strokes changes).
--dir-model NPZ [--dir-thr 0.95]: after selection, reverse a selected stroke when the p01-trained direction scorer
(dirscorer_train.py) prefers the reversed orientation by more than thr (p(rev) - p(as is) > thr).
Usage: python verify.py --pool F1 F2 ... --target T1 T2 ... --out-dir OUT --tag TAG [--inputs NPZ]"""
import argparse, hashlib, json, os, re
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def strokes(a):
    a = np.asarray(a, np.float64); e = np.flatnonzero(a[:, 3] < 0); a = a[: e[0] + 1] if e.size else a
    cut = np.flatnonzero(a[:, 2] < 0) + 1
    return [(s[:, :2] + 1.0) * 32.0 for s in np.split(a, cut) if len(s)]
def pu(A, B, tol=3.0):
    if not A or not B: return 0.0
    a0 = np.array([s[0] for s in A]); a1 = np.array([s[-1] for s in A]); b0 = np.array([s[0] for s in B]); b1 = np.array([s[-1] for s in B])
    d = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=2)
    Co = np.maximum(d(a0, b0), d(a1, b1)); C = np.minimum(Co, np.maximum(d(a0, b1), d(a1, b0)))
    f = lambda M: 2.0 * int((M[linear_sum_assignment(M)] <= tol).sum()) / (len(A) + len(B))
    return f(C) + f(Co)   # unoriented end-point agreement (pen-up v2 analogue) + oriented agreement (direction)
def inkF(S, img):
    if not S: return 0.0
    ink = img < 128; dil = ndi.binary_dilation(ink, iterations=1); P = np.vstack(S)
    c = np.clip(np.round(P[:, 0]).astype(int), 0, 63); r = np.clip(np.round(P[:, 1]).astype(int), 0, 63)
    prec = dil[r, c].mean(); M = np.zeros_like(ink); M[r, c] = True; M = ndi.binary_dilation(M, iterations=1)
    rec = (M & ink).sum() / max(ink.sum(), 1); return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
def select(cands, img):
    S = [strokes(a) for a in cands]; n = len(S)
    u = [np.mean([pu(S[i], S[j]) for j in range(n) if j != i]) if n > 1 else 0.0 for i in range(n)]
    f = [inkF(s, img) for s in S]
    return max(range(n), key=lambda i: (round(u[i], 9), round(f[i], 9), -i)), u, f
def dir_vote(cands, i, tol=3.0):
    """reverse each stroke of the selected candidate i when a strict majority of the other candidates' end-point-matched strokes
    (unoriented cost <= tol, Hungarian) run the opposite way (reversed cost < oriented cost). Rows reversed bit-exact; pen flags reset."""
    a = np.asarray(cands[i]); e = np.flatnonzero(a[:, 3] < 0); end = int(e[0]) + 1 if e.size else len(a)
    b = a[:end]; cut = np.flatnonzero(b[:, 2] < 0) + 1; R = [s for s in np.split(b, cut) if len(s)]
    S = [(s[:, :2].astype(np.float64) + 1.0) * 32.0 for s in R]
    if not S: return a.copy(), 0
    votes = np.zeros((len(S), 2))
    for j, c in enumerate(cands):
        if j == i: continue
        T = strokes(c)
        if not T: continue
        a0 = np.array([s[0] for s in S]); a1 = np.array([s[-1] for s in S]); b0 = np.array([s[0] for s in T]); b1 = np.array([s[-1] for s in T])
        d = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=2)
        Co = np.maximum(d(a0, b0), d(a1, b1)); Cr = np.maximum(d(a0, b1), d(a1, b0)); C = np.minimum(Co, Cr)
        for p, q in zip(*linear_sum_assignment(C)):
            if C[p, q] <= tol and np.linalg.norm(S[p][-1] - S[p][0]) >= 2.0: votes[p, int(Cr[p, q] < Co[p, q])] += 1
    flip = votes[:, 1] > votes[:, 0]
    if not flip.any(): return a.copy(), 0
    rows = []
    for s, f in zip(R, flip):
        s = s[::-1].copy() if f else s.copy(); s[:, 2] = 1.0; s[-1, 2] = -1.0; rows.append(s)
    Q = np.vstack(rows); Q[:, 3] = 1.0
    if e.size: Q[-1, 3] = a[end - 1, 3]
    out = a.copy(); out[:end] = Q.astype(a.dtype); return out, int(flip.sum())
def ds_feat(s, P):   # identical to dirscorer_train.feat
    x0, y0 = P.min(0); w = max(P[:, 0].max() - x0, P[:, 1].max() - y0, 1e-6)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))]
    t = np.linspace(0, d[-1], 16); q = np.c_[np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])]
    return np.r_[((q - [x0, y0]) / w).ravel(), d[-1] / w, float(d[-1] < 0.25 * (P[:, 1].max() - P[:, 1].min()))]
def ds_prob(M, x):
    h = np.maximum(x @ M["W0"] + M["b0"], 0); z = (h @ M["W1"] + M["b1"]).ravel(); return 1 / (1 + np.exp(-z))
def dir_learned(a, M, thr):
    """reverse a stroke when the p01-trained direction scorer prefers the reversed orientation by more than thr:
    p(reversed) - p(as is) > thr (strokes with endpoint distance >= 2 px). Rows reversed bit-exact; pen flags reset."""
    a = np.asarray(a); e = np.flatnonzero(a[:, 3] < 0); end = int(e[0]) + 1 if e.size else len(a)
    b = a[:end]; cut = np.flatnonzero(b[:, 2] < 0) + 1; R = [s for s in np.split(b, cut) if len(s)]
    S = [(s[:, :2].astype(np.float64) + 1.0) * 32.0 for s in R]
    if not S: return a.copy(), 0
    P = np.vstack(S); flip = []
    for s in S:
        if len(s) < 2 or np.linalg.norm(s[-1] - s[0]) < 2.0: flip.append(False); continue
        p = ds_prob(M, np.stack([ds_feat(s, P), ds_feat(s[::-1], P)])); flip.append(bool(p[1] - p[0] > thr))
    if not any(flip): return a.copy(), 0
    rows = []
    for s, f in zip(R, flip):
        s = s[::-1].copy() if f else s.copy(); s[:, 2] = 1.0; s[-1, 2] = -1.0; rows.append(s)
    Q = np.vstack(rows); Q[:, 3] = 1.0
    if e.size: Q[-1, 3] = a[end - 1, 3]
    out = a.copy(); out[:end] = Q.astype(a.dtype); return out, int(sum(flip))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", nargs="+", required=True); ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--dir-vote", action="store_true")
    ap.add_argument("--dir-model", default=""); ap.add_argument("--dir-thr", type=float, default=0.95)
    ap.add_argument("--inputs", default="$U/exp/ORDER/lanes/Repair/val1000_inputs_only.npz"); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    Z = [dict(np.load(f, allow_pickle=False)) for f in a.pool]; I = np.load(a.inputs, allow_pickle=False)
    for z in Z: assert (z["unit_ids"] == I["unit_ids"]).all(), "unit order"
    sel = np.array([select([z["preds"][k] for z in Z], I["cond_rendered"][k])[0] for k in range(len(I["unit_ids"]))])
    W = np.stack([z["preds"] for z in Z])[sel, np.arange(len(sel))].copy(); nflip = 0
    if a.dir_vote:
        for k in range(len(sel)):
            W[k], nf = dir_vote([z["preds"][k] for z in Z], int(sel[k])); nflip += nf
        print("dir-vote strokes reversed", nflip, flush=True)
    nlearn = 0
    if a.dir_model:
        DM = dict(np.load(a.dir_model))
        for k in range(len(sel)):
            W[k], nf = dir_learned(W[k], DM, a.dir_thr); nlearn += nf
        print("dir-learned strokes reversed", nlearn, flush=True)
    for f in a.target:
        m = re.match(r"^(val1000_ouhdl__.+)__seed(\d+)_preds\.npz$", os.path.basename(f)); assert m, f
        side = json.load(open(f[:-4] + ".json")); hs = sha(f); assert side.get("preds_sha256") in (None, hs), ("sha", f)
        z = dict(np.load(f, allow_pickle=False)); assert (z["unit_ids"] == I["unit_ids"]).all(); z["preds"] = W.astype(z["preds"].dtype)
        out = os.path.join(a.out_dir, f"{m.group(1)}_{a.tag}__seed{m.group(2)}_preds.npz"); np.savez(out, **z)
        side = dict(side); side["preds_sha256"] = sha(out)
        side["repair"] = dict(mode="verify_mbr_pu" + ("+dir_vote" if a.dir_vote else "") + ("+dir_learned" if a.dir_model else ""), n_strokes_reversed=int(nflip), n_strokes_reversed_learned=int(nlearn),
                              dir_model=(dict(path=a.dir_model, sha256=sha(a.dir_model), thr=a.dir_thr) if a.dir_model else None), code_sha256=sha(__file__), source=f, source_sha256=hs, pool=[dict(path=p, sha256=sha(p)) for p in a.pool],
                              selected_pool_index_counts=np.bincount(sel, minlength=len(a.pool)).tolist())
        json.dump(side, open(out[:-4] + ".json", "w"), indent=1); print(os.path.basename(out), "sel counts", np.bincount(sel, minlength=len(a.pool)).tolist(), flush=True)
if __name__ == "__main__": main()
