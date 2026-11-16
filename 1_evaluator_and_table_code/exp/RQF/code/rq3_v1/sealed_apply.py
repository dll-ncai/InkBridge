"""ORDER Synthesis, amendment A1: apply the Repair-lane k3 verifier (verify.py select(), imported unchanged, no dir-vote / dir-model)
and the random-of-3 control to the 3 sampling seeds of one COMP training seed. GT-free: reads only unit_ids + cond_rendered from --inputs.
Arm  : sel[k] = verify.select([preds_s0[k], preds_s1[k], preds_s2[k]], cond_rendered[k])[0]   (identical to verify.main)
Ctrl : rnd[k] = np.random.default_rng([20261104, SET_INDEX, TRAIN_INDEX]).integers(0, 3, size=n_units)[k]
Each of the 3 pool files is written as <stem>_<tag>__seed<j>_preds.npz with 'preds' replaced by the chosen candidate (bit-exact rows),
sidecar json copied with preds_sha256 updated and a 'repair' record (as verify.main). Usage:
python sealed_apply.py --verify verify.py --verify-sha SHA --pool F0 F1 F2 --inputs NPZ --set-index S --train-index T --out-dir OUT"""
import argparse, hashlib, importlib.util, json, os, re
import numpy as np
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", required=True); ap.add_argument("--verify-sha", required=True)
    ap.add_argument("--pool", nargs=3, required=True); ap.add_argument("--inputs", required=True); ap.add_argument("--cond-key", default="cond_rendered")
    ap.add_argument("--set-index", type=int, required=True); ap.add_argument("--train-index", type=int, required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--rng", type=int, default=20261104); a = ap.parse_args()
    assert sha(a.verify) == a.verify_sha, "verify.py sha mismatch"
    sp = importlib.util.spec_from_file_location("verify", a.verify); V = importlib.util.module_from_spec(sp); sp.loader.exec_module(V)
    os.makedirs(a.out_dir, exist_ok=True)
    Z = [dict(np.load(f, allow_pickle=False)) for f in a.pool]
    for j, f in enumerate(a.pool): assert re.search(rf"__seed{j}_preds\.npz$", f), f
    for z in Z: assert (z["unit_ids"] == Z[0]["unit_ids"]).all(), "unit order"
    I = np.load(a.inputs, allow_pickle=False); iu = I["unit_ids"]; pos = {u: i for i, u in enumerate(iu)}
    rows = np.array([pos[u] for u in Z[0]["unit_ids"]]); cond = I[a.cond_key][rows]
    n = len(rows)
    sel = np.array([V.select([z["preds"][k] for z in Z], cond[k])[0] for k in range(n)])
    rnd = np.random.default_rng([a.rng, a.set_index, a.train_index]).integers(0, 3, size=n)
    P = np.stack([z["preds"] for z in Z]); out_rec = {}
    for tag, idx, mode in (("rep3", sel, "verify_mbr_pu (k3 verify_seeds)"), ("rnd3", rnd, f"random_of_3 rng=[{a.rng},{a.set_index},{a.train_index}]")):
        W = P[idx, np.arange(n)].copy()
        for j, f in enumerate(a.pool):
            stem = os.path.basename(f)[: -len(f"__seed{j}_preds.npz")]
            side = json.load(open(f[:-4] + ".json")); hs = sha(f); assert side.get("preds_sha256") in (None, hs), ("sha", f)
            z = dict(Z[j]); z["preds"] = W.astype(z["preds"].dtype)
            o = os.path.join(a.out_dir, f"{stem}_{tag}__seed{j}_preds.npz"); np.savez(o, **z)
            side = dict(side); side["preds_sha256"] = sha(o)
            side["repair"] = dict(mode=mode, code_sha256=sha(__file__), verify_sha256=sha(a.verify), source=f, source_sha256=hs,
                                  pool=[dict(path=p, sha256=sha(p)) for p in a.pool], inputs=a.inputs, inputs_sha256=sha(a.inputs),
                                  selected_pool_index_counts=np.bincount(idx, minlength=3).tolist())
            json.dump(side, open(o[:-4] + ".json", "w"), indent=1)
        out_rec[tag] = np.bincount(idx, minlength=3).tolist()
    stem0 = os.path.basename(a.pool[0])[: -len("__seed0_preds.npz")]
    np.savez(os.path.join(a.out_dir, f"{stem0}__selection.npz"), unit_ids=Z[0]["unit_ids"], sel=sel, rnd=rnd)
    print(stem0, "counts", json.dumps(out_rec), "agree(sel==rnd)", float((sel == rnd).mean()), flush=True)
if __name__ == "__main__": main()
