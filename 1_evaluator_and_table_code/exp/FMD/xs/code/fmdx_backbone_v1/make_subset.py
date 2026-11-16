"""REPA shared subset S (prereg A5.2, capacity note section 4): rows of exp/FMD/xs/data/luo_repa_order_v1.csv with
rank_in_script < N/4 -> positions (rows of m5_luo_train.npz, sorted) .npy + meta (csv sha256, N, per-script counts, fraction of
the 79,200 Trainer1D training units in S)."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from fmd_harness import common as C
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_data as BD

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--npz", required=True)
ap.add_argument("--n", type=int, default=40000); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
cols = list(rows[0].keys())
uid_col = "unit_id" if "unit_id" in cols else [c for c in cols if "unit" in c][0]
sel = [r[uid_col] for r in rows if int(r["rank_in_script"]) < a.n // 4]
z = np.load(a.npz, allow_pickle=False); u = z["unit_ids"].astype(str); sc = z["scripts"].astype(str)
pos = {x: i for i, x in enumerate(u)}
P = np.array(sorted(pos[x] for x in sel), np.int64)
tr, va = BD.trainer1d_split(len(u))
in_tr = int(np.isin(P, tr).sum())
np.save(a.out, P)
res = dict(n=int(len(P)), per_script={s: int((sc[P] == s).sum()) for s in np.unique(sc)}, csv_columns=cols,
           in_trainer1d_train=in_tr, in_heldout=int(len(P) - in_tr), frac_of_train_units_in_S=round(in_tr / len(tr), 5),
           unit_ids_sha256=C.sha256_bytes("\n".join(u[P]).encode()), positions_sha256=C.sha256_array(P))
C.write_meta(a.out, kind="fmdx_repa_subset", inputs={"order_csv": a.csv, "order_csv_meta": a.csv + ".meta.json", "luo_npz": a.npz},
             params=dict(rule="capacity note section 4 / prereg A5.2: rank_in_script < N/4", N=a.n),
             extra=dict(result=res, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
print(json.dumps(res))
