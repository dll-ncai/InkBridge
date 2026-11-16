"""Milestone selection, same rule as E0 (jobs/e0_eval_final.sh): lowest mean native LDTW on the 1,000 random val units
(subset seed 0), ties -> higher mean native AIoU.  Means over units with a defined value (pandas skipna).
python pennet_select.py --rows EV/val1000_ms*_rows.csv --out EV/selection.json   (prints the selected milestone index)
"""
import argparse
import json
import os
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
recs = []
for f in a.rows:
    k = int(re.search(r"_ms(\d+)_rows\.csv$", os.path.basename(f)).group(1))
    d = pd.read_csv(f)
    d = d[d.protocol == "native"]
    recs.append(dict(milestone=k, step=None, n_units=int(d.unit_id.nunique()), n_errors=int((d["error"].fillna("") != "").sum()),
                     ldtw=float(d.ldtw.astype(float).mean()), aiou=float(d.aiou.astype(float).mean()),
                     dtw=float(d.dtw.astype(float).mean()), strict_order_acc=float(d.strict_order_acc.astype(float).mean()),
                     direction_acc=float(d.direction_acc.astype(float).mean()), n_empty_pred=int(d.empty_pred.fillna(0).sum()),
                     rows_file=os.path.basename(f)))
t = pd.DataFrame(recs).sort_values(["ldtw", "aiou"], ascending=[True, False]).reset_index(drop=True)
best = int(t.milestone.iloc[0])
json.dump(dict(criterion="lowest mean native LDTW on 1000 random val units (subset seed 0, rendered input), ties -> higher "
                         "AIoU; greedy decoding (deterministic); same rule as E0 selection.json",
               best_milestone=best, table=t.round(6).to_dict("records")), open(a.out, "w"), indent=1)
print(best)
