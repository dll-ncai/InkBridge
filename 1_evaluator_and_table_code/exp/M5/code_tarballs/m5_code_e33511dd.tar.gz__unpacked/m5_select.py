"""M5 checkpoint selection (E0 rule): lowest mean native LDTW over the arm's val units, ties -> higher AIoU.
Rows files are named  val__<valset>__ms<m>_rows.csv  (e0_score.py e0 output).  The mean is over all val units of the
arm pooled (equal units per script, so it equals the mean of the per-script means).

python m5_select.py --eval-dir EV --valsets casia_zh,casia_en,hpl_tamil,iam --criterion-note "..." --out EV/selection.json
prints the selected milestone (integer) as the last stdout line.
"""
import argparse
import glob
import json
import os
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True); ap.add_argument("--valsets", required=True)
    ap.add_argument("--criterion-note", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vs = a.valsets.split(",")
    recs = []
    for f in glob.glob(os.path.join(a.eval_dir, "val__*__ms*_rows.csv")):
        m = re.match(r"val__(.+)__ms(\d+)_rows\.csv$", os.path.basename(f))
        if not m or m.group(1) not in vs:
            continue
        d = pd.read_csv(f); d = d[d.protocol == "native"]
        recs.append(d.assign(valset=m.group(1), milestone=int(m.group(2))))
    df = pd.concat(recs)
    have = df.groupby("milestone").valset.nunique()
    complete = sorted(have[have == len(vs)].index)
    df = df[df.milestone.isin(complete)]
    per = df.groupby(["milestone", "valset"]).agg(ldtw=("ldtw", "mean"), aiou=("aiou", "mean"), n=("unit_id", "size"),
                                                 direction_acc=("direction_acc", "mean"), kendall_tau=("kendall_tau", "mean")).reset_index()
    pooled = df.groupby("milestone").agg(ldtw=("ldtw", "mean"), aiou=("aiou", "mean"), dtw=("dtw", "mean"),
                                         direction_acc=("direction_acc", "mean"), kendall_tau=("kendall_tau", "mean"),
                                         n_units=("unit_id", "size")).reset_index()
    pooled = pooled.sort_values(["ldtw", "aiou"], ascending=[True, False])
    best = int(pooled.milestone.iloc[0])
    json.dump(dict(criterion="lowest mean native LDTW over the pooled val units, ties -> higher AIoU; DDIM-50 eta 0 seed 0",
                   note=a.criterion_note, valsets=vs, milestones_evaluated=complete, best_milestone=best,
                   pooled=pooled.round(5).to_dict("records"), per_valset=per.round(5).to_dict("records")),
              open(a.out, "w"), indent=1)
    per.to_csv(a.out.replace(".json", "_per_valset.csv"), index=False)
    print(pooled.round(4).to_string(index=False))
    print(best)


if __name__ == "__main__":
    main()
