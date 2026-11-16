#!/usr/bin/env python3
"""FMDX arm table (prereg A4.2/A4.3) -> exp/FMD/xs/admin/arms.tsv + arm_queue.csv (one row per arm, state QUEUED / BLOCKED).

Arms per regime (WARM 40,000 steps / SCRATCH 100,000 steps, save every 10,000, seed 20261016, 512 sequences per step, A6):
  NOFM_<R>      student encoder, no FM loss (control)
  PEINIT_<R>    pe-spatial-s16-512 tower init (control)
  <fm>_<R>      REPA to the A3.3 gate layer of every roster FM with a token map (fm_roster_A3.json, arm 'repa')
  INKSIGHT_<R>  InkSight pseudo-label arm (pooled pseudo units)
init_rank (A4.3 b): NOFM, dinov3-vitl16, cleandift-sd21 (sd2.1-dift if cleandift-sd21 has not passed its staging load test),
PEINIT, then the capacity-note FM order (or roster order). The controller re-sorts FM arms by verdict priority_score.
BLOCKED rows: gated FMs pending licence acceptance (no weights), FMs not staged. Rows are never deleted; re-running adds rows
for new FMs (A3.x) and updates only the target paths of QUEUED rows whose extractor became known.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

COLS = ["arm_id", "regime", "encoder", "fm", "steps", "save_every", "repa_targets", "repa_unit_ids", "repa_lambda", "pseudo_npz",
        "pseudo_render", "pseudo_frac", "pe_weights", "init_rank", "hold_until_pass", "lanes", "needs", "status", "note"]
REG = {"warm": ("W", 40000, 10000), "scratch": ("S", 100000, 10000)}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xs", default="$U/exp/FMD/xs")
    ap.add_argument("--roster", required=True)
    ap.add_argument("--subset-name", required=True, help="capacity-note shared subset S name (targets file suffix)")
    ap.add_argument("--order", default="", help="comma list of fm ids after the A4.3(b) head (capacity note); others follow in roster order")
    ap.add_argument("--slot3", default="cleandift-sd21", help="cleandift-sd21 or sd2.1-dift (A4.3 b)")
    ap.add_argument("--pe-weights", required=True)
    ap.add_argument("--staged", default="", help="comma list of staged fm ids (others -> BLOCKED not staged)")
    a = ap.parse_args()
    xs = Path(a.xs); adm = xs / "admin"; adm.mkdir(parents=True, exist_ok=True)
    roster = json.loads(Path(a.roster).read_text())["fms"]
    staged = set(x for x in a.staged.split(",") if x)
    head = ["dinov3-vitl16", a.slot3]
    rest = [f for f in a.order.split(",") if f and f not in head] + [f for f in roster if f not in head and f not in a.order.split(",")]
    fm_order = head + rest
    rank = {"NOFM": 1, "dinov3-vitl16": 2, a.slot3: 3, "PEINIT": 4}
    k = 5
    for f in fm_order:
        if f not in rank:
            rank[f] = k; k += 1
    old = {}
    tsv = adm / "arms.tsv"
    if tsv.exists():
        rows = list(csv.DictReader(open(tsv), delimiter="\t"))
        old = {r["arm_id"]: r for r in rows}
    out = []
    for rg, (tag, steps, save) in REG.items():
        def row(arm, enc, fm, **kw):
            r = {c: "-" for c in COLS}
            r.update(arm_id=f"{arm}_{tag}", regime=rg, encoder=enc, fm=fm, steps=str(steps), save_every=str(save), repa_lambda="0.5",
                     pseudo_frac="0", init_rank=str(rank.get(arm if fm == "-" else fm, 999)), hold_until_pass="0", lanes="clusterA,clusterB,vps",
                     status="QUEUED", note="")
            r.update({k2: str(v) for k2, v in kw.items()})
            return r
        out.append(row("NOFM", "vits14", "-", note="control: student encoder, no FM loss"))
        out.append(row("PEINIT", "pe_spatial_s16", "-", pe_weights=a.pe_weights, note="control: pe-spatial-s16-512 init (FlexiViT 16->14)"))
        for f in fm_order:
            info = roster.get(f)
            if info is None:
                continue
            arm = f if info["arm"] == "repa" else "INKSIGHT"
            lanes = info["licence_lanes"]
            if info["arm"] == "pseudo":
                pd = xs / "data/pseudo"
                r = row("INKSIGHT", "vits14", f, pseudo_npz=pd / "pseudo_train.npz", pseudo_render=pd / "pseudo_train_render224.npy",
                        needs=pd / "pseudo_train.npz.meta.json", lanes=lanes, note="InkSight pseudo-label arm (pooled)")
                r["init_rank"] = str(rank.get(f, 999))
            else:
                t = xs / "targets/repa" / f / f"repa_{f}_gate_G16__{a.subset_name}.npy"
                r = row(f, "vits14", f, repa_targets=t, repa_unit_ids=str(t)[:-4] + "__unit_ids.npy",
                        needs=str(t)[:-4] + ".DONE.json", lanes=lanes, note=f"REPA {info['family']} (A3.3 gate layer)")
            if info.get("blocked_reason"):
                r["status"] = "BLOCKED"; r["note"] += "; " + info["blocked_reason"]
            elif info.get("gated_blocked_pending_licence"):
                r["status"] = "BLOCKED"; r["note"] += "; gated, licence acceptance pending (lead 3 Nov) - no weights"
            elif staged and f not in staged:
                r["status"] = "BLOCKED"; r["note"] += "; not staged"
            o = old.get(r["arm_id"])
            if o is not None and o["status"] not in ("QUEUED", "BLOCKED"):
                r["status"] = o["status"]
            out.append(r)
    tmp = tsv.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(out)
    tmp.replace(tsv)
    q = adm / "arm_queue.csv"
    have = {}
    if q.exists():
        have = {r["arm_id"]: r for r in csv.DictReader(open(q))}
    qcols = ["arm_id", "regime", "fm", "state", "clusterA_job", "clusterB_job", "vps_cmd", "updated_utc", "note"]
    qrows = []
    for r in out:
        h = have.get(r["arm_id"])
        qrows.append(h if h else dict(arm_id=r["arm_id"], regime=r["regime"], fm=r["fm"], state=r["status"], clusterA_job="", clusterB_job="", vps_cmd="",
                                      updated_utc=utc(), note=r["note"]))
    with open(q, "w", newline="") as fh:
        w = csv.DictWriter(fh, qcols); w.writeheader(); w.writerows(qrows)
    with open(adm / "arm_log.md", "a") as fh:
        fh.write(f"- {utc()} make_arms.py: arms.tsv rows {len(out)} (QUEUED {sum(r['status']=='QUEUED' for r in out)}, BLOCKED "
                 f"{sum(r['status']=='BLOCKED' for r in out)}), sha256 {sha(tsv)}; subset {a.subset_name}; slot3 {a.slot3}\n")
    print(len(out), sha(tsv))


if __name__ == "__main__":
    main()
