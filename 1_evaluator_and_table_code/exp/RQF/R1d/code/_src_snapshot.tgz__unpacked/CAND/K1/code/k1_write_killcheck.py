"""Write the K1 section of KILLCHECK.md and killcheck_verdicts.csv from the k1_summary_cand.py outputs (numbers only)."""
import argparse, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--summ", required=True); ap.add_argument("--out", required=True); ap.add_argument("--prov", required=True); ap.add_argument("--k2-note", default="K2 status not read")
a = ap.parse_args()
PF = pd.read_csv(os.path.join(a.summ, "k1_per_frame.csv")); PD = pd.read_csv(os.path.join(a.summ, "k1_paired_vs_identity.csv"))
SC = pd.read_csv(os.path.join(a.summ, "k1_strokeclass.csv")); SP = pd.read_csv(os.path.join(a.summ, "k1_strokeclass_paired.csv"))
SH = pd.read_csv(os.path.join(a.summ, "k1_strokeclass_shares.csv")); V = json.load(open(os.path.join(a.summ, "k1_verdict.json")))
P = json.load(open(a.prov))
f4 = lambda x: "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
ci = lambda m, l, h: f"{f4(m)} [{f4(l)}, {f4(h)}]"
L = []
L += ["# KILLCHECK.md — exp/CAND kill checks", "", "## K1 — canonical-frame zero-shot check (LUO ms6; identity / mirror_x / vt)", "",
      "Facts only. Exploratory round (prereg_candidates_2026-10-28.md sec. 7). Numbers from uhwr_eval 0.3.1, native protocol.", ""]
L += ["### Rule as written", "",
      "- Prereg sec. 5, K1 (verbatim): kill if \"best frame does not reach direction ≥ 0.70 or τ ≥ 0.65 (identity: 0.547 / 0.480), or AIoU drops > 0.02 below 0.7117\". Also report mirror-x vs VT on loop-dominated strokes.",
      "- Prereg sec. 5 frame selection rule: the frame with the higher Urdu test Kendall tau among those passing the K1 thresholds; ties (|dtau| < 0.01) broken by direction accuracy, then AIoU.",
      "- Track brief: direction ≥ 0.70, tau ≥ 0.65 with CIs excluding the identity values 0.5468 / 0.4799; AIoU not more than 0.02 below 0.7117 (i.e. ≥ 0.6917).",
      "- Set: ouhdl_test (4,406 units, seeds 0-2, per-unit seed mean). CI: writer-cluster percentile bootstrap, B = 2000, numpy default_rng(20261028). 'CI excludes identity value' = CI lower bound > 0.5468 (direction) / > 0.4799 (tau). AIoU check on the point estimate.",
      "- The sentence 'does not reach direction ≥ 0.70 or τ ≥ 0.65' can be parsed two ways. Both are evaluated below and neither is chosen here: **AND** = direction, tau and AIoU conditions must all hold (the reading frame_k1_summary.py in frame_code implements); **OR** = at least one of direction / tau holds (each with its CI condition), and the AIoU condition holds.", ""]
# table 1
T = PF[(PF.set == "test_ouhdl") & (PF.cluster == "writer_id")]
def g(df, fr, met, col="mean"):
    x = df[(df.frame == fr) & (df.metric == met)]
    return float(x[col].iloc[0]) if len(x) else np.nan
mets = ["aiou", "kendall_tau", "direction_acc", "tau_cp", "strict_order_acc", "ldtw", "loop_orient_acc", "penup_f1_v2"]
L += ["### Table K1-1. ouhdl_test (4,406 units) per frame — mean [95 % writer-cluster CI]", "",
      "| frame | AIoU | Kendall tau | direction | tau_cp | strict order | LDTW | loop orient | pen-up F1 v2 | n units | n writers | seeds |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for fr in ["identity", "mirror_x", "vt"]:
    if not len(T[T.frame == fr]): continue
    row = [ci(g(T, fr, m), g(T, fr, m, "ci_lo"), g(T, fr, m, "ci_hi")) for m in mets]
    n = T[(T.frame == fr) & (T.metric == "aiou")].iloc[0]
    L.append(f"| {fr} | " + " | ".join(row) + f" | {int(n.n_units)} | {int(n.n_clusters)} | {int(n.n_seeds)} |")
L += ["", f"Pack reference for LUO ms6 zero-shot ouhdl_test (prereg sec. 2): AIoU 0.7117, tau 0.4799, direction 0.5468. Identity frame in this run: AIoU {f4(V['identity_this_run'].get('aiou'))}, tau {f4(V['identity_this_run'].get('kendall_tau'))}, direction {f4(V['identity_this_run'].get('direction_acc'))}.", ""]
nd = T[(T.metric == "kendall_tau")][["frame", "n_defined"]].set_index("frame").n_defined.to_dict()
L += [f"Units with defined tau (≥ 2 matched strokes, seed mean): " + ", ".join(f"{k} {int(v)}" for k, v in nd.items()) + ".", ""]
# table 2 paired
Q = PD[(PD.set == "test_ouhdl") & (PD.cluster == "writer_id")]
L += ["### Table K1-2. ouhdl_test paired per-unit deltas vs identity (frame − identity), seed-averaged, 95 % writer-cluster CI", "",
      "| frame | metric | delta [CI] | CI excludes 0 | n pairs | frac + / − |", "|---|---|---|---|---|---|"]
for fr in ["mirror_x", "vt"]:
    for m in mets:
        x = Q[(Q.frame == fr) & (Q.metric == m)]
        if not len(x): continue
        x = x.iloc[0]; L.append(f"| {fr} | {m} | {ci(x.delta, x.ci_lo, x.ci_hi)} | {bool(x.ci_excludes_0)} | {int(x.n_pairs)} | {x.frac_pos:.3f} / {x.frac_neg:.3f} |")
L.append("")
# table 3 checks
rows_csv = []
L += ["### Table K1-3. K1 conditions on ouhdl_test", "",
      "| frame | direction ≥ 0.70 | direction CI lo > 0.5468 | tau ≥ 0.65 | tau CI lo > 0.4799 | AIoU ≥ 0.6917 | pass (AND) | pass (OR) |", "|---|---|---|---|---|---|---|---|"]
for fr, c in V["per_frame"].items():
    L.append(f"| {fr} | {c['direction_ge_070']} ({f4(c['direction_mean'])}) | {c['direction_ci_excludes_ref']} ({f4(c['direction_ci'][0])}) | {c['tau_ge_065']} ({f4(c['tau_mean'])}) | {c['tau_ci_excludes_ref']} ({f4(c['tau_ci'][0])}) | {c['aiou_within_002']} ({f4(c['aiou_mean'])}) | {c['pass_reading_AND']} | {c['pass_reading_OR']} |")
    for cid, met, val, lo, hi, thr, cmp_, res in [
        ("K1.direction_ge_070", "direction_acc", c["direction_mean"], c["direction_ci"][0], c["direction_ci"][1], 0.70, "mean >= 0.70", c["direction_ge_070"]),
        ("K1.direction_ci_excludes_identity", "direction_acc", c["direction_mean"], c["direction_ci"][0], c["direction_ci"][1], 0.5468, "ci_lo > 0.5468", c["direction_ci_excludes_ref"]),
        ("K1.tau_ge_065", "kendall_tau", c["tau_mean"], c["tau_ci"][0], c["tau_ci"][1], 0.65, "mean >= 0.65", c["tau_ge_065"]),
        ("K1.tau_ci_excludes_identity", "kendall_tau", c["tau_mean"], c["tau_ci"][0], c["tau_ci"][1], 0.4799, "ci_lo > 0.4799", c["tau_ci_excludes_ref"]),
        ("K1.aiou_within_0.02", "aiou", c["aiou_mean"], c["aiou_ci"][0], c["aiou_ci"][1], 0.6917, "mean >= 0.7117 - 0.02", c["aiou_within_002"]),
        ("K1.pass_reading_AND", "all", np.nan, np.nan, np.nan, np.nan, "direction AND tau AND aiou conditions", c["pass_reading_AND"]),
        ("K1.pass_reading_OR", "all", np.nan, np.nan, np.nan, np.nan, "(direction OR tau conditions) AND aiou condition", c["pass_reading_OR"])]:
        rows_csv.append(dict(check=cid, kill_check="K1", set="test_ouhdl", frame=fr, metric=met, value=val, ci_lo=lo, ci_hi=hi, threshold=thr, rule=cmp_, outcome="PASS" if res else "FAIL"))
wa, wo = V["winner_reading_AND"], V["winner_reading_OR"]
L += ["", f"Winning frame (selection rule, sec. 5): reading AND → **{wa or 'none'}**; reading OR → **{wo or 'none'}**.", ""]
rows_csv.append(dict(check="K1.winner_reading_AND", kill_check="K1", set="test_ouhdl", frame=wa or "none", metric="kendall_tau", rule="highest test tau among passing frames; |dtau|<0.01 -> direction, AIoU", outcome=wa or "none"))
rows_csv.append(dict(check="K1.winner_reading_OR", kill_check="K1", set="test_ouhdl", frame=wo or "none", metric="kendall_tau", rule="highest test tau among passing frames; |dtau|<0.01 -> direction, AIoU", outcome=wo or "none"))
# stroke class
L += ["### Table K1-4. ouhdl_test direction accuracy by GT stroke class (stroke-level pooled over seeds; 95 % writer-cluster CI)", "",
      "Class rule (candidates document sec. 2 / PROVENANCE_stroke_sense_canonical_frames.json): per GT stroke with ≥ 3 distinct points, dominant of |net dx/L|, |net dy/L|, |total signed turning/2π|. Re-implemented in k1_strokeclass.py on the 64×64 evaluation-canvas GT ink. Per-unit sums reproduce uhwr_eval direction_acc / loop_orient_acc (check below).", ""]
sh = SH.set_index("set")
for s in sh.index:
    x = sh.loc[s]; L.append(f"- {s}: {int(x.n_gt_strokes)} GT strokes, {int(x.n_ge3pts)} with ≥ 3 distinct points; shares horizontal {x.share_ge3_horizontal:.3f} / vertical {x.share_ge3_vertical:.3f} / chirality {x.share_ge3_chirality:.3f} (reference: stroke-sense analysis of the raw Urdu test ink, 0.222 / 0.166 / 0.612).")
L += ["", "| frame | class | direction acc [CI] | dir-eligible strokes (3 seeds) | loop-orient acc [CI] | loop-eligible strokes (3 seeds) |", "|---|---|---|---|---|---|"]
S_ = SC[SC.set == "test_ouhdl"]
for fr in ["identity", "mirror_x", "vt"]:
    for cl in ["horizontal", "vertical", "chirality", "lt3pts", "all"]:
        x = S_[(S_.frame == fr) & (S_.cls == cl)]
        if not len(x): continue
        x = x.iloc[0]
        L.append(f"| {fr} | {cl} | {ci(x.direction_acc, x.direction_lo, x.direction_hi)} | {int(x.direction_n_eligible_all_seeds)} | {ci(x.loop_orient_acc, x.loop_orient_lo, x.loop_orient_hi)} | {int(x.loop_orient_n_eligible_all_seeds)} |")
L += ["", "Paired class-level differences (same writer resamples for both frames):", "", "| contrast | class | metric | delta [CI] | CI excludes 0 |", "|---|---|---|---|---|"]
for (fr, ref) in [("mirror_x", "identity"), ("vt", "identity"), ("mirror_x", "vt")]:
    for cl in ["horizontal", "vertical", "chirality", "all"]:
        for m in ["direction_acc", "loop_orient_acc"]:
            x = SP[(SP.set == "test_ouhdl") & (SP.frame == fr) & (SP.comparator == ref) & (SP.cls == cl) & (SP.metric == m)]
            if not len(x): continue
            x = x.iloc[0]; L.append(f"| {fr} − {ref} | {cl} | {m} | {ci(x.delta, x.ci_lo, x.ci_hi)} | {bool(x.ci_excludes_0)} |")
            rows_csv.append(dict(check=f"K1.report.strokeclass.{fr}_minus_{ref}.{cl}.{m}", kill_check="K1 (reported, not gated)", set="test_ouhdl", frame=fr, metric=m, value=x.delta, ci_lo=x.ci_lo, ci_hi=x.ci_hi, rule="paired vs " + ref, outcome="reported"))
cons = V["stroke_consistency_vs_rows031"]
L += ["", f"Consistency check vs rows031: {cons['n_units_joined']} (file, unit) pairs; n_dir_eligible mismatches {cons['n_dir_eligible_mismatch']}; n_loop_eligible mismatches {cons['n_loop_eligible_mismatch']}; max |Δ direction_acc| {cons['max_abs_diff_direction_acc']}; max |Δ loop_orient_acc| {cons['max_abs_diff_loop_orient_acc']}.", ""]
# val250 and IAM
for s, title in [("val250_ouhdl", "Table K1-5. val250_ouhdl (first 250 units of e0_val, stored order; reported, not gated)"), ("val250_iam", "Table K1-6. IAM val250 (m5_val_iam, symmetric LTR control; reported, not gated)")]:
    for clname in ["writer_id", "sample_id"]:
        X = PF[(PF.set == s) & (PF.cluster == clname)]
        if not len(X): continue
        n = X[X.metric == "aiou"].iloc[0]
        L += [f"### {title} — {clname.replace('_id','')}-cluster CI ({int(n.n_clusters)} clusters)", "", "| frame | AIoU | Kendall tau | direction | tau_cp | strict order | loop orient |", "|---|---|---|---|---|---|---|"]
        for fr in ["identity", "mirror_x", "vt"]:
            if not len(X[X.frame == fr]): continue
            L.append(f"| {fr} | " + " | ".join(ci(g(X, fr, m), g(X, fr, m, "ci_lo"), g(X, fr, m, "ci_hi")) for m in ["aiou", "kendall_tau", "direction_acc", "tau_cp", "strict_order_acc", "loop_orient_acc"]) + " |")
        Y = PD[(PD.set == s) & (PD.cluster == clname)]
        L += ["", "| paired vs identity | AIoU Δ | tau Δ | direction Δ | loop orient Δ |", "|---|---|---|---|---|"]
        for fr in ["mirror_x", "vt"]:
            if not len(Y[Y.frame == fr]): continue
            cells = []
            for m in ["aiou", "kendall_tau", "direction_acc", "loop_orient_acc"]:
                x = Y[(Y.frame == fr) & (Y.metric == m)].iloc[0]; cells.append(ci(x.delta, x.ci_lo, x.ci_hi))
            L.append(f"| {fr} − identity | " + " | ".join(cells) + " |")
        L.append("")
        for fr in ["identity", "mirror_x", "vt"]:
            for m in ["aiou", "kendall_tau", "direction_acc"]:
                if len(X[(X.frame == fr) & (X.metric == m)]):
                    rows_csv.append(dict(check=f"K1.report.{s}.{clname}.{fr}.{m}", kill_check="K1 (reported, not gated)", set=s, frame=fr, metric=m, value=g(X, fr, m), ci_lo=g(X, fr, m, "ci_lo"), ci_hi=g(X, fr, m, "ci_hi"), rule=f"{clname}-cluster CI", outcome="reported"))
# released arms
L += ["### Phase 2 arms released by K1 (per the plan; mechanical mapping)", ""]
for reading, w in [("AND", wa), ("OR", wo)]:
    if w:
        L.append(f"- Reading {reading} (winner {w}): C1 2×2 at p01 — T_C1_p01_frame_replay (primary C1 arm) and T_C1_p01_frame_only, in frame {w}; the identity/replay cell is T_C3_p01_q25 (C3 track, gated on K3) and identity/none is T_COMP_p01; plus T_C1_p10_frame_replay; their S_C1_* sampling jobs; the B200 all-scripts arm named in the track brief (not a row of queue_candidates_2026-10-28.csv). T_C1C2_composed_p01 (+ S_C1C2_composed_p01) needs K1 AND K2: " + a.k2_note)
    else:
        L.append(f"- Reading {reading}: winner none → no C1 arm, no C1+C2 composed arm, no frame-based B200 all-scripts arm released; prereg decision text for this case: C2 + C3 with C3 as control.")
L += ["", "### Provenance", ""]
for k, v in P.items(): L.append(f"- {k}: {v}")
open(os.path.join(a.out, "KILLCHECK.md"), "w").write("\n".join(L) + "\n")
pd.DataFrame(rows_csv).to_csv(os.path.join(a.out, "killcheck_verdicts.csv"), index=False)
print("written", len(L), "lines;", len(rows_csv), "rows")
