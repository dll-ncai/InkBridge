"""Build RESULTS_routeA.md from the Route A table CSVs. python build_routeA_md.py DIR VERSION_NOTE"""
import sys, json, pandas as pd, numpy as np
D = sys.argv[1]; NOTE = sys.argv[2] if len(sys.argv) > 2 else ""
prov = json.load(open(f"{D}/PROVENANCE_routeA_tables.json")); res = pd.read_csv(f"{D}/results_routeA_sharing_cost.csv"); pr = pd.read_csv(f"{D}/paired_routeA.csv")
N = res[res.protocol == "native"].set_index(["set", "method"]); P = pr[pr.protocol == "native"]
M = [("aiou", "AIoU"), ("ldtw", "LDTW"), ("kendall_tau", "tau"), ("tau_cp", "tau_cp"), ("direction_acc", "direction"), ("strict_order_acc", "strict order"), ("penup_f1_v2", "pen-up F1 v2")]
SETLAB = dict(zip(res.set, res.set_label)); FT = ["FT_p01_ms4", "FT_p10_ms10", "FT_p30_ms10", "FT_p100_ms9"]
CTRL = sorted({m for (s, m) in N.index if m.startswith("E0scratch_")}, key=lambda x: int(x.split("_")[1][1:]))
def has(s, m): return (s, m) in N.index
def cell(s, m, met):
    r = N.loc[(s, m)]; v = r[met]
    return f"{v:.4f} [{r[met+'_lo']:.4f}, {r[met+'_hi']:.4f}]" if met in ("aiou", "kendall_tau") else f"{v:.4f}"
def row_tab(s, methods, labels):
    out = ["| method | n | " + " | ".join(l for _, l in M) + " |", "|---|---|" + "---|" * len(M)]
    for m, l in zip(methods, labels):
        if not has(s, m): continue
        out.append(f"| {l} | {int(N.loc[(s, m)].n)} | " + " | ".join(cell(s, m, met) for met, _ in M) + " |")
    return "\n".join(out)
def dl(s, prefix, met, exact=None):
    q = P[(P.set == s) & (P.metric == met) & ((P.contrast == exact) if exact else P.contrast.str.startswith(prefix))]
    if len(q) == 0: return "n/a"
    r = q.iloc[0]; lo, hi = (r.ci_lo_writer, r.ci_hi_writer) if r.n_writers > 2 else (r.ci_lo_unit, r.ci_hi_unit)
    return f"{r.delta:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if (lo > 0 or hi < 0) else ''}"
def delta_tab(s, contrasts):
    out = ["| contrast | " + " | ".join(l for _, l in M) + " |", "|---|" + "---|" * len(M)]
    for cs, lab in contrasts:
        if len(P[(P.set == s) & P.contrast.str.startswith(cs)]) == 0: continue
        out.append(f"| {lab} | " + " | ".join(dl(s, cs, met) for met, _ in M) + " |")
    return "\n".join(out)
SPEC = {"test_ouhdl": ("E0_ms8", "E0 ms8 (Urdu specialist)"), "test_iam500": ("X5_ms9", "X5 ms9 (IAM specialist)"), "test_casia_zh": ("E27_ms47", "E27 ms47 (CASIA specialist)"), "test_casia_en": ("X5_ms9", "X5 ms9 (English specialist)"), "test_hpl_tamil": (None, None)}
g = lambda s, m, met: N.loc[(s, m)][met]
md = [f"""# Route A - cost of sharing: specialist vs joint M5 vs LUO (Urdu left out), German zero-shot, Urdu low-resource curve

**Status: EXPLORATORY.** {NOTE} Test sets read directly; one training seed per model; sampling seeds 0-2 on the VPS RTX 4090 (DDIM-50, eta 0,
EMA, rendered 64x64 condition, L=320) except the E0-from-scratch controls and E0 ms8 on gold, sampled seeds 0-2 on an CLUSTER-A GPU (job 24620927);
scorer uhwr_eval 0.3.1 (native protocol below; standardised rows in the CSV); writer-cluster percentile bootstrap B=2000 on the per-unit
seed means (unit-cluster CI where a set has <= 2 writers: OnHW). `*` = 95% CI excludes 0. Built {prov['built_utc']} from {prov['n_rows']:,} rows
(CLUSTER-A Slurm {prov['slurm_job']}, host {prov['host']}).

Models: specialists E0 ms8 (OUHD-L Urdu), X5 ms9 (IAM English), E27 ms47 (CASIA, partial: 470k of 500k steps); joint M5 ms9 (5 scripts,
90k units); LUO ms6 (M5 recipe with Urdu left out); LUO-FT p01 ms4 / p10 ms10 / p30 ms10 / p100 ms9 (LUO fine-tuned on 1/10/30/100 % of
OUHD-L train); E0-from-scratch controls E0scratch_p01/p10/p30 (E0 recipe, 10k steps, same FT subsets, val-selected milestone in the name).
Specialist and joint rows are the Stage A prediction files (same device, same seeds); RB4 rows (CLUSTER-A A30, seed 0, evaluator rows without
pen-up F1 v2) are the reference for the German X5 / E0 numbers and the device check.

## Headline (native protocol, AIoU unless stated)
- Cost of sharing (joint M5 - specialist, paired): Urdu {dl('test_ouhdl','joint - specialist','aiou')} AIoU but tau {dl('test_ouhdl','joint - specialist','kendall_tau')}; IAM {dl('test_iam500','joint - specialist','aiou')} AIoU, tau {dl('test_iam500','joint - specialist','kendall_tau')}; CASIA en {dl('test_casia_en','joint - specialist','aiou')} (X5 is out of domain there); CASIA zh {dl('test_casia_zh','joint - specialist','aiou')} (confounded, see note b).
- Urdu in the mix helps the other scripts (LUO - joint, paired AIoU): IAM {dl('test_iam500','LUO - joint','aiou')}, zh {dl('test_casia_zh','LUO - joint','aiou')}, en {dl('test_casia_en','LUO - joint','aiou')}, Tamil {dl('test_hpl_tamil','LUO - joint','aiou')}; order metrics move little (tau IAM {dl('test_iam500','LUO - joint','kendall_tau')}).
- Zero-shot German: joint {g('german_unipen_de_test','M5_joint_ms9','aiou'):.4f} / LUO {g('german_unipen_de_test','LUO_ms6','aiou'):.4f} / X5 {g('german_unipen_de_test','X5_ms9_rb4','aiou'):.4f} / E0 {g('german_unipen_de_test','E0_ms8_rb4','aiou'):.4f} on UNIPEN-de; {g('german_onhw_all','M5_joint_ms9','aiou'):.4f} / {g('german_onhw_all','LUO_ms6','aiou'):.4f} / {g('german_onhw_all','X5_ms9_rb4','aiou'):.4f} / {g('german_onhw_all','E0_ms8_rb4','aiou'):.4f} on OnHW (RB4 values reproduced within 0.002).
- Urdu low-resource curve (test 4,406): LUO-FT p01 {g('test_ouhdl','FT_p01_ms4','aiou'):.4f}, p10 {g('test_ouhdl','FT_p10_ms10','aiou'):.4f}, p30 {g('test_ouhdl','FT_p30_ms10','aiou'):.4f}, p100 {g('test_ouhdl','FT_p100_ms9','aiou'):.4f} vs E0 ms8 {g('test_ouhdl','E0_ms8','aiou'):.4f}"""]
if CTRL:
    md.append(f"  vs E0 from scratch at the same fraction: " + ", ".join(f"{m.split('_')[1]} {g('test_ouhdl', m, 'aiou'):.4f}" for m in CTRL if has('test_ouhdl', m)) + "; FT - from-scratch (paired AIoU): " + ", ".join(f"{ft.split('_')[1]} {dl('test_ouhdl', f'{ft} - E0scratch', 'aiou')}" for ft in FT[:3]) + ".")
md.append(f"""- Forgetting: every Urdu fine-tune collapses on German (UNIPEN-de AIoU {g('german_unipen_de_test','FT_p01_ms4','aiou'):.3f}-{g('german_unipen_de_test','FT_p100_ms9','aiou'):.3f}, tau <= {max(g('german_unipen_de_test',ft,'kendall_tau') for ft in FT):.2f}; direction 0.36-0.44 = writing German right-to-left).

## 1. Cost of sharing per script (native protocol)
""")
for s in ["test_ouhdl", "test_iam500", "test_casia_zh", "test_casia_en", "test_hpl_tamil"]:
    sp, spl = SPEC[s]; methods = ([sp] if sp else []) + ["M5_joint_ms9", "LUO_ms6"]; labels = ([spl] if sp else []) + ["M5 joint ms9", "LUO ms6 (Urdu left out)"]
    md.append(f"### {SETLAB[s]}\n\n" + row_tab(s, methods, labels) + "\n\nPaired per-unit deltas (writer-cluster CIs):\n\n" + delta_tab(s, [("joint - specialist", "joint - specialist"), ("LUO - specialist", "LUO - specialist"), ("LUO - joint", "LUO - joint")]) + "\n")
md.append("""Notes. (a) On OUHD-L Urdu, LUO ms6 is a zero-shot model (Urdu never seen): its row is the floor of the FT curve in section 3.
(b) CASIA zh confound: 46.2 % of the 500 X4a zh units exceed L=320 points and are resampled to 320, and E27 ms47 received the X4a/E0-rendered
images rather than its own Nagamatsu input pipeline; the E27 row is therefore not E27's in-domain performance (its own protocol gives AIoU 0.814
on seen classes, RB4 X2) and the joint - specialist delta on zh is not a cost-of-sharing estimate. (c) HPL Tamil has no specialist.

## 2. Zero-shot German (UNIPEN-de test 827 units / 7 writers; OnHW 5,925 units / 2 writers)
""")
for s in ["german_unipen_de_test", "german_onhw_all"]:
    methods = ["M5_joint_ms9", "LUO_ms6", "M5_joint_ms9_rb4", "LUO_ms6_rb4", "X5_ms9_rb4", "E0_ms8_rb4"]
    labels = ["M5 joint ms9 (4090, seeds 0-2)", "LUO ms6 (4090, seeds 0-2)", "M5 joint ms9 (RB4: A30, seed 0)", "LUO ms6 (RB4: A30, seed 0)", "X5 ms9 (RB4: A30, seed 0)", "E0 ms8 (RB4: A30, seed 0)"]
    md.append(f"### {SETLAB[s]}\n\n" + row_tab(s, methods, labels) + "\n\nPaired deltas:\n\n" + delta_tab(s, [("LUO - joint (effect", "LUO - joint (4090 rows)"), ("X5 - joint (RB4", "X5 - joint (RB4 rows)"), ("E0 - joint (RB4", "E0 - joint (RB4 rows)"), ("LUO - joint (RB4", "LUO - joint (RB4 rows)"), ("joint 4090 3-seed - joint RB4", "device check: joint 4090 3-seed - joint RB4 A30 seed 0"), ("LUO 4090 3-seed - LUO RB4", "device check: LUO 4090 3-seed - LUO RB4 A30 seed 0")]) + "\n")
md.append("""Notes. Set-level AIoU of the 4090 3-seed rows and the RB4 A30 seed-0 rows agree within 0.002 for joint and LUO (device check rows),
so the RB4 X5 / E0 numbers are comparable at the set level; the per-unit device-check deltas have wide CIs because individual samples differ
across devices. tau_cp is NOT comparable between 1-seed and 3-seed rows: it sets undefined tau to 0 per unit after the seed mean, so a unit
with tau defined in one of three seeds counts as defined (3-seed rows are higher by construction). RB4 report values for reference:
UNIPEN-de AIoU E0 0.5886 / joint 0.7921 / LUO 0.7794 / X5 0.7768; OnHW 0.6520 / 0.7554 / 0.7433 / 0.7765 (identical to the RB4 rows here).

## 3. Urdu low-resource curve: LUO-FT p01/p10/p30/p100 vs E0 ms8 (full data) and vs E0 from scratch at the same fraction
""")
for s in ["test_ouhdl", "gold_ouhdl"]:
    methods = ["E0_ms8", "M5_joint_ms9", "LUO_ms6", "FT_p01_ms4", "FT_p10_ms10", "FT_p30_ms10", "FT_p100_ms9"] + CTRL
    labels = ["E0 ms8 (100 %, from scratch, 4090)" if s == "test_ouhdl" else "E0 ms8 (100 %, from scratch, CLUSTER-A GPU)", "M5 joint ms9", "LUO ms6 (0 %, zero-shot)", "LUO-FT p01 ms4", "LUO-FT p10 ms10", "LUO-FT p30 ms10", "LUO-FT p100 ms9"] + [f"E0 from scratch {m.split('_')[1]} ({m.split('_')[2]}, CLUSTER-A GPU)" for m in CTRL]
    md.append(f"### {SETLAB[s]}\n\n" + row_tab(s, methods, labels) + "\n\nPaired deltas:\n\n" +
              delta_tab(s, [(f"{ft} - E0_ms8", f"{ft} - E0 ms8") for ft in FT] + [(f"{ft} - E0scratch", f"{ft} - E0 from scratch (same fraction)") for ft in FT[:3]] + [(f"{m} - E0_ms8", f"{m} - E0 ms8") for m in CTRL] +
                           [("FT_p100_ms9 - joint", "FT_p100 - joint"), ("FT_p10_ms10 - FT_p01", "p10 - p01"), ("FT_p30_ms10 - FT_p10", "p30 - p10"), ("FT_p100_ms9 - FT_p30", "p100 - p30")]) + "\n")
md.append("""Device note: LUO-FT and E0 ms8 (test) rows were sampled on the VPS 4090; the E0-from-scratch controls and E0 ms8 (gold) on an CLUSTER-A GPU
(job 24620927, device in the sidecar json). Set-level agreement across these devices is within 0.002 AIoU (section 2 device check), per-unit
samples differ, so the FT - from-scratch paired CIs are somewhat wider than same-device CIs. The controls ran 10k steps at effective batch 512
(E0 recipe, lr 8e-5, seed 20261016) with the E0 val rule (lowest native LDTW on a 1,000-unit val subset) selecting the milestone.
Reference values from the CLUSTER-A M5 record (A30, seed 0): FT AIoU p01 0.7522, p10 0.7842, p30 0.8015, p100 0.8075; E0 ms8 0.7987.\nBudget caveat for the controls: E0 from scratch at p01/p10/p30 trained 10k steps at effective batch 512 (the same optimisation budget as the LUO-FT runs, whose\n4k-10k fine-tuning steps start from the 60k-step LUO model), whereas E0 ms8 is the 80k-step milestone of the full-data run. The FT - from-scratch deltas therefore measure\ntransfer from the four-script pretraining at a matched fine-tuning budget, not the asymptote of from-scratch training on the subset. The from-scratch models mostly emit\nsingle-stroke trajectories (tau_cp 0.05-0.78 with tau defined on few units; strict order 0.25-0.57), which is why their tau looks high while tau_cp and pen-up F1 are low.\n

## 4. What Urdu fine-tuning does to the held-out scripts (German, zero-shot)
""")
for s in ["german_unipen_de_test", "german_onhw_all"]:
    md.append(f"### {SETLAB[s]}\n\n" + row_tab(s, ["LUO_ms6"] + FT, ["LUO ms6", "LUO-FT p01", "LUO-FT p10", "LUO-FT p30", "LUO-FT p100"]) + "\n\nPaired deltas FT - LUO:\n\n" + delta_tab(s, [(f"{ft} - LUO", f"{ft} - LUO") for ft in FT]) + "\n")
md.append("""Every Urdu fine-tune loses the LTR scripts it is no longer trained on: AIoU falls by 0.06-0.13 (UNIPEN-de -0.077 to -0.131, OnHW -0.059 to -0.071)
and Kendall tau turns negative (the fine-tuned models write the German words right-to-left), at every fraction including p01 (4k steps). This is the
forgetting cost that D1 (token-routed joint), D2 (frozen backbone + adapters) and D3 (routed specialists / distilled student) are meant to avoid.

## 5. Files
- `results_routeA_sharing_cost.csv` - per set x method x protocol: means, writer-cluster CIs, seed SD, n defined, prediction files + sha256
- `paired_routeA.csv` - paired per-unit deltas (writer- and unit-cluster CIs, fraction of units positive/negative) for every contrast above
- `results_routeA_per_seed.csv` - set means per seed; `inputs_routeA.csv` - which rows went in (set, method, seed, source)
- `PROVENANCE_routeA_tables.json` - builder sha256, row files, Slurm job
- Per-unit rows: CLUSTER-A `exp/D3_ROUTE/routeA/score_out/rows/` (Route A, Slurm array 24621997), `exp/D3_ROUTE/stageA/score_out/rows/` (specialists, joint; 24619444), `exp/RB4_baselines/german/rows/`, `exp/D2_MOE/controls/score_out/rows/` (controls + E0 ms8 gold; array 24620928)
- Predictions: CLUSTER-A `exp/D3_ROUTE/routeA/preds/` (69 files; VPS runner jobs 170-183), `exp/D2_MOE/controls/preds/` (21 files, job 24620927; SHA256SUMS.txt); control runs `exp/D2_MOE/controls/p{01,10,30}/` (jobs 24618752/53/54, eval/best_milestone.txt)
- Scoring code: `exp/D3_ROUTE/routeA/code/` (routeA_score.py, routeA_tables.py; same evaluator call as Stage A / table builder)
""")
open(f"{D}/RESULTS_routeA.md", "w").write("\n".join(md)); print("written", f"{D}/RESULTS_routeA.md")
