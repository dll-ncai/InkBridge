# checks_v12 summary (InkBridge v12 review checks, existing cluster outputs)

Run: cluster Slurm 25171113 on <node> (cpuidle), 2026-11-15T16:34:19Z to 2026-11-15T16:38:39Z; script checks_v12.py sha256 5a7e55355433992a; 604 input files hashed in PROVENANCE_checks_v12.json; outputs on the cluster in exp/RV12/checks_v12/out_r2. Run 1 (Slurm 25171088) was cancelled because the script lacked a main guard; its folders out/ and smoke/ are not used. No training and no sampling were run.

## randctl (review item 7)
- Cells: 24 non-Urdu (4 bases x 6 sets, exp/RQF/R3/read_all3) and 6 Urdu (5 bases of read_all3 and the FT1% cell of exp/ORDER/sealed/COMP_rep3); 8 metrics per cell; 240 rows in checks_v12_results.csv.
- Same units, models, settings: unit sets equal in 30 of 30 cells; file stems of selected and candidate rows match in 30 of 30; ground-truth hash and evaluator equal in 30 and 30 of 30; 87,760 of 87,760 selected rows equal exactly one of their three candidate rows on 10 score columns; sampling sidecars of the three candidates (found for 41 of 44 model-set pairs; the 3 pairs of the ORDER sealed cell were not checked) differ only in: peak_mem_gib (seed, hash, path and time keys excluded).
- Selected minus mean of three candidates, recomputed with the paper's writer-cluster bootstrap (B=2000, rng 20261029): equals the stored rep3-pool3 contrast (max absolute difference 1.0e-16 on the estimate, 2.4e-16 on the bounds).
- Non-Urdu, cells (of 24) with the 95% interval above / below zero, and the same with Bonferroni-8 intervals: aiou 24/0 (24/0); penup_f1_v2 21/0 (20/0); strict_order_acc 17/0 (12/0); direction_acc 12/0 (4/0); kendall_tau 1/4 (1/2); tau_cp 3/0 (3/0); ldtw 0/17 (0/12); stroke_count_err 0/15 (0/10).
- Urdu, cells (of 6) with the 95% interval above / below zero: aiou 6/0; penup_f1_v2 6/0; strict_order_acc 6/0; direction_acc 1/0; kendall_tau 0/0; tau_cp 5/0; ldtw 0/6; stroke_count_err 0/6; pen-up F1 gain +0.0142 to +0.0288, strict order +0.0126 to +0.0198; the stored random pick minus the mean has 0 of 48 intervals excluding zero.
- Counts over the 24 non-Urdu cells: gain rule on selected minus mean (with family Holm) 16; gain rule and pre-specified flag 9; gain rule and randomization control 15.
- Pre-specified flag (v11): tests one stored random draw (rnd3) against the same mean and needs all 8 unadjusted 95% intervals to contain zero; 13 of 24 cells pass (recomputed: 13); 24 of 192 cell-metric intervals exclude zero, 16 of them on OnHW (2 writer clusters).
- Randomization control (v11, post hoc): tests the stored draw against 2,000 redraws with Bonferroni over 8 metrics; 23 of 24 cells pass (file L08_control_per_cell.csv).
- All metrics accounted for by Bonferroni-8 intervals on the stored draw minus the mean: 16 of 24 cells pass, 20 of 192 intervals exclude zero, and the gain rule with this control gives 10 of 24.
- Agreement: the three counts are 16, 9 and 15 of 24; the 9 cells are a subset of the 15, and the 15 of the 16. Cells in the 16 but not the 9: COMP_p01|german_onhw_all, COMP_p01|test_casia_en, COMP_p01|test_hpl_tamil, M5|german_unipen_de_test, NOFM_W_comp|german_onhw_all, NOFM_W_comp|german_unipen_de_test, NOFM_W_comp|test_casia_zh. Cell in the 16 but not the 15: COMP_p01|test_casia_en.
- The comparison in review item 7 uses no random draw: the mean of the three candidates is the comparator of the paper's gain test, so it yields the 16 and neither the 9 nor the 15.
- Kendall tau and direction are averaged per unit over the samples where they are defined; with all three candidates and the selected sample required to be defined, the estimate differs by at most 0.0445 (non-Urdu) and uses 43 to 2517 units for Kendall tau; the other six metrics are identical under both pairings.

## ft1abs (review item 14)
- Plain FT1% on the 4,406 common units (32 writers), same functions and bootstrap as absolute_common_units.csv (writer-cluster multinomial, rng 0, B=2000): AIoU 0.7553 [0.7515, 0.7592], pen-up F1 0.7383, strict order 0.6528, direction 0.9316, tau_cp 0.8052 (defined units: direction 4400, tau_cp 2691; 9 row files, RTX4090).
- These are the 9 candidate files (3 training seeds x 3 sampling seeds, per-unit mean) from which the 'FT1% + endpoint selection' row is selected (file list equals the ORDER sealed spec: True).
- Stored 'FT1% + endpoint selection' row recomputed: AIoU 0.7645 [0.7605, 0.7683], pen-up F1 0.7671, strict order 0.6714, direction 0.9307, tau_cp 0.7743 (defined units: direction 4394, tau_cp 2691; 3 row files, RTX 4090 (verifier on CPU)); the 9 stored rows are reproduced with max absolute difference 1.1e-16.
- Same model on the other device (comparator rows of tab:warm, fmdx_monitor/score_out_cmp): AIoU 0.7558 [0.7515, 0.7599], pen-up F1 0.7387, strict order 0.6530, direction 0.9312, tau_cp 0.7986 (defined units: direction 4402, tau_cp 2691; 9 row files, RTX A4000 (per v10 table-source record)).

## n4358 (review item 12)
- Both arms (leave-Urdu-out backbone, identity and mirror_x frame, 3 sampling seeds each) have 4,406 units with AIoU; direction accuracy is defined in 4,373 units (identity), 4,388 (mirror) and 4,358 in both.
- The 48 excluded units have direction undefined in all three samples of an arm (no stroke eligible for direction scoring, n_dir_eligible = 0): 30 in identity only, 15 in mirror only, 3 in both; none has an empty prediction; they come from 26 writers.
- On the 4,358 units: direction disagreement >= 0.5 in 1,863 (42.7%), AIoU difference <= 0.01 in 676 (15.5%), both in 276 (6.3%); equal to the M3 file values. Mean AIoU 0.7129 against 0.7102 (0.7117 against 0.7090 on all 4,406); the AIoU share on all 4,406 units is 15.55%.

## inkinv (review item 12)
- Evaluator uhwr_eval 0.3.1 (OpenCV 4.11.0), native protocol, 500 ground-truth units from each of 7 test sets (3,500 units, 1,963 with two or more strokes, rng 20261115); OUHD-L, UNIPEN-de and OnHW are samples of 4,406, 827 and 5,925 units, the other four sets are complete.
- AIoU of the ground truth against itself with strokes changed: reversed point order 0 of 3,500 units changed; stroke order reversed 0; stroke order shuffled 0; both 0; maximum absolute change 0.0; the same with the changed ink as the reference.
- Evaluation steps: converted points, integer pixel coordinates and rendered masks (thickness 1 and 2) are identical in 3,500 of 3,500 units for every variant; no step changes AIoU.
- The variants were applied: direction accuracy is 1.0 for the unchanged ink and 0.0 with reversed points; strict order is 1.000 unchanged and 0.439 with reversed stroke order.
- Mean AIoU of a ground truth against itself is 0.8795; 21 of 3,500 units contain a stroke with fewer than two points, which the evaluator's renderer does not draw.

## fmscreen (review item 12/14, count check)
- exp/FMD/xs/read_v1/v1_1_fmdx_gate_read.csv (FINAL_2026-11-07T0047Z, 116 rows): PASS_XS 19, PASS_URDU_ONLY 6, FAIL 85, not assessed 6 (4: screen cut-off (A3.c.3); 1: no public weights (verdicts/blocked/blocked_inkfm.json); 1: no extractor (A3.a.2/A3.b; verdict status_reason)); sum 116. The 116 verdict files give the same counts (the 6 as PENDING).
- v11 states 19, 6, 85, 6 of 116; the task text's '9 pass' does not match the file (19). One of the 19 PASS_XS verdicts has urdu_pass = False.
