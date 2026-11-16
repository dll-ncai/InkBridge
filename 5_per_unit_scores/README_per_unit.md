# Per-unit score tables

Source: the scored rows files `<set>__<tag>__seed<k>__rows031.csv` written by `lane_score.py` with uhwr_eval 0.3.1. Each rows file has two rows per unit (protocols native and standardised) and 44 columns.
Reduction applied on the cluster (ARCHIVE_BUILD/stage2.py): rows with protocol = native only, which is the protocol the table scripts read; columns kept: unit, error flag, and the six metrics the table scripts read (aiou, penup_f1_v2, strict_order_acc, direction_acc, kendall_tau, ldtw) as float64 parsed with round-trip precision. No averaging over seeds and no rounding. The sha256, size and date of every source rows file, and its evaluator, prediction and ground-truth checksums, are in FILES_INDEX.csv.

## tables_5_6_8_10_11_12/ (1,192 rows files, 2,215,628 unit rows, errors 0)
| dir                                             |   files |   tags |   unit_rows |
|:------------------------------------------------|--------:|-------:|------------:|
| exp/CAND/method_round/COMP/score_out/4090/rows  |      27 |      3 |       65910 |
| exp/D3_ROUTE/routeA/score_out/rows              |      39 |      3 |      106422 |
| exp/D3_ROUTE/stageA/score_out/rows              |      24 |      2 |       52692 |
| exp/FMD/xs/runs/fmdx_monitor/score_out/rows     |     114 |      6 |      183972 |
| exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows |      45 |      4 |       92166 |
| exp/RQF/R3/score_out/rows                       |     136 |     23 |      227852 |
| exp/RQF/score_out/rows                          |     807 |     39 |     1486614 |

## tables_13_and_reference_systems/ (249 rows files, 412,488 unit rows, errors 0)
| dir                                |   files |   tags |   unit_rows |
|:-----------------------------------|--------:|-------:|------------:|
| exp/D3_ROUTE/routeA/score_out/rows |      30 |      4 |       71196 |
| exp/D3_ROUTE/stageA/score_out/rows |      51 |      4 |       83448 |
| exp/RQF/R1/score_out/rows          |      84 |      8 |       99948 |
| exp/RQF/R1d/score_out/rows         |      84 |     12 |      157896 |

## table_15_selectivity/
e1_v2_scores_per_unit.csv.gz: per-unit scores of the perturbation runs behind the metric-selectivity table (all columns).

## Files
- `rows__<source dir>__<set>.npz`: arrays file_idx (row of FILES_INDEX.csv), unit_idx (row of UNITS_<set>.tsv), error, metrics (n x 6), metric_names.
- `UNITS_<set>.tsv`: unit_idx, unit_id, writer_id (writer clusters of the bootstrap).
- `FILES_INDEX.csv`: one row per source rows file: dir, name, set, tag (checkpoint tag), seed (sampling seed), source sha256/bytes/mtime, row counts, evaluator and input checksums.
- `../load_rows.py`: loader.

## Check
CHECK_reproduce_T5_T6.csv: the pooled paired deltas of Tables 5 and 6 (two warm variants, six metrics) recomputed from these files (mean over sampling seeds per unit, mean over the three training seeds, minus the pooled comparator, mean over units) against T56_per_seed.csv. Largest absolute difference: 9.7e-17. Intervals were not recomputed.
Tag conventions for that check: 1 % warm arms = `<ARM>_ft_ms*` (first training seed, monitor folder) and `<ARM>_p01_s<seed>_ft_ms*`; 10 % = `<ARM>_p10_s<seed>_ft_ms*`; comparators = `comp_p01_identity_s<seed>_*` in the score_out_cmp folder (1 %) and `LUOFT_p10_s<seed>_*` (10 %); ARM = NOFM_W (unguided warm) or PEINIT_W (PE-initialised warm).

## Left out (size limit of about 100 MB)
Scored rows files that exist on the cluster but are not in this archive (1,541 files, 3.94 GB as CSV), by folder:
| dir                                             |   files |   tags |   GB_csv |
|:------------------------------------------------|--------:|-------:|---------:|
| exp/CAND/method_round/COMP/score_out/4090/rows  |      15 |      3 |     0.01 |
| exp/FMD/xs/runs/fmdx_monitor/score_out/rows     |     570 |     30 |     1.26 |
| exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows |       9 |      3 |     0.01 |
| exp/RQF/score_out/rows                          |     947 |     48 |     2.65 |
These are the replay single-model arms (SMR_*), adapter arms at 10 %, step-matched controls (LUOC40*), 30 % arms, the other foundation-model warm arms of the screen, and scores on validation, gold and mirrored sets. The full list is ARCHIVE_BUILD/rows_listing_all.csv (column in_archive).
Also left out: the standardised-protocol rows and the 38 other columns of each rows file.

## Added in the rebuild
- `UNITS_german_unipen_de_test.tsv` and `UNITS_german_onhw_all.tsv` hold pseudonymised unit IDs and renumbered writer indices (see the top-level README); the npz files are unchanged.
- `tables_13_and_reference_systems/trace_line_rows/`: four scored rows files of the TRACE baseline on the 253 OUHD-L test lines (identity and mirrored frame), all columns.
- CHECK_reproduce_T10.csv: deltas of the eight rows of Table 10 recomputed from the per-unit rows.

## Added in the public repository
- `reproduce_warm_deltas.py`: recomputes the 24 pooled differences of CHECK_reproduce_T5_T6.csv from these files and compares them with T56_per_seed.csv (numpy and pandas only).
- `ARCHIVE_BUILD/` named above is at `provenance/archive_v11/ARCHIVE_BUILD/`.
