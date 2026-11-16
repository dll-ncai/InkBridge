# InkBridge reviewer archive (v11)

Built 2026-11-15 for peer review. It holds the evaluator, the table scripts, run configurations, unit-ID lists, the OUHD-L ligature-unit boundaries, per-unit score tables and the table-level result files of the paper. It was built from existing records; no number was recomputed except the checks listed below.

## Layout
| folder | content |
|---|---|
| `1_evaluator_and_table_code/` | `uhwr_eval-0.3.1/` (package, metric registry, release records; the release tarball is unpacked in `release/uhwr_eval-0.3.1.tar.gz__unpacked/`; sha256 of the original tarball f8d967f56726f77f7384d7a78a2617ef5da6feeb00e5c1fc46cbc4666b244dc8); scoring entry `exp/D3_ROUTE/routeA/code/lane_score.py`; bootstrap and contrast scripts of every table under `exp/.../code/`; code tarballs unpacked into `<name>__unpacked/` |
| `2_run_configs/` | `run_config.json`, `job_history.txt`, `training_complete.json` of each run; arm tables and queues (`admin/`); pre-registrations and amendments (`plans/`); experiment provenance files; job scripts |
| `3_unit_ids/` | `lists/` (one TSV per test set and training subset), `UNIT_ID_INDEX.md` (sampling rule and seed of each list), `source_records/` (data cards, build records, router manifests) |
| `4_ouhdl_division_v3/` | unit boundaries for all 42,370 OUHD-L units (CSV), ground-truth ink of the test and val units (npz), build code, `SEGMENTATION_PROCEDURE.md` |
| `5_per_unit_scores/` | per-unit scores behind Tables 5, 6, 8, 10, 11, 12 (and 13, 15), `README_per_unit.md`, `load_rows.py` |
| `6_table_sources/` | table-level CSV/JSON/Markdown results at their cluster-relative paths |
| `supplementary/` | recipe, screen-roster, run-date and milestone-validation tables; facts pack and L08/L14 analysis tables; router 1 %-label ablation (SPEC, results, per-unit, checkpoint checksums); related-work search records; see `supplementary/README.md` |
| `ARCHIVE_BUILD/` | specification and collection scripts of this archive, list of all scored rows files |
| `MANIFEST.sha256`, `PROVENANCE.json`, `NOT_INCLUDED.md` | checksums; per-file source, source sha256, record date, sharing note; items left out |

Path convention: below each numbered folder, files keep their path relative to the project root on the cluster (`$U`) or the local project folder (`$LOCAL`).

## Paper table -> files
| Table | files | check done at build time |
|---|---|---|
| 3 (test sets) | 3_unit_ids/lists/*.tsv; 3_unit_ids/UNIT_ID_INDEX.md | unit and writer counts recomputed from the lists (CHECKS_unit_ids.json) |
| 4 (cross-script failure) | 6_table_sources/exp/RQF/R1/tables/R1_absolute.csv, R1_paired.csv, RESULTS_R1a.md, RESULTS_R1b.md, RESULTS_R1c.md, R1_verdicts.json (per-set mirrored-control tables) | 9 of 15 printed numbers appear in R1_absolute.csv (single values); the other 6 were not traced |
| 5 (1 %, warm minus plain fine-tune) | 6_table_sources/exp/RQF/tables/reviewer_followup_v1/T56_per_seed.csv; script 1_evaluator_and_table_code/exp/RQF/code/reviewer_v1/reviewer_followup.py; per-unit: 5_per_unit_scores/tables_5_6_8_10_11_12 | 10 of 10 printed delta [CI] triples found in the CSV; pooled deltas recomputed from the per-unit rows (CHECK_reproduce_T5_T6.csv) |
| 6 (10 %) | same files as Table 5 (rows T6_p10_vs_FT10) | 10 of 10 triples found; pooled deltas recomputed from the per-unit rows |
| 7 (absolute scores) | 6_table_sources/exp/RQF/tables/paired_ext/absolute_common_units.csv; script 1_evaluator_and_table_code/exp/RQF/code/rqf_paired_ext.py; PEN-Net per-unit rows: 6_table_sources/exp/PENNET/results/test_rendered_ms10_rows.csv | 9 of 9 triples found; per-unit rows of the adapter, Urdu-only and InkSight systems are not in the archive |
| 8 (retention on six sets) | 6_table_sources/exp/RQF/tables/RQF_R2_contrasts.csv; script 1_evaluator_and_table_code/exp/RQF/code/rqf_v1/build_tables_rqf.py; per-unit: tags *_cmp_* and LUO_ms6* | 33 of 36 triples found in the CSV; the other 3 were not traced |
| 9 (German direction) | 6_table_sources/exp/RQF/tables/RQF_R2_R2e_unipen_de_per_writer.csv | per-writer values; the table prints unit-weighted aggregates and ranges, 6 of 16 printed numbers appear directly |
| 10 (single-model variants) | 6_table_sources/exp/RQF/tables/RQF_SM_contrasts.csv, 6_table_sources/exp/RQF/tables/RQF_SM_solves.csv, 6_table_sources/exp/RQF/tables/RQF_SM_provenance.json; script 1_evaluator_and_table_code/exp/RQF/code/sm_v1/sm_read.py; per-unit: tags SMA_*_p01_* | 32 of 32 printed delta (lower bound) pairs found; deltas of all 8 rows recomputed from the per-unit rows (CHECK_reproduce_T10.csv); 540 of the 1,310 rows files named in RQF_SM_provenance.json are in the archive with equal sha256, the rest are replay and 10 % arms not printed in Table 10 |
| 11 (selection on OUHD-L) | 6_table_sources/exp/RQF/R3/split_v1/R3_split_paired.csv, 6_table_sources/exp/RQF/tables/reviewer_followup_v1/R3_tau_all_cells.csv; scripts 1_evaluator_and_table_code/exp/RQF/code/rq3_v1/rq3_read.py, loop_tables.py; per-unit: folder R3 (tags *_rep3, *_rnd3) | 20 of 20 triples found |
| 12 (selection on six sets) | same files as Table 11 plus 6_table_sources/exp/RQF/R3/R3_summary_3seed_cells.csv | the table prints counts of sets; the counts were not re-derived by the archive build |
| 13 (frame choice) | 6_table_sources/exp/RQF/R1d/read/R1d_summary_paired.csv; per-unit: 5_per_unit_scores/tables_13_and_reference_systems (R1d, R1) | 21 of 21 printed numbers appear in the CSV (single values, weaker evidence than triples) |
| 15 (metric selectivity) | 6_table_sources/exp/RQF/M1_selectivity/out/m1_leak_wide_native.csv; code 1_evaluator_and_table_code/exp/RQF/M1_selectivity/code/leak_table.py; per-unit: 5_per_unit_scores/table_15_selectivity | 35 of 35 printed numbers appear in the CSV (single values) |
| 16 (settings) | 2_run_configs/exp/*/runs/*/run_config.json; 2_run_configs/exp/RQF/admin; supplementary/supp_recipes_runs_v11.csv, supp_recipes_families_v11.csv | not checked value by value |
| 17 (corpora) | 3_unit_ids/source_records/data/m5_multiscript_v1/DATA_CARD.md; data cards under 3_unit_ids/source_records/data/d2_v2 and exp/X4a | not checked value by value |

Statistics of the contrast tables (from the script headers): per-unit value = mean over sampling seeds 0-2 of the native-protocol rows; contrast = paired per-unit difference on common units; 95 % percentile interval from 2,000 writer-cluster bootstrap resamples (units on OnHW, which has two writers).

## What is not included
- **Model weights.** No checkpoint of any model is in the archive (backbones, warm backbones, fine-tunes, adapters, router classifier). Checksums of the router checkpoints are in `supplementary/router336_checkpoints_sha256.csv`.
- **Ink, images and transcripts of the third-party corpora.** For CASIA, IAM-OnDB, UNIPEN, HPL Tamil and OnHW the archive holds identifiers, writer indices and counts only. Ground-truth ink is included for OUHD-L only (test and validation units; the 95 MB training file is left out and is rebuilt with the included code from Zenodo doi 10.5281/zenodo.20642162).
- **Per-unit score files left out for size.** 1,541 of the 2,982 scored rows files (3.94 GB as CSV) are not included: replay single-model arms, adapter arms at 10 %, step-matched controls, 30 % arms, the other foundation-model warm arms of the screen, and scores on validation, gold and mirrored sets. The included 1,441 files hold the native-protocol rows and six metrics; the standardised-protocol rows and the other 38 columns are left out. Per-unit rows of the adapter, Urdu-only and InkSight systems of Table 7 are not included.
- **Predicted trajectories** are not included.
- **Original UNIPEN and OnHW identifiers** (see next section).
Details and the list of skipped source files: `NOT_INCLUDED.md`.

## Pseudonymised identifiers (UNIPEN, OnHW)
The UNIPEN conditions of use ask that writers are not named. All UNIPEN and OnHW writer codes, source IDs and unit IDs in this archive are pseudonyms (`DE-A##`, `DE-S##`, `DE-O##`; unit IDs `<pseudonym>/<contributor>_<index>_u<k>`), taken from a pseudonymised copy of the unit tables made on 2026-10-23; writer indices were renumbered with them. The same pseudonyms are used in the ID lists, router manifests, per-unit score tables and supplementary files, so the files join on unit ID. The mapping to the original IDs is not in the archive. **The original IDs are available to the editor on request.** In build records of these two corpora, source paths without a unit-level pseudonym are masked as `<unipen-path>`.

## Sharing limits
Ground-truth ink is included for OUHD-L only (Zenodo doi 10.5281/zenodo.20642162). For CASIA, IAM-OnDB, UNIPEN, HPL Tamil and OnHW: no ink, no rendered images, no labels or transcripts (label, transcript, bounding-box and device-serial columns were dropped; files holding transcripts were left out). No model weights.

## Anonymisation
Author names, user names, e-mail addresses, host and node names, site names, IP addresses, scheduler account names, absolute host paths and the identifiers of software agents named in the records were replaced by placeholders in file contents and file names: `<author>`, `<user>`, `<email>`, `<node>`, `<site>`, `<ip>`, `<account>`, `<agent>`, `clusterA` / `CLUSTER-A`, `clusterB` / `CLUSTER-B`, `$U`, `$U_B`, `$SCRATCH`, `$SCRATCH_ROOT`, `$SCRATCH_B`, `$HOME`, `$LOCAL`, `$VPS_ROOT`. Consequences: scripts contain placeholders instead of real paths and do not run unchanged; sha256 values recorded inside older provenance files refer to the unscrubbed originals; `<site>` also replaces a site abbreviation where it is used as a short dataset name; `<author>` also replaces the names of the manuscript authors where they appear as authors of cited work, and first names shared with them (<author>, <author>, <author>, <author>, <author>), in the related-work search records. Scheduler partition names were not replaced. The evaluator release tarball is included unpacked, because its change log names a site twice; its original sha256 is given above.

Replacements made in file contents (count of replaced strings):
| class | replacements |
|---|---|
| abs_path_clusterA | 19087 |
| abs_path_clusterB | 140 |
| abs_path_home | 1075 |
| abs_path_local | 44 |
| abs_path_vps | 874 |
| account_or_group | 16 |
| agent_id | 2466 |
| author_name | 346 |
| host_or_site | 31181 |
| ip_address | 103 |
| username | 17 |

UNIPEN/OnHW replacements: 231,036 unit IDs, 13,504 source IDs, 59 writer codes, 7 path prefixes; 87 source paths masked.

A scan of every file (text, gzip content, string arrays of npz files, file names) for user names, author names, site and node names, IP addresses, agent identifiers, e-mail addresses, absolute paths and original UNIPEN/OnHW writer codes and source paths found 0 matches for the manuscript authors, users, sites, hosts, paths and original UNIPEN/OnHW identifiers after scrubbing (2026-11-15, rebuild). The scan reports 1 pattern hit that is not such an identifier: the first name of a third-party author (`Esma Fatıma Bilgin Taşdemir`) in `supplementary/prior_art_records_v10.csv`.
