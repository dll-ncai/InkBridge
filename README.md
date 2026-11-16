# InkBridge

Code, evaluation records and data identifiers for the paper

> **InkBridge: Low-Resource Adaptation and Endpoint Consensus for Handwriting Trajectory Recovery**
> Ali Hussain, Eshal Fatima, Anna Scius-Bertrand, Momina Moetesum, Adnan Ul-Hasan, Faisal Shafait.
> Manuscript submitted to the International Journal on Document Analysis and Recognition (IJDAR).

The paper is complete without this repository. The repository holds the material needed to check and reproduce its numbers: the evaluator, the scoring and table scripts, the run configurations, the unit-ID lists of every split and subset, the OUHD-L ligature-unit boundaries with the segmentation code, per-unit scores of the main comparisons, the protocol records, the checks and figure data of the submitted version, and the source of Online Resource 1 (the supplementary information of the paper). Model checkpoints are released through the links in [Model checkpoints](#model-checkpoints).

## Layout

| folder | content |
|---|---|
| `1_evaluator_and_table_code/` | `uhwr_eval-0.3.1/` (evaluator package, metric registry, release records); scoring entry `exp/D3_ROUTE/routeA/code/lane_score.py`; bootstrap and contrast scripts of every table under `exp/.../code/`; code tarballs unpacked into `<name>__unpacked/`; router classifier code under `local/experiments/RV11/router336/code/` |
| `2_run_configs/` | `run_config.json`, `job_history.txt`, `training_complete.json` of each run; arm tables and queues (`admin/`); recorded rules and amendments (`plans/`); experiment provenance files; job scripts |
| `3_unit_ids/` | `lists/` (one TSV per test set and training subset), `UNIT_ID_INDEX.md` (sampling rule and seed of each list), `source_records/` (data cards, build records, router manifests) |
| `4_ouhdl_division_v3/` | unit boundaries for all 42,370 OUHD-L units (CSV), ground-truth ink of the test and validation units (npz), build code, `SEGMENTATION_PROCEDURE.md` |
| `5_per_unit_scores/` | per-unit scores behind the result tables, `README_per_unit.md`, `load_rows.py`, `reproduce_warm_deltas.py` |
| `6_table_sources/` | table-level CSV/JSON/Markdown results at their cluster-relative paths |
| `supplementary/` | recipe, screen-roster, run-date and milestone-validation tables; facts pack and analysis tables; router 1 %-label ablation; related-work search records; see `supplementary/README.md` |
| `7_checks_v12/` | check script, job script and outputs for the submitted version: random-control table, ink-invariance test, plain fine-tuning row of the absolute-score table, facts file, input checksums |
| `8_figures_v12/` | `make_figs_v12.py`, the plotted values of the three data figures (`fig_*_data.csv`), captions, provenance |
| `10_online_resource_1/` | `ESM_1.tex` and `references.bib`: LaTeX source of the submitted supplementary information (Online Resource 1), unchanged, so it carries the author block of the paper; `build_esm_1.py`: derives `ESM_1.tex` from `supplementary_v12.1.tex`, the earlier supplement source, which is kept with its build script and edit log; `supplement_map_v12.1.csv`: map from paper items to supplement items; `ESM_1_meta_provenance.json` |
| `checkpoints/` | `checkpoint_manifest.csv` (every released checkpoint file with size, checksums, family and loading command), its provenance file, `verify_checkpoints.py` |
| `provenance/` | `archive_v11/`: README, `NOT_INCLUDED.md`, `MANIFEST.sha256`, `PROVENANCE.json` and build scripts (`ARCHIVE_BUILD/`) of the archive from which folders 1 to 6 and `supplementary/` were taken, unchanged; `PROVENANCE_repo_v12.json`: source, checksum and edits of every file added or changed for this repository |
| `MANIFEST.sha256`, `NOT_INCLUDED.md`, `LICENSE`, `CITATION.cff` | checksum of every file; items left out; licence; citation |

Path convention: below each numbered folder, files keep their path relative to the project root on the compute cluster (`$U`) or the local project folder (`$LOCAL`). Documents inside folders 1 to 6 and `supplementary/` were written when that material was first collected; where they name `ARCHIVE_BUILD/`, `PROVENANCE.json` or `MANIFEST.sha256`, read `provenance/archive_v11/ARCHIVE_BUILD/`, `provenance/archive_v11/PROVENANCE.json` and `provenance/archive_v11/MANIFEST.sha256`.

## What the repository adds, by paper section

| Paper section | What the repository adds | Where |
|---|---|---|
| Experimental Setup: Datasets and Target Data Budget | unit-ID list of every test set, training sample and target subset; sampling rule and seed of each list; data cards; router manifests | `3_unit_ids/lists/`, `3_unit_ids/UNIT_ID_INDEX.md`, `3_unit_ids/source_records/` |
| Experimental Setup: Datasets and Target Data Budget | OUHD-L unit boundaries, ground-truth ink of the test and validation units, segmentation code and procedure | `4_ouhdl_division_v3/` |
| Experimental Setup: Evaluation Metrics | evaluator package `uhwr_eval` 0.3.1 and the scoring entry script | `1_evaluator_and_table_code/uhwr_eval-0.3.1/`, `1_evaluator_and_table_code/exp/D3_ROUTE/routeA/code/lane_score.py` |
| Experimental Setup: Evaluation Metrics | ink-invariance test: AIoU of ground-truth ink under reversed point order, reversed stroke order and shuffled stroke order (summary and per-unit table) | `7_checks_v12/checks_v12_inkinv.csv`, `7_checks_v12/checks_v12_inkinv_units.csv` |
| Experimental Setup: Implementation Details | run configuration, job history and completion record of each run; recipe tables per run and run family; milestone-validation table | `2_run_configs/`, `supplementary/supp_recipes_runs_v11.csv`, `supplementary/supp_recipes_families_v11.csv`, `supplementary/supp_warm_milestone_validation_v11.csv` |
| Experimental Setup: Evaluation Protocol | recorded rules and amendments with SHA-256 checksum files; run-date table | `2_run_configs/exp/*/plans/`, `supplementary/supp_run_dates_v11.csv` |
| Results: Warm Adaptation on the Target | per-seed and pooled contrast table, absolute scores on common units, table scripts, per-unit scores | `6_table_sources/exp/RQF/tables/reviewer_followup_v1/T56_per_seed.csv`, `6_table_sources/exp/RQF/tables/paired_ext/absolute_common_units.csv`, `1_evaluator_and_table_code/exp/RQF/code/reviewer_v1/reviewer_followup.py`, `5_per_unit_scores/` |
| Results: Warm Adaptation on the Target | plain fine-tuning at 1% row of the absolute-score table (both devices) and the recomputed selection row | `7_checks_v12/checks_v12_ft1abs.csv` |
| Results: Retention on Six Non-Urdu Sets | contrast table for the six sets; per-writer direction on UNIPEN-de | `6_table_sources/exp/RQF/tables/RQF_R2_contrasts.csv`, `6_table_sources/exp/RQF/tables/RQF_R2_R2e_unipen_de_per_writer.csv` |
| Results: Exploratory Routed Adapter Results | contrast, verdict and provenance files of the routed-adapter variants | `6_table_sources/exp/RQF/tables/RQF_SM_contrasts.csv`, `6_table_sources/exp/RQF/tables/RQF_SM_solves.csv`, `6_table_sources/exp/RQF/tables/RQF_SM_provenance.json` |
| Results: Endpoint Consensus Selection | paired table on Urdu, Kendall tau for all cells, per-cell summary on the non-Urdu sets | `6_table_sources/exp/RQF/R3/split_v1/R3_split_paired.csv`, `6_table_sources/exp/RQF/tables/reviewer_followup_v1/R3_tau_all_cells.csv`, `6_table_sources/exp/RQF/R3/R3_summary_3seed_cells.csv` |
| Results: Endpoint Consensus Selection | random-control table: selected sample against the mean of its three candidates and against the random pick, per cell and metric, with both control counts | `7_checks_v12/checks_v12_randctl_cells.csv`, `7_checks_v12/checks_v12_results.csv` |
| Results: Negative Results | frame-choice paired summary; roster of the foundation-model screen | `6_table_sources/exp/RQF/R1d/read/R1d_summary_paired.csv`, `supplementary/supp_fm_screen_roster_v11.csv` |
| Introduction | direction values per system and the per-set tables of the mirrored left-to-right control (also used in the Discussion) | `6_table_sources/exp/RQF/R1/tables/` |
| Discussion | metric-selectivity table, perturbation code, per-unit scores | `6_table_sources/exp/RQF/M1_selectivity/out/m1_leak_wide_native.csv`, `1_evaluator_and_table_code/exp/RQF/M1_selectivity/code/leak_table.py`, `5_per_unit_scores/table_15_selectivity/` |
| Discussion | explanation of the 4,358 paired units (48 of 4,406 units without a defined direction in one frame) and the unit-level shares | `7_checks_v12/checks_v12_facts.json` (key `n4358`) |
| all v12 checks | check script and input hashes | `7_checks_v12/checks_v12.py`, `7_checks_v12/PROVENANCE_checks_v12.json` |

Folder names under `5_per_unit_scores/` and table numbers inside folders 1 to 6 follow an earlier draft of the paper. The crosswalk:

| Earlier number | Content | Submitted paper | Online Resource 1 (label in `10_online_resource_1/supplement_map_v12.1.csv`) |
|---|---|---|---|
| 3 | test sets | Table 2 | |
| 4 | cross-script failure | | `tabS:failure` |
| 5, 6 | warm variants against plain fine-tuning at 1 % and 10 % | Table 3 (1 %) | `tabS:warm` (with the 10 % columns) |
| 7 | absolute scores | Table 4 | |
| 8 | retention on six non-Urdu sets | Table 5 | |
| 9 | direction on the German sets | | `tabS:german` |
| 10 | routed adapter variants | Table 6 | `tabS:singleapp`, `tabS:singleR` |
| 11 | selection on OUHD-L | Table 7 | |
| 12 | selection on the six non-Urdu sets | | `tabS:r3b` |
| 13 | frame choice | | `tabS:r1d` |
| 15 | metric selectivity | | `tabS:m1` |
| 16 | settings | | `tabS:hyper` |
| 17 | corpora | | `tabS:corpora` |

The file-by-file list for each of these tables, with the check done when the files were collected, is in `provenance/archive_v11/README.md` (section "Paper table -> files").

## Install the evaluator and rerun a table from the per-unit scores

Python 3.10 or later.

```bash
python -m venv .venv && . .venv/bin/activate
pip install ./1_evaluator_and_table_code/uhwr_eval-0.3.1

# 1. evaluator self-test: toy values of DTW, LDTW and AIoU, mark-order and timing checks
cd 1_evaluator_and_table_code/uhwr_eval-0.3.1
python -c "from uhwr_eval import validate_toy as V; V.run('metric_sanity_check.csv')"
python tests_marks.py && python tests_timing.py
cd ../..

# 2. warm-adaptation table: pooled paired differences recomputed from the per-unit scores
python 5_per_unit_scores/reproduce_warm_deltas.py
```

Step 1 needs the dependencies of the evaluator (among them torch and lpips, which downloads the torchvision AlexNet weights on first use) and rewrites `toy_validation.csv` in the evaluator folder. Step 2 reads `5_per_unit_scores/tables_5_6_8_10_11_12/` with `load_rows.py`, recomputes the 24 pooled differences (two warm variants, 1 % and 10 %, six metrics) and compares them with `6_table_sources/exp/RQF/tables/reviewer_followup_v1/T56_per_seed.csv`; it exits with code 1 if a difference exceeds `--tol` (default 1e-12). Intervals are not recomputed by this script.

The other tables are rebuilt the same way: `load_rows.load(folder, set_name)` returns one row per scored file and unit (set, checkpoint tag, sampling seed, unit ID, writer ID, six metrics); the per-unit value of a system is the mean over its sampling seeds, a contrast is the paired per-unit difference on common units, and the interval is the 95 % percentile interval of 2,000 writer-cluster bootstrap resamples (units on OnHW, which has two writers). The original table scripts are under `1_evaluator_and_table_code/exp/.../code/`; they read the full scored rows files at cluster paths, which are replaced by placeholders here (see [Placeholders](#placeholders)), so they document the computation and do not run unchanged. `5_per_unit_scores/README_per_unit.md` lists the tag conventions and the scored files that are not included.

Scoring new predictions: `uhwr_eval.score(gt, pred, canvas, protocol)` (see `1_evaluator_and_table_code/uhwr_eval-0.3.1/README.md`); the entry script used for all runs is `1_evaluator_and_table_code/exp/D3_ROUTE/routeA/code/lane_score.py`.

## Model checkpoints

Checkpoints are released on Google Drive, not in git. `checkpoints/checkpoint_manifest.csv` lists every released file: 137 files (121 run checkpoints and 16 script-router classifiers), 36.47 GB, with its size and checksums; `checkpoints/checkpoint_manifest_meta_provenance.json` records how the manifest was made. 7 runs of the run roster (`supplementary/supp_recipes_runs_v11.csv`) are not among them: they have no selection record and no scored result file names them. 11 of the released files were trained on a second host; their Drive copies were checked by size, the other 126 by size and MD5 (column `drive_check`).

**Download folder:** https://drive.google.com/drive/folders/1NXiF5ca7EpbRkULIjmPaNW5oFfTZKYMw (Google Drive folder `InkBridge/InkBridge_checkpoints`; the files are in `checkpoints/<family>/<name>.pt` below it, the same layout as `public_file` in the manifest).

Each row of the table below gives a checkpoint group, its notation in the paper and run-name pattern, the number of files, their total size and the Drive folder. The SHA-256 checksum and the direct link of every file are in the manifest (columns `sha256`, `drive_url`).

| Checkpoint group (`family`) | Paper notation; run-name pattern | Files | GB | Link |
|---|---|---|---|---|
| `plain_FT_1pct` | `plain_FT_1pct (FT1%; comp_p01_identity_*)` | 3 | 1.30 | [folder](https://drive.google.com/drive/folders/1nrFNTBpDwJmtUCj-0u35hwN-NmdqdybF) |
| `warm_backbone` | `warm_backbone (NOFM_W, PEINIT_W)` | 2 | 0.78 | [folder](https://drive.google.com/drive/folders/1uN0pHGZyGi9oY6glWUat55EQE8NS7JJd) |
| `FMguided_warm_backbone` | `FMguided_warm_backbone (<fm>_W)` | 10 | 3.89 | [folder](https://drive.google.com/drive/folders/1rNR72M2_Rhi-2wYw2QztzfeEbWwhSxYr) |
| `warm_FT` | `warm_FT (WFT, WFT^PE; NOFM_W/PEINIT_W FT)` | 14 | 5.46 | [folder](https://drive.google.com/drive/folders/1gTB1HsgXVTTtefUrMcWIXZxOMHjWwhMP) |
| `FMguided_warm_FT` | `FMguided_warm_FT (WFT^FM)` | 10 | 3.89 | [folder](https://drive.google.com/drive/folders/1Z2ouTokk5ekUYicO_38PUpqQTHILfhPF) |
| `plain_FT_10pct_30pct` | `plain_FT_10pct_30pct (FT; LUOFT_p10_*, M5 ft/p*)` | 7 | 3.04 | [folder](https://drive.google.com/drive/folders/1xJ19nHdPgW9-6MpV4v0o4VXp8og1cmNy) |
| `joint_model` | `joint_model (f_all; M5 joint)` | 1 | 0.43 | [folder](https://drive.google.com/drive/folders/18l_OxMW8SkAPCUoa7iiHI_eG2vcPq6yi) |
| `backbone_leave_Urdu_out` | `backbone_leave_Urdu_out (f_not_ur; M5 luo)` | 1 | 0.43 | [folder](https://drive.google.com/drive/folders/1VDxXnEVVmbMcSy1rXlgFO1bPBER5Vj6b) |
| `stepmatched_backbone` | `stepmatched_backbone (LUOC40_bb)` | 1 | 0.43 | [folder](https://drive.google.com/drive/folders/1xMr3tgX3KM2E7Z7h25XgqXSVXdzskYQl) |
| `stepmatched_FT` | `stepmatched_FT (SFT; LUOC40_p*)` | 5 | 2.17 | [folder](https://drive.google.com/drive/folders/1YdiIIRzvdbWlhExDSt3zn7sQmw6iY-Na) |
| `routed_adapter` | `routed_adapter (AD; SMA_*)` | 32 | 0.66 | [folder](https://drive.google.com/drive/folders/1dy2kX9wf9FvCiQwbqxyJp6AB9zkoGdAO) |
| `replay_single_model` | `replay_single_model (RP; SMR_*)` | 31 | 12.21 | [folder](https://drive.google.com/drive/folders/1rrRFY9CKgaUevtAkl4gW-gUDq4x0ErTT) |
| `script_router` | `script-router classifiers` | 16 | 0.02 | [folder](https://drive.google.com/drive/folders/1XuXKjeWy7gWgq97x7rXYJfa90CEKdfwE) |
| `stepmatched_FT_VPS_copy` | `stepmatched_FT_VPS_copy (LUOC40VPS_*; exploratory)` | 3 | 1.30 | [folder](https://drive.google.com/drive/folders/12FvDJdvsEriDVIk8Df956J_yhJU3fij5) |
| `stepmatched_backbone_VPS_copy` | `stepmatched_backbone_VPS_copy (LUOC40_bb_vps; exploratory)` | 1 | 0.43 | [folder](https://drive.google.com/drive/folders/1LxjOKSnlel3PuhUSfKaBBViFnl6FsqMo) |

### Families

| `family` | files | GB | `file_kind` |
|---|---|---|---|
| `plain_FT_1pct` | 3 | 1.30 | `full_checkpoint` |
| `warm_backbone` | 2 | 0.78 | `stripped_backbone_fmdx_bb` |
| `FMguided_warm_backbone` | 10 | 3.89 | `stripped_backbone_fmdx_bb` |
| `warm_FT` | 14 | 5.46 | `stripped_backbone_fmdx_bb` |
| `FMguided_warm_FT` | 10 | 3.89 | `stripped_backbone_fmdx_bb` |
| `plain_FT_10pct_30pct` | 7 | 3.04 | `full_checkpoint` |
| `joint_model` | 1 | 0.43 | `full_checkpoint` |
| `backbone_leave_Urdu_out` | 1 | 0.43 | `full_checkpoint` |
| `stepmatched_backbone` | 1 | 0.43 | `full_checkpoint` |
| `stepmatched_FT` | 5 | 2.17 | `full_checkpoint` |
| `routed_adapter` | 32 | 0.66 | `adapter_checkpoint` |
| `replay_single_model` | 31 | 12.21 | `stripped_backbone_fmdx_bb` |
| `script_router` | 16 | 0.02 | `script_router_classifier` |
| `stepmatched_FT_VPS_copy` | 3 | 1.30 | `full_checkpoint` |
| `stepmatched_backbone_VPS_copy` | 1 | 0.43 | `full_checkpoint` |

### Manifest columns

| column | meaning |
|---|---|
| `public_file` | path of the file in the released folder: `checkpoints/<family>/<name>.pt` |
| `family`, `family_label` | checkpoint family; its notation in the paper and run-name pattern |
| `run` | run name as in the run roster |
| `selected_milestone` | milestone of the run that the file holds |
| `selection_record`, `selection_record_sha256`, `selection_record_mtime_utc` | file inside the run folder that records the milestone selection, its checksum and date (`not applicable` for the router classifiers) |
| `selection_record_argmin_ldtw_milestone` | milestone with the lowest LDTW in that record |
| `roster_selected_milestone_txt`, `roster_selected_milestone_selection_json`, `roster_matches_record` | milestone given in the two roster columns, and whether they agree with the record |
| `file_kind` | `full_checkpoint`, `stripped_backbone_fmdx_bb`, `adapter_checkpoint` or `script_router_classifier`; decides the loader (see below) |
| `bytes`, `sha256`, `md5` | size and checksums of the released file |
| `hash_cross_check` | whether the SHA-256 equals an earlier hash record of the same file |
| `encoder`, `fm_teacher`, `init_checkpoint` | copied from the run roster (`not available` = not recorded in the run configuration); `init_checkpoint` is a cluster path with the placeholder `$U` |
| `paper_floats_v12_2`, `paper_floats_basis` | tables and figures of the submitted paper that use the symbol of the family; a family-level statement, not a mapping of single runs to table cells |
| `loading_command`, `loading_code_source` | call or command that loads the file (required arguments as placeholders) and the code file it comes from |
| `source_file_mtime_utc` | modification time of the original file |
| `selection_evidence` | what shows that the file is the selected checkpoint of the run |
| `drive_url`, `drive_check` | direct Google Drive link of the file; what was compared between the Drive copy and the manifest after upload |

### Verify a download

Keep the folder structure of `public_file` (for example `checkpoints/script_router/router_reported.pt` below the repository root). Then either compare one file by hand,

```bash
shasum -a 256 checkpoints/script_router/router_reported.pt    # Linux: sha256sum
grep router_reported.pt checkpoints/checkpoint_manifest.csv    # column sha256
```

or check every downloaded file (size and SHA-256) against the manifest:

```bash
python checkpoints/verify_checkpoints.py                 # files present below the repository root
python checkpoints/verify_checkpoints.py --root /path/to/download --require-all
```

The script uses the standard library only and exits with code 1 if a file differs (with `--require-all` also if a file is missing).

### Which loader reads which family

| `file_kind` | files | families | loading code in this repository | `loading_command` of the manifest |
|---|---|---|---|---|
| `full_checkpoint` | 17 | `plain_FT_1pct`, `plain_FT_10pct_30pct`, `joint_model`, `backbone_leave_Urdu_out`, `stepmatched_backbone`, `stepmatched_FT` | `1_evaluator_and_table_code/exp/RQF/admin/d2moe_code_9a020b75.tgz__unpacked/d2moe_code_9a020b75/uhwr_route_common.py`; `1_evaluator_and_table_code/exp/RQF/R1d/code/_src_snapshot.tgz__unpacked/CAND/code/frame_code_a44e78b0/frame_sample.py` | `sd, info = uhwr_route_common.load_checkpoint_weights("<file>", which="ema")` ; `python frame_sample.py --label <label> --run-dir <run_dir> --milestone <ms> --checkpoint <file> --stage <stage> --out-dir <out> --sets <sets>` |
| `stripped_backbone_fmdx_bb` | 64 | `warm_backbone`, `FMguided_warm_backbone`, `warm_FT`, `FMguided_warm_FT`, `replay_single_model` | `1_evaluator_and_table_code/exp/FMD/xs/code/fmdx_backbone_v1/bb_model.py` | `unet = bb_model.load_backbone_for_inference("<file>", U, dev="cpu", which="ema")` |
| `adapter_checkpoint` | 29 | `routed_adapter` | `1_evaluator_and_table_code/exp/RQF/code/sm_v1/sma_sample.py` | `python sma_sample.py --kind <kind> --ckpt <file> --backbone <backbone checkpoint> --gt-npz <npz> --set-name <set> --route <all\|router manifest csv> --out <out>` |
| `script_router_classifier` | 16 | `script_router` | `1_evaluator_and_table_code/local/experiments/RV11/router336/code/script_clf.py`; `1_evaluator_and_table_code/local/experiments/RV11/router336/code/rr336_eval.py` | `m = ScriptCNN(); ck = torch.load('<file>', map_location='cpu', weights_only=False); m.load_state_dict(ck['state_dict']); m.eval()` |

The manifest names the router loading code by its path in the local project folder (`experiments/RV11/router336/code/`); in this repository that folder is `1_evaluator_and_table_code/local/experiments/RV11/router336/code/` (`script_clf.py` defines `ScriptCNN`; `rr336_eval.py`, line 17, holds the loading call; `rr336_train.py` and `run_all.sh` are the training script and its driver). Example for a router classifier:

```python
import sys, torch
sys.path.insert(0, "1_evaluator_and_table_code/local/experiments/RV11/router336/code")
from script_clf import ScriptCNN
m = ScriptCNN()
ck = torch.load("checkpoints/script_router/router_reported.pt", map_location="cpu", weights_only=False)
m.load_state_dict(ck["state_dict"]); m.eval()
```

`script_clf.py` runs unchanged. The other loading files contain path placeholders (see [Placeholders](#placeholders)); import the named function or pass your own paths on the command line.

## Data access

- **OUHD-L** (Urdu): Zenodo, doi:[10.5281/zenodo.20642162](https://doi.org/10.5281/zenodo.20642162). This repository holds the ligature-unit boundaries of all units, the ground-truth ink of the test and validation units, and the code that rebuilds the training units from the Zenodo release (`4_ouhdl_division_v3/`).
- **CASIA-OLHWDB, IAM-OnDB, HP Labs Tamil, UNIPEN and OnHW** are third-party datasets with their own licences and are not redistributed. For these sets the repository holds unit identifiers, writer indices and counts only: no ink, no rendered images, no labels or transcripts. Obtain the data from the original providers and select the units with the lists in `3_unit_ids/lists/`.

Items that are not in the repository are listed in `NOT_INCLUDED.md`.

## Pseudonymised identifiers (UNIPEN, OnHW)

The UNIPEN conditions of use ask that writers are not named. All UNIPEN and OnHW writer codes, source IDs and unit IDs in this repository are pseudonyms (`DE-A##`, `DE-S##`, `DE-O##`; unit IDs `<pseudonym>/<contributor>_<index>_u<k>`), and writer indices were renumbered with them. The same pseudonyms are used in the ID lists, router manifests, per-unit score tables, supplementary files and `7_checks_v12/checks_v12_inkinv_units.csv`, so the files join on unit ID. The mapping to the original IDs is not released. In build records of these two corpora, source paths without a unit-level pseudonym are masked as `<unipen-path>`.

## Placeholders

User names, e-mail addresses, host and node names, IP addresses, scheduler account names, absolute host paths and the identifiers of software agents named in the records were replaced by placeholders in file contents and file names: `<user>`, `<email>`, `<node>`, `<site>`, `<ip>`, `<account>`, `<agent>`, `clusterA` / `CLUSTER-A`, `clusterB` / `CLUSTER-B`, `$U`, `$U_B`, `$SCRATCH`, `$SCRATCH_ROOT`, `$SCRATCH_B`, `$HOME`, `$LOCAL`, `$VPS_ROOT`. In folders 1 to 6 and `supplementary/`, author names are also replaced by `<author>` and site names by `<site>`; replacement counts are in `provenance/archive_v11/README.md`. In folders 7, 8 and 10, `checkpoints/` and the router classifier code the paths, user names, node names and agent identifiers are replaced, and the name of the compute site is replaced by the word `cluster` (folders 1 to 6 and `supplementary/` use `<site>`, as stated above). Consequences: scripts contain placeholders instead of real paths and do not run unchanged; sha256 values recorded inside provenance files refer to the files before replacement. `provenance/PROVENANCE_repo_v12.json` gives, for each of these files, the checksum before and after replacement and the replacement counts.

## Licence and citation

The code and records of this repository are released under the MIT licence (`LICENSE`). `1_evaluator_and_table_code/uhwr_eval-0.3.1/uhwr_eval/_nagamatsu_metrics.py` is copied unchanged from the repository named in the evaluator README and remains under the terms of that repository. The third-party datasets keep their own licences.

Please cite the paper named above (`CITATION.cff`); the BibTeX entry will be added on publication.
