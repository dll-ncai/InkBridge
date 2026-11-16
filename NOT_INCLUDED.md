# Not included

## Items that are not in the repository
| item | reason |
|---|---|
| Model weights and checkpoints (backbones, warm backbones, fine-tunes, adapters, router classifier) | not stored in git; released on Google Drive: 78 files listed with checksums in `checkpoints/checkpoint_manifest.csv`, links in the section "Model checkpoints" of `README.md`; 18 runs of the run roster are not released (13 without a selection record, 5 whose weights were not on the compute cluster) |
| Checkpoints of runs that are not relevant to the paper (no number, table or figure of the paper or of Online Resource 1 comes from them, and no released file needs them) | not released |
| Ground-truth ink and rendered images of CASIA, IAM-OnDB, UNIPEN, HPL Tamil and OnHW units | third-party datasets with their own licences: identifiers only for these corpora |
| Predicted trajectories (sample files) behind the scores | sizes not measured; per-unit scores are included instead |
| Ground-truth ink of the OUHD-L training units under division v3.0 (ligatures_v3_train.npz, 95 MB) | size; unit boundaries of all training units are included and the file is rebuilt from the Zenodo release with the included build code |
| Per-unit scores of arms not printed in the result tables: 1541 scored rows files (3.94 GB as CSV) | size; listed in provenance/archive_v11/ARCHIVE_BUILD/rows_listing_all.csv (in_archive = False) |
| Per-unit scores of the adapter, Urdu-only-from-scratch and InkSight rows of the absolute-score table | not fetched; the table-level file and the PEN-Net per-unit rows are included |
| Standardised-protocol rows and the 38 other columns of each scored rows file | size; the table scripts read the native-protocol rows and the six kept metrics |
| Original UNIPEN and OnHW writer codes, source IDs and unit IDs, and the mapping from pseudonyms to them | the UNIPEN conditions of use ask that writers are not named; the repository holds pseudonyms; the mapping is not released |
| Statement whether division v3.0 is part of the Zenodo record | not checked; the unit boundaries themselves are included here |
| Full unit lists of the IAM-OnDB test and training splits (units_test.csv 8.4 MB, units_train.csv 12.7 MB) | size cap of 3 MB per record file; the sampled IAM500 and training-sample IDs are included |
| exp/M5/report/m5_results_bundle.tar.gz (13.7 MB) | size cap of 3 MB per record file; the M5 report tables are included |
| Manuscript source and PDF | under submission |
| Figure files of the paper | `8_figures_v12/` holds the figure script and the plotted values |

## Source files skipped during collection
| source | reason |
|---|---|
| `$U/exp/FMD/xs/read_v1/docs_snap.tgz` | interim snapshot tarball of tables that are included in final form |
| `$U/exp/FMD/xs/read_v1/snap_20261105T2029Z.tgz` | interim snapshot tarball of tables that are included in final form |
| `$U/exp/FMD/xs/read_v1/snap_final_20261107T0047Z.tgz` | interim snapshot tarball of tables that are included in final form |
| `$U/exp/FMD/xs/read_v1/warm_tables_snap.tgz` | interim snapshot tarball of tables that are included in final form |
| `$U/data/d2_v2/iam/sources_test.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/iam/sources_train.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/iam/sources_val1.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/iam/sources_val2.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/onhw/sources_all.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/onhw/word_types_both_writers.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/unipen_de/sources_test.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$U/data/d2_v2/unipen_de/sources_train.csv` | holds third-party transcripts; sharing limit is IDs only |
| `$LOCAL/data/ouhdl_ligatures_v3/ligatures_v3_train.npz` | 95 MB ground-truth ink of the OUHD-L training units; left out for size; unit boundaries of all 33,615 training units are in ligatures_v3_train.csv and the file is rebuilt from the Zenodo release with build/build_ligature_dataset.py; its sha256 is in PROVENANCE.json |
| `$U/exp/FMD/xs/admin/hf_search_raw.txt` | raw model-hub search dump holding third-party account names; not a run record |
| `$U/exp/D3_ROUTE/router/script_clf.pt` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/RQF/code/rq3_split_v1/s.b64` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/screen/rows_diffusion.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/screen/rows_encoders.csv.tmp278801.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/screen/rows_encoders.csv.tmp365830.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/screen/rows_ink.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/read_v1/v2_fmdx_warm_forest.pdf` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/read_v1/v2_fmdx_warm_forest.png` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_casia_en_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_casia_zh_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_hpl_tamil_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_iam_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_ouhdl_val1000.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_enc_unipen_de_tr3927.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_casia_en_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_casia_zh_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_hpl_tamil_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_iam_val250.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_ouhdl_val1000.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/FMD/xs/screen/rows_xs_genvlm_unipen_de_tr3927.csv.lock` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/exp/M5/report/m5_results_bundle.tar.gz` | over the 3 MB per-file cap (13728441 bytes) |
| `$U/data/m5_multiscript_v1/m5_ft_ouhdl_p01.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_ft_ouhdl_p10.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_ft_ouhdl_p30.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_joint_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_luo_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_test_iam500.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_val_casia_en.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_val_casia_zh.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_val_hpl_tamil.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_val_iam.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/m5_multiscript_v1/m5_val_ouhdl.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/e0_test.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/e0_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/e0_val1.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/e0_val2.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/units_test.csv` | over the 3 MB per-file cap (8411047 bytes) |
| `$U/data/d2_v2/iam/units_test.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/units_train.csv` | over the 3 MB per-file cap (12684526 bytes) |
| `$U/data/d2_v2/iam/units_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/units_val1.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/iam/units_val2.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/onhw/e0_all.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/onhw/units_all.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/unipen_de/e0_test.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/unipen_de/e0_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/unipen_de/units_test.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |
| `$U/data/d2_v2/unipen_de/units_train.npz` | binary type not copied (arrays with ink or images, weights, figures, lock files) |

## Residuals of the placeholder replacement
- Scheduler partition names and GPU model names remain in job scripts and logs.
- Dataset-internal writer and contributor codes of the third-party corpora remain in the ID lists (they are needed for the writer-cluster bootstrap).
- `$HOME` also replaces a container-internal path (`$HOME`) in package-freeze files.
