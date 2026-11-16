# Unit-ID lists

Folder `lists/` holds one TSV per set (columns: unit_ids, sample_ids, writer_ids and, where stored, set_ids or scripts). For CASIA, IAM, UNIPEN, HPL Tamil and OnHW the files hold identifiers only: no ink, no rendered images, no labels or transcripts. Folder `source_records/` holds the data cards, build records and router manifests the lists come from, at their cluster-relative paths (`$U` = project root on the cluster, `$LOCAL` = local project folder).

| list | content | sampling rule | seed | record of the rule |
|---|---|---|---|---|
| ouhdl_test_4406 | OUHD-L test, all units of the official test split under division v3.0 | no sampling | n/a | `4_ouhdl_division_v3/README.md` |
| ouhdl_train_33615 / ouhdl_val_4349 | OUHD-L train and val, all units under division v3.0 | no sampling | n/a | `4_ouhdl_division_v3/README.md` |
| ouhdl_train_p01_336, _p10_3362, _p30_10085 | Urdu 1 %, 10 %, 30 % training subsets | nested prefixes of one permutation, numpy default_rng([20261016, 9]).permutation(N); k = floor(frac*N + 0.5); N = 33,615 | [20261016, 9] | `3_unit_ids/source_records/data/m5_multiscript_v1/ft_subsets_M5.json` |
| ouhdl_val1000 | Urdu validation units used for checkpoint selection | rows of the val split, numpy default_rng(0).choice(4349, 1000, replace=False), sorted (rule as stated in the paper appendix; list taken from a scored val1000 file and found equal to the rule) | 0 | `6_table_sources/exp/PENNET/results/val1000_ms10_rows.csv` |
| ouhdl_gold_354 | units of 21 hand-corrected test lines (own unit division, not a subset of the 4,406) | hand-corrected lines | n/a | `4_ouhdl_division_v3/README.md` |
| casia_zh_test_500, casia_en_test_500, hpl_tamil_test_500 | CASIA OLHWDB1.1 test (zh, en) and HPL IWFHR06 test (Tamil), 500 units each | up to 500 test units per set; numpy default_rng(2026), a fresh generator per set, choice(n_eligible, size, replace=False), units kept in loader order | 2026 | `3_unit_ids/source_records/local/D2/METHODS_D2.md (line 6); 3_unit_ids/source_records/exp/X4a/data/DATA_CARD.md` |
| iam500_test_500 | IAM-OnDB testset_f units with a known writer | np.sort(numpy default_rng(2026).choice(okw, min(500, len(okw)), replace=False)), okw = indices of test units with a known writer | 2026 | `1_evaluator_and_table_code/exp/M5/code_tarballs/m5_code_e33511dd.tar.gz__unpacked/m5_build.py (lines 332-335)` |
| unipen_de_test_827 | UNIPEN German, all units of the writer-disjoint test split | per contributor, writers in seeded random order; a writer goes to test if the test strokes of the contributor stay <= 0.30 of its strokes; stop once >= 0.20 | 20261016 | `3_unit_ids/source_records/data/d2_v2/unipen_de/split_writers.json` |
| onhw_all_5925 | OnHW words, all units (single split, 2 writers) | no sampling | n/a | `3_unit_ids/source_records/data/d2_v2/onhw/DATA_CARD.md` |
| selectivity_units_7x500 | 500 units per set for the metric-selectivity table | sets with more than 500 units: numpy default_rng(20261016).choice(n, 500, replace=False), sorted; else all units | 20261016 | `1_evaluator_and_table_code/exp/RQF/M1_selectivity/code/build_m1_units.py` |
| val250_<script> | 250 validation units per training script | n_val units drawn uniformly without replacement from the validation-writer pool | 20261016 | `3_unit_ids/source_records/data/m5_multiscript_v1/DATA_CARD.md` |
| members_<script>_train.csv (source_records) | 20,000 training units per script | uniformly at random without replacement from the pool of non-validation writers | 20261016 | `3_unit_ids/source_records/data/m5_multiscript_v1/DATA_CARD.md` |
| manifest_*.csv (source_records/exp/D3_ROUTE/router) | router output per unit for the seven test sets and five validation sets (IDs, router probabilities, route) | no sampling; same unit lists as above | n/a | `3_unit_ids/source_records/exp/D3_ROUTE/router/PROVENANCE_D3_ROUTE_router.json` |

## Checks run during the archive build (CHECKS_unit_ids.json)
- ouhdl_test_list_equals_division_v3_test_index: True
- ouhdl_val_list_equals_division_v3_val_index: True
- ouhdl_train_list_equals_division_v3_train_index: True
- p01_subset_of_p10: True
- p10_subset_of_p30: True
- p30_subset_of_train: True
- ouhdl_train_p01_336_equals_first_k_of_permutation_set: True
- ouhdl_train_p01_336_n_writers: 183
- ouhdl_train_p01_336_n_lines: 305
- ouhdl_train_p10_3362_equals_first_k_of_permutation_set: True
- ouhdl_train_p10_3362_n_writers: 247
- ouhdl_train_p10_3362_n_lines: 1569
- ouhdl_train_p30_10085_equals_first_k_of_permutation_set: True
- ouhdl_train_p30_10085_n_writers: 248
- ouhdl_train_p30_10085_n_lines: 1903
- gold_units_in_test_list: 0
- val1000_n: 1000
- val1000_equals_rule_default_rng0_choice: True
- ouhdl_test_4406_equals_router_manifest: True
- ouhdl_test_4406_equals_scored_units: True
- ouhdl_test_4406_n_writers: 32
- casia_zh_test_500_equals_router_manifest: True
- casia_zh_test_500_equals_scored_units: True
- casia_zh_test_500_n_writers: 60
- casia_en_test_500_equals_router_manifest: True
- casia_en_test_500_equals_scored_units: True
- casia_en_test_500_n_writers: 60
- hpl_tamil_test_500_equals_router_manifest: True
- hpl_tamil_test_500_equals_scored_units: True
- hpl_tamil_test_500_n_writers: 52
- iam500_test_500_equals_router_manifest: True
- iam500_test_500_equals_scored_units: True
- iam500_test_500_n_writers: 63
- unipen_de_test_827_equals_router_manifest: True
- unipen_de_test_827_equals_scored_units: True
- unipen_de_test_827_n_writers: 7
- onhw_all_5925_equals_router_manifest: True
- onhw_all_5925_equals_scored_units: True
- onhw_all_5925_n_writers: 2
- selectivity_units_per_set: {'casia_en': 500, 'casia_zh': 500, 'iam500': 500, 'onhw': 500, 'ouhdl': 500, 'tamil': 500, 'unipen_de': 500}
- selectivity_ouhdl_equals_rule: True
- selectivity_unipen_equals_rule: True

## Pseudonymised IDs
UNIPEN and OnHW writer codes, source IDs and unit IDs in the lists, the router manifests and the build records are pseudonyms (DE-A##, DE-S##, DE-O##); writer indices were renumbered with them. The mapping to the original IDs is not released. Rechecked after the replacement: unipen_de_test_827_equals_router_manifest: True; unipen_de_test_827_equals_scored_units_tables_5_6_8_10_11_12: True; unipen_de_test_827_equals_scored_units_tables_13_and_reference_systems: True; unipen_de_test_827_n_writers: 7; unipen_de_test_827_n_units: 827; unipen_de_test_827_all_pseudonymised: True; onhw_all_5925_equals_router_manifest: True; onhw_all_5925_equals_scored_units_tables_5_6_8_10_11_12: True; onhw_all_5925_equals_scored_units_tables_13_and_reference_systems: True; onhw_all_5925_n_writers: 2; onhw_all_5925_n_units: 5925; onhw_all_5925_all_pseudonymised: True; selectivity_unipen_equals_rule: True; selectivity_onhw_equals_rule: True; unipen_units_csv_equals_list: True.

## Not in this folder
- Router 1 %-label ablation: see `supplementary/SPEC_router336.md` and `supplementary/router336_per_unit.csv.gz`.
- Full unit lists of the IAM-OnDB test and training splits (8.4 MB and 12.7 MB CSV): left out for size; the 500 sampled test IDs and the 20,000 sampled training IDs are included.
- The index of the 500-unit CASIA and Tamil selection in its original folder (D2 pilot): not fetched; the same IDs are in the lists above and the rule is quoted from METHODS_D2.md.
