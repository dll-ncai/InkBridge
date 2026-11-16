# DATA CARD: d2_v2_iam (X3, D2 v2 component units)

- **Content:** IAM-OnDB t1 line strokes -> connected-component units. Language/script: English (Latin script). Source unit: line.
- **Built:** 2026-10-19T16:36:33Z on CLUSTER-A (<node>, Slurm 24480488), status: full. Provenance: PROVENANCE_X3_iam.json.
- **Licence of the source:** FKI/IAM terms: free for non-commercial research; registration with the FKI group requested (not yet done); no redistribution of the data or derived copies. Redistributable: False. Cite: M. Liwicki, H. Bunke. IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard. ICDAR 2005, pp. 956-961.
- **Derived-data rule:** CLUSTER-A only; not an artifact and not redistributed (source licence).

## Unit rule
OUHD-L v1 connected-component rule, code unchanged (`uhwr_baselines.ouhdl.make_units`, E0 code tree 1a5ee72e): strokes drawn as
2 px masks on the line canvas (height 64, ink height 56 px) are merged when they touch (8-connectivity); components with both bbox
sides < 0.18 x 56 px are marks and are attached to the main body whose x-extent contains their x-centre (else nearest ink).
Unit strokes keep recording order. Only change: units are numbered left to right by unit ink min x (`rank_ltr`).
Known property of this rule on Latin script: small separate letters (< ~10 px on the 56 px line ink height) count as marks and are
attached to a neighbouring body; `n_cc_raw` (components before attachment, in sources_*.csv) and `n_diacritic_strokes` record this.

## Splits
Official IAM-OnDB-t1 splits from task1.tar.gz: train = trainset.txt, val1 = testset_v.txt, val2 = testset_t.txt, test = testset_f.txt (form ids; every line XML of a listed form). Writer ids from original-xml-part via iam_form_writer_M4.csv (lines of forms without a writer id have writer_int -1).

| split | sources | writers | units | units with marks | source errors | seq len median / max | units with step > 1 px |
|---|---|---|---|---|---|---|---|
| test | 3859 | 63 | 54831 | 9631 | 0 | 147.0 / 320.0 | 81 |
| train | 5364 | 92 | 81659 | 15019 | 0 | 141.0 / 320.0 | 54 |
| val1 | 1438 | 20 | 18624 | 3239 | 0 | 137.0 / 320.0 | 8 |
| val2 | 1518 | 20 | 20327 | 4075 | 0 | 150.0 / 320.0 | 74 |

## Files
- `units_<split>.csv`: one row per unit (unit_id, source_id, split, writer_key, writer_int, source_label, rank_ltr, stroke_idx, n_strokes,
  n_diacritic_strokes, n_points_raw, n_seq, step, line_bbox). Unit text is not aligned; `source_label` is the line/word label.
- `units_<split>.npz`: raw points per unit (columns x, y, t, p in source units, y down, NaN where the source lacks the channel), raw_stroke_id, raw_offsets.
- `e0_<split>.npz`: E0/Nagamatsu-format arrays (cond_rendered 64x64, seq [x,y,pen,eos] <= 320 points, seq_offsets, step, gt_points/gt_stroke_id/gt_offsets on the 64x64 evaluation canvas).
- `sources_<split>.csv`: one row per source line/word (n_strokes, n_points, n_cc_raw, n_units, error, src_sha256).
- `build_stats.json`: counts and timing.


## Licence

LICENSE: IAM-OnDB terms (FKI, University of Bern): non-commercial research use; registration with the FKI group required (registration e-mail drafted 22 Oct 2026, not sent); no redistribution of the data or derived copies. CLUSTER-A only; release derivation code and split/ID lists only.

(Licence line added 2026-10-22 by the uhwr licence pass; register: shared/licences/licence_register_20261022.csv; dataset: iam.)
