# DATA CARD: d2_v2_unipen_de (X3, D2 v2 component units)

- **Content:** UNIPEN train_r01_v07 German (sie word + char, atu) -> connected-component units. Language/script: German (Latin script). Source unit: word / char / short phrase.
- **Built:** 2026-10-19T16:54:56Z on CLUSTER-A (<node>, Slurm 24480820), status: full. Provenance: PROVENANCE_X3_unipen_de.json.
- **Licence of the source:** Zenodo record 10.5281/zenodo.1195803: CC BY 4.0; shipped iUF conditions of use: non-commercial research use, no commercial use of raw coordinates/text, publications must name the release and describe subset selection, do not name writers. Redistributable: False. Cite: I. Guyon, L. Schomaker, R. Plamondon, M. Liberman, S. Janet. UNIPEN project of on-line data exchange and recognizer benchmarks. ICPR 1994, pp. 29-33.
- **Derived-data rule:** CLUSTER-A only; not an artifact and not redistributed (source licence).

## Unit rule
OUHD-L v1 connected-component rule, code unchanged (`uhwr_baselines.ouhdl.make_units`, E0 code tree 1a5ee72e): strokes drawn as
2 px masks on the line canvas (height 64, ink height 56 px) are merged when they touch (8-connectivity); components with both bbox
sides < 0.18 x 56 px are marks and are attached to the main body whose x-extent contains their x-centre (else nearest ink).
Unit strokes keep recording order. Only change: units are numbered left to right by unit ink min x (`rank_ltr`).
Known property of this rule on Latin script: small separate letters (< ~10 px on the 56 px line ink height) count as marks and are
attached to a neighbouring body; `n_cc_raw` (components before attachment, in sources_*.csv) and `n_diacritic_strokes` record this.

## Splits
Canonical German split = M4 rule (m4_data.unipen_de_split, seed 20261016): writer-disjoint train / test per contributor; see split_writers.json. No validation split (German is a zero-shot evaluation set).

| split | sources | writers | units | units with marks | source errors | seq len median / max | units with step > 1 px |
|---|---|---|---|---|---|---|---|
| test | 475 | 7 | 827 | 42 | 0 | 178.0 / 292.0 | 0 |
| train | 1587 | 12 | 3927 | 308 | 0 | 168.0 / 320.0 | 38 |

## Files
- `units_<split>.csv`: one row per unit (unit_id, source_id, split, writer_key, writer_int, source_label, rank_ltr, stroke_idx, n_strokes,
  n_diacritic_strokes, n_points_raw, n_seq, step, line_bbox). Unit text is not aligned; `source_label` is the line/word label.
- `units_<split>.npz`: raw points per unit (columns x, y, t, p in source units, y down, NaN where the source lacks the channel), raw_stroke_id, raw_offsets.
- `e0_<split>.npz`: E0/Nagamatsu-format arrays (cond_rendered 64x64, seq [x,y,pen,eos] <= 320 points, seq_offsets, step, gt_points/gt_stroke_id/gt_offsets on the 64x64 evaluation canvas).
- `sources_<split>.csv`: one row per source line/word (n_strokes, n_points, n_cc_raw, n_units, error, src_sha256).
- `build_stats.json`: counts and timing.
- `split_writers.json`: test/train writer lists and integer writer codes.
- UNIPEN-de is stored y-up; y is flipped to y-down here.


## Licence

LICENSE: UNIPEN Train-R01/V07 (Zenodo 10.5281/zenodo.1195803): CC BY 4.0, plus the shipped iUF conditions of use (non-commercial research). Derived copies stay on CLUSTER-A; released German files use pseudonymous writer codes (shared/licences/german_writer_pseudonyms_PRIVATE.csv holds the private map).

(Licence line added 2026-10-22 by the uhwr licence pass; register: shared/licences/licence_register_20261022.csv; dataset: unipen_de.)
