# DATA CARD: d2_v2_onhw (X3, D2 v2 component units)

- **Content:** OnHW-wordsTraj raw recordings (2 persons x 29 recordings) -> words -> connected-component units. Language/script: German (Latin script). Source unit: word.
- **Built:** 2026-10-19T16:55:39Z on CLUSTER-A (<node>, Slurm 24480848), status: full. Provenance: PROVENANCE_X3_onhw.json.
- **Licence of the source:** Shared privately by F. Ott (Fraunhofer IIS) on 16 Oct 2026 for research use (Gmail thread 1a13ba55dd19248b); the public OnHW-wordsTraj release is CC BY-NC 4.0; redistribution of the raw recordings not granted; participant identifiers removed here. Redistributable: False. Cite: F. Ott et al., OnHW datasets (Fraunhofer IIS); cite the OnHW-wordsTraj publication.
- **Derived-data rule:** CLUSTER-A only; not an artifact and not redistributed (source licence).

## Unit rule
OUHD-L v1 connected-component rule, code unchanged (`uhwr_baselines.ouhdl.make_units`, E0 code tree 1a5ee72e): strokes drawn as
2 px masks on the line canvas (height 64, ink height 56 px) are merged when they touch (8-connectivity); components with both bbox
sides < 0.18 x 56 px are marks and are attached to the main body whose x-extent contains their x-centre (else nearest ink).
Unit strokes keep recording order. Only change: units are numbered left to right by unit ink min x (`rank_ltr`).
Known property of this rule on Latin script: small separate letters (< ~10 px on the 56 px line ink height) count as marks and are
attached to a neighbouring body; `n_cc_raw` (components before attachment, in sources_*.csv) and `n_diacritic_strokes` record this.

## Splits
Single split 'all' (2 writers; evaluation only). word_types_both_writers.csv lists word types written by both.

| split | sources | writers | units | units with marks | source errors | seq len median / max | units with step > 1 px |
|---|---|---|---|---|---|---|---|
| all | 4675 | 2 | 5925 | 1951 | 2 | 234.0 / 320.0 | 237 |

## Files
- `units_<split>.csv`: one row per unit (unit_id, source_id, split, writer_key, writer_int, source_label, rank_ltr, stroke_idx, n_strokes,
  n_diacritic_strokes, n_points_raw, n_seq, step, line_bbox). Unit text is not aligned; `source_label` is the line/word label.
- `units_<split>.npz`: raw points per unit (columns x, y, t, p in source units, y down, NaN where the source lacks the channel), raw_stroke_id, raw_offsets.
- `e0_<split>.npz`: E0/Nagamatsu-format arrays (cond_rendered 64x64, seq [x,y,pen,eos] <= 320 points, seq_offsets, step, gt_points/gt_stroke_id/gt_offsets on the 64x64 evaluation canvas).
- `sources_<split>.csv`: one row per source line/word (n_strokes, n_points, n_cc_raw, n_units, error, src_sha256).
- `build_stats.json`: counts and timing.
- `recordings.csv`: per recording only non-personal meta fields (pen_nr, recording_type, hw/fw version, app name/version) and file sha256s.
- `word_types_both_writers.csv`: word types with samples from both persons (count per person).

## De-identification (OnHW)
meta.json person fields (person_name, person_year_of_birth, person_handedness, person_gender) are never copied. setup.jpg and the
IMU sensor_data.csv are not copied. Writers are `DE-O01` / `DE-O02` (folder names of the release).
Stroke rule: touch rows within [start, stop) of a word label; a stroke ends where a hover row falls between two touch rows or
the gap exceeds 100 ms. Label = `correction` column if non-empty, else `Label`. Kept channels: x, y, t (ms from word start), pressure.


## Licence

LICENSE: OnHW raw recordings shared privately by the dataset author (Fraunhofer IIS, 16 Oct 2026) for research use; no redistribution permission recorded. Public OnHW-wordsTraj release: CC BY-NC 4.0. CLUSTER-A only; release code only; meta.json participant fields are never included; released German files use pseudonymous writer codes.

(Licence line added 2026-10-22 by the uhwr licence pass; register: shared/licences/licence_register_20261022.csv; dataset: onhw.)
