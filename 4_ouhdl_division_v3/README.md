# OUHD-L ligature units, division v3.0

Derived from OUHD-L v1.0 (Zenodo 10.5281/zenodo.20642162); same writer-disjoint train / val / test split.
A unit is one ligature as written: a maximal run of joined letters (single isolated letters included,
e.g. دلائل → د | لا | ئل) together with its own dots and marks (dots, kaf/gaf bar, ط of ٹ ڈ ڑ, madda, hamza).
Units were found per line by aligning the pen-down strokes (in writing order) to the line's transcript
(ligseg_v3.py, per-letter lengths fitted on train only).

| split | lines | writers | units | text ligatures | single-stroke | with marks | spanning >1 text ligature | lines: units = text ligatures |
|---|---|---|---|---|---|---|---|---|
| train | 1911 | 248 | 33,615 | 35,320 | 38.2% | 56.5% | 1541 | 57.5% |
| val | 239 | 31 | 4,349 | 4,513 | 38.7% | 56.1% | 158 | 60.2% |
| test | 253 | 32 | 4,406 | 4,628 | 38.9% | 56.2% | 206 | 54.1% |

## Quality (test split, 21 lines hand-corrected by <author>, 354 units)
Exact-unit F1: v1 (connected components) 0.667, v2 (global rule) 0.742, **v3 0.834** (v3 − v2: +0.035 to +0.152, 95% CI).
Remaining v3 errors on those lines: part of a body split off (25), separate ligatures merged (19), own mark lost (5), foreign mark included (5).
These 21 test lines are a held-out gold set: do not tune later versions on them.

## Known limitations
- One continuous pen stroke that crosses a non-joining letter stays one unit (it cannot be split without cutting the stroke). Linguistically these are separate ligatures; open decision.
- Units use the transcript: where the ink differs from the transcript (e.g. 1000_0022_5, 5000_0031_4) the division is less reliable.

## Files
- `ligatures_v3_<split>.csv`: one row per unit (unit_id, sample_id, writer_id, writing_rank = order of the unit's first *body* stroke (a mark written before its body does not count; differs from first-stroke order on 212 lines), rtl_rank = right-to-left order by rightmost ink, stroke indices into the line's pen-down strokes, aligned_text, bbox in tablet units, ...).
- `ligatures_v3_<split>.npz`: points (x, y, time, pressure, local stroke number; raw tablet units, writing order) + unit_offsets + unit_ids.
- `lines_v3_<split>.csv`: per line counts (units, text ligatures, skipped ligatures, unaligned strokes).
- `ouhdl_ligatures.py`: `idx, units = load_split(root, 'train'); units('<unit_id>')` → list of (n, 4) strokes.
- `PROVENANCE.json`: input checksums, code hashes, parameters, command, environment, output checksums.

Rebuild (all code is in build/): `cd build && PYTHONPATH=.:. python build_ligature_dataset.py --root <ouhdl> --lengths letter_lengths.json --out <dir>`
(build/uhwr_baselines and build/uhwr_eval are the exact adapter and evaluator versions used; their hashes are in PROVENANCE.json).

## Independent review (16 Oct 2026)
Six reviewer agents checked every unit of 60 random lines (24 train, 12 val, 12 test, plus 12 of the hand-corrected gold lines, hidden among them).
Against the gold lines the reviewers had precision 0.95 and recall 0.93 (41 of 44 wrong units found, 2 false flags), so their flags estimate the true error rate closely.
Estimated share of wrong units: train 15.1% (95% CI 8%–22%), val 27.4% (15%–43%), test 16.9% (10%–25%).
Errors cluster in some lines and writers (one val line of writer 49: 14 of 20 units wrong). Most common: separate ligatures merged, then marks attached to the wrong unit.
Every unit is still self-consistent (its image and trajectory come from the same strokes), so division errors change the unit mix but do not mislabel trajectories.
An integrity audit of all 2,403 lines and 42,370 units passed (checksums, stroke partition, points vs raw CSV, writer-disjoint splits). Details: REVIEW_2026-10-16.json.
