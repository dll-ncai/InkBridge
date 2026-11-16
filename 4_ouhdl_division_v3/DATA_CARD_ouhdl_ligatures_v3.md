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
- `ligatures_v3_<split>.csv`: one row per unit (unit_id, sample_id, writer_id, writing_rank, rtl_rank, stroke indices into the line's pen-down strokes, aligned_text, bbox in tablet units, ...).
- `ligatures_v3_<split>.npz`: points (x, y, time, pressure, local stroke number; raw tablet units, writing order) + unit_offsets + unit_ids.
- `lines_v3_<split>.csv`: per line counts (units, text ligatures, skipped ligatures, unaligned strokes).
- `ouhdl_ligatures.py`: `idx, units = load_split(root, 'train'); units('<unit_id>')` → list of (n, 4) strokes.
- `PROVENANCE.json`: input checksums, code hashes, parameters, command, environment, output checksums.

Rebuild: `PYTHONPATH=<adapter>:<uhwr_eval>:<ligds> python build_ligature_dataset.py --root <ouhdl> --lengths letter_lengths.json --out <dir> --check-test units_v3_test.json`
(code in ~/Work/uhwr/baselines/ligseg_v3/ and this folder's build/ copy).
