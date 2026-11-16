# OUHD-L ligature division v3.0: segmentation procedure

Source of this description: the module docstring of `build/ligseg_v3.py` (sha256 c08b62d3404e75ffec11316bdddfc8d2f95b4f234dd8c133b0154d36f3cee983), the parameters stored in `PROVENANCE.json`, and `README.md`, all in this folder. Nothing here is new.

## What a unit is
One ligature as written: a maximal run of joined letters (single isolated letters included) with its own dots and marks. Units are found per line by aligning the pen-down strokes, in writing order, to the transcript of the line. Per-letter stroke lengths are fitted on the training split only.

## Procedure (verbatim module docstring)
```
OUHD-L ligature division, version 3: per-line, transcript-aligned (14-15 Oct 2026).

Each line is solved on its own, from its own pen-down strokes (in recording order) and its own
transcript.  No unit is decided by a global threshold alone.

1. Stroke classes.  Every pen-down stroke is a *dot* (both bbox sides < 0.18 x 56 px), a
   *secondary* stroke (delayed and small: path < 30 px and height < 12 px; or delayed and a
   near-straight bar: chord/path > 0.9, path < 40 px, height < 20 px; never a straight steep
   stroke with height > 2 x width), or a *body* stroke.  'Delayed' = it x-overlaps (>= 50 % of
   its width) an earlier-started stroke that extends below it, or an earlier-started stroke lies
   to its left.  Same rules as v2, applied per stroke instead of per touching component.
2. Text ligatures.  The transcript is split into ligatures after Urdu non-joining letters
   (ا آ أ د ڈ ذ ر ڑ ز ژ و ؤ ے ۓ ء ۂ ۃ); digits and punctuation are separate.
3. Alignment.  Dynamic programming over (body strokes in time order) x (text ligatures).  A
   group of 1..R consecutive body strokes is aligned to 1..MJ consecutive ligatures of one word.
   Cost of a group = gap cost (each later stroke's nearest distance to the group's earlier
   strokes, beyond 1 px, / 5 px) + length cost lam * log((L+5)/(E+5))^2, where L is the group's
   ink path length and E the expected length (sum of per-letter lengths x a per-line scale) +
   touch penalty when the group's first stroke touches the previous body stroke + Sj per extra
   joined ligature + Sm if a multi-stroke group spans several ligatures.  Skipping a text
   ligature costs St, leaving a body stroke unaligned costs Si (it is then attached as a mark).
   Touching ligatures are split whenever they are separate strokes.  Optional ``gmax`` (off by
   default) forbids a group whose later stroke is more than gmax px from the group's earlier strokes;
   On the 160 judged regions (Sj = 1.0), gmax = None / 8 / 5 / 3 / 2 px matched judge 1's winner in
   58 / 60 / 59 / 62 / 62 of 130 regions and judge 2's winner in 102 / 101 / 95 / 91 / 90 of 145, so it
   slightly raises agreement with judge 1 and lowers it with judge 2; it is off by default.
4. Marks (dots, secondary strokes, unaligned strokes) go to the group minimising
   2*dx + 0.5*d_nearest (px) + timing cost (0 for the group most recently started before the
   mark; wsame for an earlier group of the same word; wother otherwise) + wexp if the group's
   text has no dotted or marked letter.
5. Per-letter lengths are fitted on the TRAIN split only (EM: solve, then non-negative least
   squares of group length on letter counts, normalised per line; letters with < 30
   occurrences keep 1.0).

CLI:  python ligseg_v3.py fit   --root DATA --out letter_lengths.json
      python ligseg_v3.py apply --root DATA --split test --lengths letter_lengths.json --out DIR
```

## Parameters used for v3.0 (PROVENANCE.json, method.params)
`{"wc": 1.0, "lam": 1.0, "wt": 1.5, "St": 4.0, "Si": 3.0, "R": 6, "Sj": 1.0, "MJ": 3, "Sm": 4.0, "wsame": 3.0, "wother": 10.0, "wexp": 8.0, "gmax": null}`

Letter lengths: `build/letter_lengths.json` (sha256 2f23011622e60314a98458929836fc61d7bfeb584d4700b5a005dbbf94d21ba2), fitted on the training split.

## Unit boundaries
`ligatures_v3_<split>.csv` has one row per unit: `unit_id` (= `<sample_id>_L<writing rank>`), `sample_id` (the OUHD-L line), `writer_id`, `stroke_idx` (indices into the pen-down strokes of the line that form the unit), `body_stroke_idx`, `mark_stroke_idx`, `aligned_text`, `rtl_rank`, and the bounding box in tablet units. Rows: train 33,615, val 4,349, test 4,406.
`ligatures_v3_test.npz` and `ligatures_v3_val.npz` hold the ground-truth points per unit. The training npz (95 MB) is left out for size; it is rebuilt from the Zenodo release (doi 10.5281/zenodo.20642162) with:
`cd build && PYTHONPATH=. python build_ligature_dataset.py --root <OUHD-L root> --lengths letter_lengths.json --out <dir>`
Output checksums of all three splits are in `PROVENANCE.json`.

## Quality records
`README.md` (exact-unit F1 on 21 hand-corrected test lines; reviewer estimate of wrong units per split), `REVIEW_2026-10-16.json`, `review_flags_per_unit.csv`.

## Not established by this archive
Whether division v3.0 itself is part of the Zenodo record was not checked during the archive build.
