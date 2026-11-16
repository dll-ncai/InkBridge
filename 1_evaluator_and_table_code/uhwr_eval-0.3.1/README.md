# uhwr_eval 0.3.1

Shared evaluator for offline->online handwriting trajectory recovery (uhwr baselines phase).

* `uhwr_eval/_nagamatsu_metrics.py` is `src/utils/metrics.py` from
  hirokinaga/Handwriting-Trajectory-Recovery-with-Diffusion-Models, commit
  7c5cf62392e36cdc9f86445b4acb16c1e9198c68, copied byte-for-byte
  (sha256 7e2d330558f2ddf828cbd618ee80b36cada1818820315523ad826d2f3bbbb0c4; checked at runtime by
  `validate_toy.run`).  DTW, LDTW and AIoU come only from this file.
* Shared stroke format: list of (n,2) float arrays in evaluation-canvas pixels (y down), drawing
  order, no NaN rows.  `to_nagamatsu()` produces [x, y, pen, eos] in [-1,1] with pen=-1 on the last
  point of each stroke and eos=-1 on the final point (the code's convention).
* `score(gt, pred, canvas, protocol)` with protocol `native` (as recorded / as emitted) or
  `standardised` (per-stroke arc-length resampling, 1 px step) returns all metrics; `METRICS` lists
  direction, tag (established/proposed) and definition for each.
* Canvases: ligature units 64x64 (isotropic fit in 56x56); lines height 64 (ink bbox height 56 px).
* `python -c "from uhwr_eval import validate_toy as V; V.run('metric_sanity_check.csv')"` reproduces the
  DTW/LDTW/AIoU toy values (asserted to 1e-3) and writes toy_validation.csv.

Dependencies: numpy, scipy, opencv-python, tslearn, numba, pandas, torch, torchvision, lpips
(LPIPS downloads torchvision AlexNet weights on first use).

## Changes in 0.2.0 (16 Oct 2026; plan item S4, timing error)

* New `uhwr_eval/timing.py` with two proposed timing metrics, registered in `METRICS`:
  * `timing_profile_tv` (lower is better, 0-1): for each matched stroke pair (same Hungarian
    stroke matching as the order metrics; prediction oriented to the GT stroke), the
    total-variation distance between the dwell-time distributions over 16 normalised
    arc-length bins, averaged with GT stroke durations as weights.
  * `timing_duration_tv` (lower is better, 0-1): total-variation distance between GT and
    predicted shares of pen-down time over matched strokes (>= 2 matches).
  Both ignore pen-up (hover) time and are NaN when either side has no timestamps.
* `score(..., gt_times=None, pred_times=None)`: optional per-stroke timestamp arrays aligned with
  the stroke points (timing is computed on the inks as given, before any resampling protocol).
* `score(..., aiou_gt_thickness=2, aiou_max_dilation=4)`: AIoU settings (Nagamatsu
  `adaptive_iou` arguments); defaults reproduce 0.1.0 exactly.
* Every 0.1.0 metric and default is unchanged; `_nagamatsu_metrics.py` is byte-identical.
* `tests_timing.py`: checks identity = 0, path-preserving time warp > 0, stroke-order swap = 0,
  direction reversal with times reversed = 0.

## Changes in 0.3.0 (19 Oct 2026; plan item DOT2, mark direction metric)

* New `uhwr_eval/marks.py`: `mark_order_metrics(gt, pred, canvas, gt_is_mark, gt_group=None)` returns
  body/mark-split direction accuracy (chord >= 2 px and >= 4 px), body/mark recall, body/mark Kendall
  tau, a mark-before-body order flag, `mark_slot_exact` / `mark_slot_body_exact` (`gt_mark_before_body_flag`, `mark_before_body_recall`,
  `mark_after_body_kept`) and `mark_slot_exact`, each with an `n_` count.  Registered in `MARK_METRICS`.
* Stroke matching is `metrics.match_strokes`, unchanged (the matcher behind `kendall_tau` / `direction_acc`).
* `score()` and every 0.1.0 / 0.2.0 metric and default are unchanged; `metrics.py`, `strokes.py`,
  `render.py`, `timing.py` and `_nagamatsu_metrics.py` are byte-identical to 0.2.0.
* `tests_marks.py`: toy checks (identity, reversed marks, marks moved to the end).

## Changes in 0.3.0 (19 Oct 2026; plan item E1-v2, pen-up and timing redesign; separate 0.3.0 build)

* New `uhwr_eval/v2metrics.py`: `penup_f1_v2` (orientation/order-invariant pen-up F1 on unordered stroke
  end-point pairs, Hungarian, TP if cost <= 3 px) and `timing_profile_tv_v2` (as `timing_profile_tv`, predicted
  segment midpoints binned by projected arc position on the matched GT stroke).  Both registered in `METRICS`
  and returned by `score()` (`timing_profile_tv_v2` only when timestamps are given).
* `metrics.py` edits: two `METRICS` entries, two calls in `score()`; every 0.2.0 metric value is unchanged.

## Changes in 0.3.1 (20 Oct 2026; merge of the two 0.3.0 builds)

* Two different trees were labelled 0.3.0 (E1-v2: tarball sha256
  a9d0a1e130f11addadc5321e52f5e17989e1eb54d950314a3e344873cce0ed02; DOT2: tarball sha256
  143627cc883bce7646a1d733b08e3402b6bfc25e22e43643f5c7e474c513f6a3).  0.3.1 contains both sets of additions.
* `metrics.py` and `v2metrics.py` are byte-identical to the E1-v2 build; `marks.py`, `tests_marks.py` and every
  other file except `__init__.py`, `README.md`, `pyproject.toml` are byte-identical to the DOT2 build.
* `__init__.py` = 0.2.0 + both 0.3.0 docstring entries + the `marks` import + version 0.3.1.
* New `metric_registry_0.3.1.csv` (generated from `METRICS` and `MARK_METRICS`); `metric_registry.csv` is kept
  byte-identical to 0.2.0.  See CHANGELOG.md for the verification record.
