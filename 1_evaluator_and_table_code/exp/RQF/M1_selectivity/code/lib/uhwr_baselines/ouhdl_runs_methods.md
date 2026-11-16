# OUHD-L baseline runs: methods note (uhwr baselines phase)

All numbers produced by these runs carry provenance `our-run`. Per-sample scores are in
`ouhdl_baseline_results.csv`, and aggregates (mean, SD, bootstrap 95% CI, n) are in the `results_rows` returned to the
baselines matrix.

## 1. Data

* **Dataset.** OUHD-L v1.0, Zenodo record 10.5281/zenodo.20642162. Only the core archive `ouhdl_v1.0_core.tar` was
  used (8,610,631,680 bytes). Its sha256, d2cdf97b…5278ae, matches the record's SHA256SUMS. The archive was fetched
  in parallel byte ranges because a single stream ran at about 0.2 MB/s.
* **Local copy.** The extracted data are in `$LOCAL/data/ouhdl`. The 129 downloaded `parts/`
  chunks are still there (8.6 GB) and can be deleted.
* **Split.** We used the Zenodo writer-disjoint manifests: train 1,911, val 239 and **test 253 lines (32 writers)**.
  All runs are on the full test split unless a subset is declared below.
* **Sample IDs.** IDs have the form `<series>_<board>_<line>`, e.g. `1000_0014_5`. Board IDs repeat across the 8
  line series, so the file basename alone is not unique: only 215 of the 253 basenames are distinct.
* **Strokes.** A stroke is a maximal run of rows with `pen_down == 1`. Hover rows (`pen_down == 0`, pressure 0) are
  dropped. Coordinates come from `X cood.` / `Y cood.` (tablet units, y pointing down). There are no NaN or sentinel
  rows. Test lines have 15–94 strokes (median 37) and 2,297–20,563 pen-down samples (median 5,625, sampled at 500 Hz).
* **Paired images.** OUHD-L ships no image of the real physical ink; there are no scans of the Wacom paper. The
  `img/*.png` files are renderings of the same online strokes, for four reasons:
  * They are tight min–max crops of the raw coordinates with isotropic scale (s_x ≈ s_y ≈ 0.09–0.12 px per tablet
    unit).
  * 99–100 % of pen-down samples fall on ink.
  * Stroke width is near-uniform (median 2.8–4 px) and does not track pressure (Spearman ρ between −0.18 and 0.00 on
    11 test lines).
  * The `Thickness` column is constant within a line.

  The two input conditions are therefore:
  * **rendered**: our PEN-Net-style rendering of the GT strokes, anti-aliased, black on white, with a uniform width of
    2 px per 64 px of evaluation-canvas height.
  * **dataset**: the OUHD-L `img/` PNG exactly as shipped. This is a second, independent renderer with a different
    width, anti-aliasing and resolution. It is *not* a real-ink scan.

  A real-ink condition needs the physical Wacom sheets to be scanned and registered (see the next steps).

## 2. Units and normalisation

* **Line unit.** All strokes in writing order, mapped to an evaluation canvas of height 64 px. The isotropic scale
  maps the ink bounding-box height to 56 px with 4 px margins; width = ceil(56·aspect) + 8 (test range 269–918 px,
  median 557). All line metrics are in these 64-px-height pixel units and are reported separately from ligature
  metrics.
* **Ligature / connected-component unit.** Built in line-canvas space:
  1. Draw every stroke as a 2 px mask.
  2. Merge strokes whose masks touch or overlap under 8-connectivity (union-find).
  3. Mark a component as a *diacritic* if both bounding-box sides are < 0.18 × 56 px (≈ 10 px). Every other
     component is a *main body*.
  4. Attach each diacritic to the main body whose x-extent contains the diacritic's x-centre (nearest ink among
     those if there are several). If none qualifies, attach it to the main body with the nearest ink point.
  5. Keep the unit's strokes in their original temporal order and order units right-to-left by main-body max x.
  6. Fit each unit isotropically into 56 × 56 px, centred on a **64 × 64** canvas. This is the space used by
     Nagamatsu's metrics.

  The 253 test lines give **4,796 units** (9–30 per line, median 19; 0.71 diacritic strokes per unit on average).

  Known limitations: strokes of one letter that do not touch (the gaf bar, a detached keshida) become separate units
  unless they are small enough to count as diacritics. Kerned ligatures whose ink overlaps merge into one unit.
* **Oracle segmentation.** Ligature units come from the GT. In the dataset-image condition, each unit's crop is
  cleaned by erasing all ink farther than 5 px (image pixels) from the unit's own strokes. The same oracle defines
  which strokes are rendered in the rendered condition.

## 3. Evaluator (`uhwr_eval` 0.1.0)

* **Nagamatsu metrics.** DTW, LDTW and AIoU come from Nagamatsu `src/utils/metrics.py`, commit 7c5cf623, copied
  byte-for-byte (sha256 7e2d3305…). Conventions:
  * The input encoding is [x, y, pen, eos] in [-1, 1], with x_n = 2x/W − 1, pen = −1 on the last point of each
    stroke and eos = −1 on the final point.
  * DTW is tslearn's: the square root of the summed squared distances along the path, in canvas pixels.
  * AIoU draws the GT at 2 px and the prediction at 1 px, and takes the best IoU over 0–4 dilations. Strokes with
    fewer than 2 points are not drawn.
* **Other metrics.** Each metric has a tag and a definition in `metric_registry.csv`:
  * established: LPIPS (AlexNet; 2 px renderings), Chamfer (symmetric mean), Hausdorff, discrete Fréchet (on the
    concatenated drawing-order sequence), RMSE (each ink resampled to K points uniformly in pen-down arc length;
    K = 320 for ligatures, K = 2048 for lines) and stroke-count error.
  * proposed: pen-up F1, stroke-match recall and precision, strict order accuracy, Kendall τ, direction accuracy and
    loop orientation agreement.
  * Pen-up F1: stroke end points matched one-to-one (Hungarian), counted as a hit within 3 px.
  * Strict order accuracy, Kendall τ, direction accuracy and loop orientation all rest on one-to-one Hungarian
    matching of 5 px stroke masks, accepting a match at IoU ≥ 0.5. Splits and merges therefore fail strict order.
  * Direction accuracy counts only matched strokes whose GT start and end are ≥ 2 px apart.
  * Loop orientation counts only matched strokes with |signed area| ≥ 4 px² and 4π|A|/L² ≥ 0.1.
* **Protocols.** *native* scores the GT as recorded and the prediction as emitted. *standardised* first resamples
  both per stroke at a fixed 1 px arc-length step. Order and stroke metrics do not depend on the protocol and are
  reported once.
* **Toy validation.** On the toy glyph in `metric_sanity_check.csv`, the native protocol reproduces DTW/LDTW/AIoU to
  1e-3 (asserted in `validate_toy.run`). The new metrics catch what AIoU misses:
  * swapped strokes give strict order 0 and τ = −1;
  * a reversed bar gives direction accuracy 0;
  * a reversed loop gives loop orientation 0;
  * AIoU stays at 0.966 in all three cases.

## 4. Methods run

| id | method | units | inputs | notes |
|---|---|---|---|---|
| geo_rtl | Skeleton/graph geometric baseline, RTL prior, dots last | line, ligature | rendered (f = 4), dataset | Otsu → skeletonize → graph cut at junction clusters. Walk smallest-turn edges, starting each body at its rightmost (then topmost) endpoint. Main bodies right-to-left by max x; dots (< 0.18 × unit height) after all main bodies, right-to-left. |
| geo_rtl_local | Same, dots emitted after their nearest main body | line | rendered, dataset | Line-level variant; for ligature units it is identical to geo_rtl. |
| momna_s123 | Momna Waheed's Stages 1–3 (dll-<site>/Synthetic-Stroke-Trajectory, shipped checkpoints) | line | rendered (64 × W, letterboxed to 64 × 1600 by the repo), dataset (PNG fed directly) | Repo `PathBExtractor` code, defaults: U-Net `unet_phase2_thinner.pt`, learned order (`shape_cnn_best_noDIAC.pt` + `order_predictor.pt`). Read out in canvas pixels before the z-scoring tokenizer; no `simplify_trajectory`; Stage 4/5 not used. One stroke per component, sub-paths concatenated as emitted. |
| momna_s123_subpaths | Same output, each skeleton sub-path counted as its own stroke | line | rendered, dataset | Diagnostic reading. |
| inksight_small_p | InkSight Small-p, zero-shot, public weights (HF `Derendering/InkSight-Small-p`, rev 8514f23) | ligature (declared subset) | rendered, dataset (256 × 256 unit crops) | Follows the notebook: `scale_and_pad` to 224 (black pad), prompt "Recognize and derender." with fallback "Derender the ink.", `detokenize`, then mapped back. CPU only (TF 2.20 / tensorflow-text 2.20.1, isolated venv). |

* **Momna and ligatures.** Momna is not run at ligature level: its U-Net and order predictor are line-level, with a
  64 × 1600 input.
* **InkSight and lines.** InkSight is not run at line level. It is a word-level model. Its page pipeline relies on
  docTR/Tesseract word detection, which does not cover Urdu Nastaliq, and line runs would cost about 19 crops per
  line at 15–600 s per crop on CPU.
* **InkSight subset.** Units were drawn at random from the 4,796 test units: 40 with `numpy.random.default_rng(0)`,
  plus an extension of 80 drawn with `default_rng(1)` from the remaining units. The same units are used in both
  input conditions. The units that were actually scored are listed in `ouhdl_baseline_results.csv`
  (`subset_draw` column), and their count is given in the results notes. The geometric baseline is also aggregated on
  exactly the same units (method `geo_rtl` on the InkSight subset) for a paired comparison.
* **InkSight runs completed.** All 120 units were scored in both conditions, from 99 lines and 31 writers. The primary
  prompt returned non-empty ink every time, so the fallback was never used. CPU time per crop was median 27 s
  (range 8–150 s) for rendered inputs and median 28 s (range 6–590 s) for dataset inputs; the longest crops are
  degenerate long outputs. Running 2–4 processes in parallel took about 2.5 h in total. The first test call on the
  repo's own word image took about 275 s, so even a 5,000-unit line-level run is out of reach on this CPU.
* **InkSight languages.** The model card lists en / zh / ja / vi; Urdu and Arabic script are absent. The recognised
  text it emits for Urdu crops is Latin, Vietnamese or CJK.

## 5. Aggregation

* The value is the mean over samples with a defined metric value. Undefined cases are excluded and counted:
  Kendall τ needs ≥ 2 matched strokes; direction and loop metrics need eligible strokes.
* SD uses ddof = 1. The 95 % CI is a percentile bootstrap with B = 2000 and seed 0, resampling lines. For ligature
  units it is a cluster bootstrap over lines.
* Strict order accuracy, direction accuracy and loop orientation are reported in %.
* No geometric or Momna prediction was empty, and no scoring call failed.

* **Single-stroke units.** 47.9 % of ligature units have exactly one GT stroke. For these, strict order accuracy
  reduces to 'exactly one predicted stroke matched', and Kendall τ is undefined (excluded). Ligature-level order
  numbers should be read with this in mind.

## 6. Files

* `ouhdl_baseline_results.csv`: one row per (sample/unit × method × input × protocol), with all metrics,
  bookkeeping fields (`n_gt_pts`, `n_pred_pts`, `n_gt_strokes`, `n_pred_strokes`, `runtime_s`), InkSight's prompt
  and recognised text, and the `in_inksight_subset` / `subset_draw` flags.
* `ouhdl_baseline_summary.csv`: headline table of means with 95 % CIs.
* `uhwr_eval-0.1.0.tar.gz`: the evaluator.
* `ouhdl_adapter_baselines_code.tar.gz`: the adapter and runners.
* `ouhdl_units_example.png`: unit segmentation example.
