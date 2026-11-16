# CHANGELOG uhwr_eval

## 0.3.1 (20 Oct 2026)

Merge of two parallel 0.3.0 builds into one release.

| parent | tarball sha256 | location on CLUSTER-A | adds |
|---|---|---|---|
| E1-v2 0.3.0 | a9d0a1e130f11addadc5321e52f5e17989e1eb54d950314a3e344873cce0ed02 | exp/E1_v2/out/uhwr_eval-0.3.0.tar.gz | v2metrics.py: penup_f1_v2, timing_profile_tv_v2 (in score()) |
| DOT2 0.3.0 | 143627cc883bce7646a1d733b08e3402b6bfc25e22e43643f5c7e474c513f6a3 | exp/DOT2/code/eval/uhwr_eval-0.3.0.tar.gz | marks.py: mark_order_metrics, MARK_METRICS; tests_marks.py |
| common base 0.2.0 | b1664bd3bce3a445dd2bfc06fee3487b534b09b95c2963bcf7575481dc966e8e | (artifact uhwr_eval-0.2.0.tar.gz) | - |

* Files changed by the E1-v2 build vs 0.2.0: uhwr_eval/__init__.py, uhwr_eval/metrics.py, +uhwr_eval/v2metrics.py.
* Files changed by the DOT2 build vs 0.2.0: README.md, pyproject.toml, uhwr_eval/__init__.py, +uhwr_eval/marks.py, +tests_marks.py.
* The only file both builds changed is uhwr_eval/__init__.py (docstring, version, one import); merged by text edits
  asserted in build_uhwr_eval_031.py.  No metric function differs from its parent.
* score() returns every 0.2.0 key with 0.2.0 values, plus penup_f1_v2 (always) and timing_profile_tv_v2 (with timestamps).
* Verification on CLUSTER-A (job record in release/PROVENANCE_uhwr_eval_0.3.1.json): score() vs 0.2.0 on DOT2's 711-unit
  check set and on E0 ms8 predictions for all 4,406 v3 test units (rendered + dataset inputs, native + standardised);
  E1-v2 rerun (penup_f1_v2, timing_profile_tv_v2) vs E1-v2 outputs; DOT2 dot2.py rerun (marks.py) vs DOT2 outputs.
  Results: release/verify_summary_uhwr_eval_0.3.1.json.

## 0.3.0 (19 Oct 2026) - two builds, see table above

## 0.2.0 (16 Oct 2026)

timing_profile_tv, timing_duration_tv; optional timestamps in score(); AIoU arguments.

## 0.1.0

Initial evaluator.
