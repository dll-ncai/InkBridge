"""uhwr_eval: shared trajectory-recovery evaluator for the uhwr baselines phase.

0.3.1 (20 Oct 2026): merge of the two 0.3.0 builds (E1-v2 tarball sha256 a9d0a1e1..., DOT2 tarball
sha256 143627cc...).  Contains both penup_f1_v2 / timing_profile_tv_v2 (v2metrics.py, in score()) and
marks.py (mark_order_metrics, separate call).  No metric code differs from either parent; see CHANGELOG.md.

0.3.0 (19 Oct 2026, DOT2): adds marks.py (mark_order_metrics: body/mark-split direction accuracy,
mark recall, mark-before-body order flag, mark slot).  score() and every 0.2.0 metric are unchanged.

0.2.0 (16 Oct 2026): adds timing_profile_tv / timing_duration_tv (timing.py), optional
per-point timestamps in score(), and AIoU GT-thickness / dilation-range arguments.
DTW/LDTW/AIoU defaults and every 0.1.0 metric are unchanged.

0.3.0 (19 Oct 2026, E1-v2): adds penup_f1_v2 and timing_profile_tv_v2 (v2metrics.py). Every 0.2.0 metric is
unchanged; the two new keys are added to score() output."""
from .strokes import (clean_ink, to_nagamatsu, from_normalized, from_nan_separated, from_pen_flags,
                      resample_ink_step, resample_ink_fixed_count, fit_transform_unit,
                      fit_transform_line, apply_transform)
from .render import render_mask, render_image
from .metrics import score, METRICS, PROTOCOLS, metric_table, NAGAMATSU_SHA256, NAGAMATSU_COMMIT

__version__ = "0.3.1"
from .timing import timing_metrics
from .marks import mark_order_metrics, MARK_METRICS
