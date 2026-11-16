"""Toy checks for uhwr_eval 0.3.0 mark_order_metrics.  python tests_marks.py"""
import numpy as np
from uhwr_eval import mark_order_metrics, score

def line(x0, x1, y):
    return np.c_[np.linspace(x0, x1, 20), np.full(20, y)]

body1 = line(60, 30, 40); dot1 = line(48, 44, 20); body2 = line(28, 5, 40); dot2 = line(18, 14, 52)
gt = [body1, dot1, body2, dot2]; isM = [False, True, False, True]; grp = [0, 0, 1, 1]
c = (64, 64)
r = mark_order_metrics(gt, gt, c, isM, grp)
assert r['mark_direction_acc'] == 1 and r['body_direction_acc'] == 1 and r['mark_slot_exact'] == 1 and r['mark_slot_body_exact'] == 1 and r['mark_recall'] == 1, r
rev = [body1, dot1[::-1], body2, dot2[::-1]]
r = mark_order_metrics(gt, rev, c, isM, grp)
assert r['mark_direction_acc'] == 0 and r['body_direction_acc'] == 1, r
late = [body1, body2, dot1, dot2]
r = mark_order_metrics(gt, late, c, isM, grp)
assert r['mark_slot_exact'] == 0.0 and r['mark_slot_body_exact'] == 0.5 and r['mark_after_body_kept'] == 1, r
early = [dot1, body1, body2, dot2]
r = mark_order_metrics(gt, early, c, isM, grp)
assert r['gt_mark_before_body_flag'] == 0 and r['mark_after_body_kept'] == 0.5, r
r2 = mark_order_metrics(early, gt, c, [True, False, False, True], [0, 0, 1, 1])
assert r2['gt_mark_before_body_flag'] == 1 and r2['mark_before_body_recall'] == 0, r2
s = score(gt, late, c, with_lpips=False)
assert np.isfinite(s['kendall_tau'])
print('tests_marks ok')
