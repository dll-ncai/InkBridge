"""Build uhwr_eval 0.3.0 from the 0.2.0 tree by scripted, asserted text patches (record in a diff)."""
import os, shutil, sys, subprocess
src, dst, v2 = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "._*"))
shutil.copy(v2, os.path.join(dst, "v2metrics.py"))
p = os.path.join(dst, "__init__.py"); s = open(p).read()
assert '__version__ = "0.2.0"' in s
s = s.replace('__version__ = "0.2.0"', '__version__ = "0.3.0"')
s = s.replace('"""uhwr_eval: shared', '"""uhwr_eval: shared', 1)
s = s.replace('every 0.1.0 metric are unchanged."""', 'every 0.1.0 metric are unchanged.\n\n0.3.0 (19 Oct 2026, E1-v2): adds penup_f1_v2 and timing_profile_tv_v2 (v2metrics.py). Every 0.2.0 metric is\nunchanged; the two new keys are added to score() output."""')
open(p, "w").write(s)
p = os.path.join(dst, "metrics.py"); s = open(p).read()
a1 = '    "loop_orient_acc": ('
assert s.count(a1) == 1
s = s.replace(a1, '    "penup_f1_v2": ("\\u2191", "proposed", "uhwr_eval 0.3.0: orientation/order-invariant pen-up F1 on unordered stroke end-point pairs; cost = min over orientation of max end-point distance; Hungarian; TP if <= 3 px"),\n    "timing_profile_tv_v2": ("\\u2193", "proposed", "uhwr_eval 0.3.0: timing_profile_tv with predicted segment midpoints binned by projected arc position on the matched GT stroke (not the prediction\'s own arc length); no orientation step"),\n' + a1)
a2 = '    res["penup_f1"] = penup_f1(gt, pr)\n'
assert s.count(a2) == 1
s = s.replace(a2, a2 + '    from .v2metrics import penup_endpoint_f1\n    res["penup_f1_v2"] = penup_endpoint_f1(gt, pr)\n')
a3 = '                   stroke_match_recall=0.0, strict_order_acc=0.0, penup_f1=0.0, aiou=0.0)'
assert s.count(a3) == 1
s = s.replace(a3, '                   stroke_match_recall=0.0, strict_order_acc=0.0, penup_f1=0.0, penup_f1_v2=0.0, aiou=0.0)')
a4 = '        res.update(timing_metrics(g0, p0, gt_t, pr_t, match_strokes(g0, p0, canvas)))\n'
assert s.count(a4) == 1
s = s.replace(a4, '        _m = match_strokes(g0, p0, canvas)\n        res.update(timing_metrics(g0, p0, gt_t, pr_t, _m))\n        from .v2metrics import timing_profile_tv_v2\n        res["timing_profile_tv_v2"] = timing_profile_tv_v2(g0, p0, gt_t, pr_t, _m)\n')
open(p, "w").write(s)
d = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "._*", src, dst], capture_output=True, text=True).stdout
open(os.path.join(os.path.dirname(dst), "uhwr_eval_0.2.0_to_0.3.0.diff"), "w").write(d)
print("built", dst, "diff lines", len(d.splitlines()))
