"""Sanity checks for uhwr_eval 0.2.0 timing metrics: python tests_timing.py"""
import numpy as np
from uhwr_eval import score

rng = np.random.default_rng(0)
a = np.c_[np.linspace(10, 50, 60), 20 + 5 * np.sin(np.linspace(0, 3, 60))]
b = np.c_[np.linspace(50, 15, 40), np.linspace(40, 45, 40)]
gt = [a, b]
ta, tb = np.linspace(0, 0.3, 60) ** 1.3, 0.5 + np.linspace(0, 0.2, 40)
gt_t = [ta, tb]
r = score(gt, gt, (64, 64), gt_times=gt_t, pred_times=gt_t, with_lpips=False)
assert r["timing_profile_tv"] == 0 and r["timing_duration_tv"] == 0, r
u = lambda t: t + 0.2 * np.sin(2 * np.pi * (t - t.min()) / (np.ptp(t) + 1e-9)) * np.ptp(t) / (2 * np.pi)
r = score(gt, gt, (64, 64), gt_times=gt_t, pred_times=[u(ta), tb], with_lpips=False)
assert r["timing_profile_tv"] > 0.01 and r["aiou"] == score(gt, gt, (64, 64), with_lpips=False)["aiou"], r
r = score(gt, [b, a], (64, 64), gt_times=gt_t, pred_times=[tb - 0.5, ta + 0.25], with_lpips=False)
assert r["timing_profile_tv"] < 1e-9 and r["timing_duration_tv"] < 1e-9, r
r = score(gt, [a[::-1], b], (64, 64), gt_times=gt_t, pred_times=[(ta[0] + ta[-1] - ta)[::-1], tb], with_lpips=False)
assert r["timing_profile_tv"] < 1e-9 and r["direction_acc"] == 0.5, r
r = score(gt, gt, (64, 64), with_lpips=False)
assert np.isnan(r["timing_profile_tv"])
print("timing tests passed")
