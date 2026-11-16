"""Wrapper around Momna Waheed's Stages 1-3 (dll-<site>/Synthetic-Stroke-Trajectory).

Uses the repo's own ``PathBExtractor`` pipeline unchanged (U-Net ``unet_phase2_thinner.pt``
-> instance extraction -> skeleton multi-pass walk -> learned stroke order with
``shape_cnn_best_noDIAC.pt`` + ``order_predictor.pt``), default thresholds, as configured
in ``stage5_two_stage_training/generate_synthetic.py``.  The only change is the output
hook: instead of ``path_b_polylines_to_tokens`` (which z-scores coordinates), we keep the
per-stroke polylines in U-Net canvas pixels (64 x 1600) and map them back through the
letterbox transform returned by ``preprocess_uhwr_image``.  Stage 4/5 (recogniser) and
``simplify_trajectory`` are not applied.

Two output readings are returned:
  * ``emitted``: one stroke per detected component, its skeleton sub-paths concatenated in
    walk order (exactly the stroke segmentation the recogniser sees: pen_state = 1 only on
    the last point of the component);
  * ``subpaths``: every skeleton sub-path as its own stroke.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("MOMNA_REPO", Path(__file__).resolve().parents[1] / "ext" / "momna"))
for sub in ["", "common", "stage1_unet_segmentation", "stage2_trajectory_recovery",
            "stage3_order_predictor", "stage4_recognizer", "stage5_two_stage_training"]:
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import config  # noqa: E402
from recog_datasets import PathBExtractor  # noqa: E402
from test_uhwr import preprocess_uhwr_image  # noqa: E402
from clean_preprocess import preprocess_uhwr_image_clean  # noqa: E402
from infer_unet import predict_semantic, extract_instances  # noqa: E402
from full_pipeline_unet_uhwr import instance_mask_to_components  # noqa: E402
from template_stitcher import extract_skeleton  # noqa: E402
from mask_to_training_patch import order_skeleton_multi_pass, _smooth_polyline, mask_to_training_style_patch  # noqa: E402
from order_predictor import extract_spatial_features  # noqa: E402


class MomnaStages123(PathBExtractor):
    def __init__(self, device="cpu"):
        torch.set_num_threads(int(os.environ.get("UHWR_TORCH_THREADS", "4")))
        super().__init__(unet_ckpt=config.UNET_DEPLOY, device=torch.device(device),
                         classifier_ckpt=config.SHAPE_CNN, vocab_dir=config.VOCAB_DIR_K100,
                         order_ckpt=config.ORDER_PREDICTOR, knn_index=config.KNN_INDEX)
        assert self.use_learned_order, "learned order predictor failed to load"

    def recover(self, raw_gray: np.ndarray):
        """raw_gray: uint8 line image, dark ink on light background (any size).
        Returns dict(emitted=[...], subpaths=[...]) with strokes in raw_gray pixel coords."""
        image, meta = preprocess_uhwr_image(raw_gray)
        pp = preprocess_uhwr_image_clean((np.clip(image, 0, 1) * 255).astype(np.uint8),
                                         min_noise_area=self.min_noise_area,
                                         diacritic_keep_distance=self.diacritic_keep_distance,
                                         compute_skeleton=False)
        clean = pp["clean_binary"]
        H, W = clean.shape
        probs = predict_semantic(self.unet, clean, self.device)
        inst = extract_instances(probs, self.thresh_interior, self.thresh_boundary, self.min_area)
        comps = instance_mask_to_components(inst, self.min_area)
        n = len(comps)
        if n == 0:
            return dict(emitted=[], subpaths=[], meta=meta)
        patches = np.stack([mask_to_training_style_patch(c) for c in comps])
        batch = torch.from_numpy(patches).unsqueeze(1).float().to(self.device)
        with torch.no_grad():
            emb = self.extractor_fn(batch).cpu().numpy()
        spatial = np.zeros((n, 6), dtype=np.float32)
        for i, c in enumerate(comps):
            spatial[i] = extract_spatial_features(c["bbox"], n, canvas_size=(H, W))
        with torch.no_grad():
            scores = self.order_model(torch.from_numpy(emb).float().to(self.device),
                                      torch.from_numpy(spatial).float().to(self.device)).cpu().numpy()
        sort_idx = np.argsort(scores)
        per_stroke = []
        for ci in sort_idx:
            c = comps[int(ci)]
            bx0, by0, bx1, by1 = c["bbox"]
            blob = c["mask"][by0:by1, bx0:bx1]
            if blob.sum() < 3:
                per_stroke.append([np.array([[c["x_center"], c["y_center"]]], np.float32)])
                continue
            skel = extract_skeleton(blob)
            if skel.sum() < 2:
                per_stroke.append([np.array([[c["x_center"], c["y_center"]]], np.float32)])
                continue
            subs = []
            for sp in order_skeleton_multi_pass(skel):
                sp = _smooth_polyline(sp, sigma=self.smooth_sigma).copy()
                sp[:, 0] += bx0
                sp[:, 1] += by0
                subs.append(sp)
            per_stroke.append(subs)
        # canvas (64x1600 letterbox) -> raw_gray pixels
        s, xo, yo = meta["scale"], meta["x_off"], meta["y_off"]
        if xo < 0 or yo < 0:  # centre-crop branch in preprocess_uhwr_image
            xo, yo = -max(-xo, 0), -max(-yo, 0)
        back = lambda a: np.c_[(a[:, 0] - xo) / s, (a[:, 1] - yo) / s].astype(np.float64)
        emitted = [back(np.vstack(subs)) for subs in per_stroke if len(subs)]
        subpaths = [back(sp) for subs in per_stroke for sp in subs if len(sp)]
        return dict(emitted=emitted, subpaths=subpaths, meta=meta)
