"""Trajectory metrics used by handwriting generation and evaluation."""

import cv2
import numpy as np
from tslearn.metrics import dtw_path


def _validate_sequence(sequence, name):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {sequence.shape}.")
    if sequence.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if sequence.shape[1] < 2:
        raise ValueError(f"{name} must contain at least x and y coordinates.")
    if not np.isfinite(sequence).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return sequence


def trim_at_end_flag(trajectory, end_flag_channel=3):
    """Trim a time-major trajectory after its first negative end flag."""
    trajectory = _validate_sequence(trajectory, "trajectory")
    if trajectory.shape[1] <= end_flag_channel:
        return trajectory

    end_indices = np.flatnonzero(trajectory[:, end_flag_channel] < 0)
    if end_indices.size == 0:
        return trajectory
    return trajectory[: end_indices[0] + 1]


def dtw_and_ldtw(sequence_a, sequence_b):
    """Return DTW distance and path-length-normalized DTW distance."""
    sequence_a = _validate_sequence(sequence_a, "sequence_a")
    sequence_b = _validate_sequence(sequence_b, "sequence_b")

    path, dtw = dtw_path(sequence_a[:, :2], sequence_b[:, :2])
    if not path:
        raise RuntimeError("DTW returned an empty alignment path.")

    dtw = float(dtw)
    return dtw, dtw / len(path)


def _to_image_coordinates(sequence, image_shape):
    """Map normalized x/y coordinates from [-1, 1] to image coordinates."""
    sequence = _validate_sequence(sequence, "sequence").copy()
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError(
            f"image_shape must contain two positive values, got {image_shape}."
        )

    height, width = image_shape
    sequence[:, :2] = (sequence[:, :2] + 1.0) / 2.0
    sequence[:, :2] *= np.array([width, height], dtype=np.float32)
    return sequence


def dtw_distance(sequence_a, sequence_b):
    """Return the DTW distance between two trajectories."""
    dtw, _ = dtw_and_ldtw(sequence_a, sequence_b)
    return dtw


def ldtw_distance(sequence_a, sequence_b):
    """Return DTW distance normalized by the alignment-path length."""
    _, ldtw = dtw_and_ldtw(sequence_a, sequence_b)
    return ldtw


def _render_trajectory(trajectory, image_shape, thickness):
    trajectory = _validate_sequence(trajectory, "trajectory")
    if trajectory.shape[1] < 3:
        raise ValueError("trajectory must contain x, y, and pen-state channels.")

    height, width = image_shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    normalized = (trajectory[:, :2] + 1.0) / 2.0
    pixels = normalized * np.array([width - 1, height - 1], dtype=np.float32)

    pen_up_indices = np.flatnonzero(trajectory[:, 2] < 0) + 1
    strokes = np.split(pixels, pen_up_indices)
    strokes = [
        stroke.astype(np.int32).reshape((-1, 1, 2))
        for stroke in strokes
        if len(stroke) >= 2
    ]
    if strokes:
        cv2.polylines(
            canvas,
            strokes,
            isClosed=False,
            color=1,
            thickness=thickness,
        )
    return canvas


def adaptive_iou(
    ground_truth,
    prediction,
    image_shape=(64, 64),
    ground_truth_thickness=2,
    max_dilation_iterations=4,
):
    """Return the best IoU before and during prediction-mask dilation.

    ``max_dilation_iterations`` is the number of dilation operations, so a
    value of five evaluates six masks: the original prediction and the result
    after each of five successive dilations.
    """
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError(f"image_shape must contain two positive values, got {image_shape}.")
    if ground_truth_thickness <= 0:
        raise ValueError("ground_truth_thickness must be positive.")
    if max_dilation_iterations < 0:
        raise ValueError("max_dilation_iterations must be non-negative.")

    ground_truth_mask = _render_trajectory(
        ground_truth,
        image_shape,
        thickness=ground_truth_thickness,
    )
    prediction_mask = _render_trajectory(
        prediction,
        image_shape,
        thickness=1,
    )

    maximum_iou = 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_prediction = prediction_mask.copy()

    for dilation_iteration in range(max_dilation_iterations + 1):
        intersection = np.logical_and(ground_truth_mask, dilated_prediction).sum()
        union = np.logical_or(ground_truth_mask, dilated_prediction).sum()
        iou = float(intersection / union) if union else 0.0
        maximum_iou = max(maximum_iou, iou)
        if dilation_iteration < max_dilation_iterations:
            dilated_prediction = cv2.dilate(
                dilated_prediction, kernel, iterations=1
            )

    return maximum_iou


def trajectory_metrics(
    ground_truth,
    prediction,
    image_shape=(64, 64),
    ground_truth_thickness=2,
    max_dilation_iterations=4,
):
    """Calculate DTW, LDTW, and AIoU for two time-major trajectories."""
    ground_truth = trim_at_end_flag(ground_truth)
    prediction = trim_at_end_flag(prediction)
    ground_truth_image_coordinates = _to_image_coordinates(
        ground_truth,
        image_shape,
    )
    prediction_image_coordinates = _to_image_coordinates(
        prediction,
        image_shape,
    )
    dtw, ldtw = dtw_and_ldtw(
        ground_truth_image_coordinates,
        prediction_image_coordinates,
    )
    aiou = adaptive_iou(
        ground_truth,
        prediction,
        image_shape=image_shape,
        ground_truth_thickness=ground_truth_thickness,
        max_dilation_iterations=max_dilation_iterations,
    )
    return {"dtw": dtw, "ldtw": ldtw, "aiou": aiou}
