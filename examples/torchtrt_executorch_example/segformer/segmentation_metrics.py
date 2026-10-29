# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""19-class semantic IoU on human labels; not agreement with model predictions.

Cityscapes IDs: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
IoU: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
No dataset download or license acceptance is performed here.
"""

import numpy as np

LABELS = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
IGNORE = 255


def encode_mask(mask, encoding):
    """Map raw Cityscapes labelIds to contiguous train IDs without resizing."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype.kind not in "iu":
        raise ValueError(
            "Expected a nonempty 2D integer ID mask, not an RGB/color mask"
        )
    if encoding == "label_ids":
        if not np.isin(mask, list(range(34)) + [255]).all():
            raise ValueError("Unknown Cityscapes label ID")
        result = np.full(mask.shape, IGNORE, dtype=np.uint8)
        for train_id, label_id in enumerate(LABEL_IDS):
            result[mask == label_id] = train_id
        return result
    if encoding == "train_ids":
        if not np.isin(mask, list(range(19)) + [IGNORE]).all():
            raise ValueError("Expected train IDs 0..18 or ignore ID 255")
        return mask.astype(np.uint8)
    raise ValueError(f"Unknown mask encoding: {encoding}")


def confusion_matrix(truth, prediction):
    truth = encode_mask(truth, "train_ids")
    prediction = np.asarray(prediction)
    if prediction.shape != truth.shape or prediction.dtype.kind not in "iu":
        raise ValueError(
            "Prediction must be an integer map with the ground-truth shape"
        )
    if ((prediction < 0) | (prediction >= 19)).any():
        raise ValueError("Predicted class IDs must be 0..18")
    valid = truth != IGNORE
    encoded = truth[valid].astype(np.int64) * 19 + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=19 * 19).reshape(19, 19)


def summarize(matrix):
    """Aggregate counts first, then compute IoU. Never average per-image IoUs."""
    matrix = np.asarray(matrix)
    if matrix.shape != (19, 19) or matrix.dtype.kind not in "iu" or (matrix < 0).any():
        raise ValueError("Expected a nonnegative 19x19 integer confusion matrix")
    hits = matrix.diagonal()
    truth_count, predicted_count = matrix.sum(1), matrix.sum(0)
    unions = truth_count + predicted_count - hits
    per_class = {
        name: {
            "iou": float(hits[i] / unions[i]) if unions[i] else None,
            "ground_truth_pixels": int(truth_count[i]),
            "predicted_pixels": int(predicted_count[i]),
            "intersection": int(hits[i]),
            "union": int(unions[i]),
        }
        for i, name in enumerate(LABELS)
    }
    scores = [entry["iou"] for entry in per_class.values() if entry["iou"] is not None]
    total = int(matrix.sum())
    return {
        "valid_pixels": total,
        "pixel_accuracy": float(hits.sum() / total) if total else None,
        "mean_iou": float(np.mean(scores)) if scores else None,
        "evaluated_classes": len(scores),
        "absent_class_policy": "Exclude zero-union classes; false-positive-only classes have IoU 0",
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_axes": "rows=ground truth, columns=prediction",
    }
