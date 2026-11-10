# SPDX-License-Identifier: Apache-2.0
"""Recorded-input preprocessing for the fixed-shape BEVFormer-Tiny adapter.

Conventions follow BEVFormer commit 66b65f3: nuscenes_dataset.get_data_info,
forward_test, and the Tiny Normalize -> half-scale -> pad test pipeline.
This module does not infer calibration, fill missing motion data, or run a model.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def absolute_bus(frame):
    bus = array(frame["can_bus"], (18,), "can_bus").copy()
    pose = array(frame["ego_translation"], (3,), "ego_translation")
    q = array(frame["ego_rotation_wxyz"], (4,), "ego_rotation_wxyz")
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
        raise ValueError("Ego quaternion must be unit length, ordered w,x,y,z")
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    degrees = np.degrees(yaw) % 360.0
    bus[:3], bus[3:7] = pose, q
    bus[-2], bus[-1] = np.radians(degrees), degrees
    return bus


def motion_bus(frame, previous):
    bus = absolute_bus(frame)
    history = previous is not None and previous["scene_token"] == frame["scene_token"]
    if history:
        if frame["timestamp_us"] <= previous["timestamp_us"]:
            raise ValueError("Frames within a scene must increase in timestamp")
        if frame["prev_token"] != previous["sample_token"]:
            raise ValueError(
                "Selected sequence skips a frame; select consecutive samples"
            )
        old = absolute_bus(previous)
        bus[:3] -= old[:3]
        # Deliberately preserve upstream subtraction across the 0/360 boundary.
        bus[-1] -= old[-1]
    else:
        bus[:3], bus[-1] = 0, 0
    return bus.astype(np.float32)[None], history


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "bevformer-recorded-v1":
        raise ValueError("Expected schema bevformer-recorded-v1")
    if manifest.get("input_kind") not in ("recorded", "synthetic-test"):
        raise ValueError(
            "input_kind must explicitly identify recorded or synthetic-test inputs"
        )
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("No frames")
    tokens = set()
    for frame in frames:
        token = frame["sample_token"]
        if not token or token in tokens:
            raise ValueError("Sample tokens must be nonempty and unique")
        tokens.add(token)
        if not frame["scene_token"] or not isinstance(frame["timestamp_us"], int):
            raise ValueError("Scene token and integer timestamp_us are required")
        absolute_bus(frame)
        if set(frame["cameras"]) != set(CAMERAS):
            raise ValueError("All six named cameras are required")
        for camera in CAMERAS:
            c = frame["cameras"][camera]
            matrix = array(c["lidar2img"], (4, 4), camera + " lidar2img")
            if not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError("lidar2img must be a homogeneous projection matrix")
            image = Path(c["image"])
            if not image.is_absolute():
                image = path.parent / image
            c["image"] = str(image.resolve(strict=True))
    previous = None
    for frame in frames:
        motion_bus(frame, previous)
        previous = frame
    return manifest


def preprocess(frame):
    """Original 1600x900 BGR -> normalize RGB -> 800x450 -> zero-pad 800x480."""
    tensors, projections = [], []
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    scale = np.diag([0.5, 0.5, 1, 1])
    for name in CAMERAS:
        camera = frame["cameras"][name]
        bgr = cv2.imread(camera["image"], cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (900, 1600, 3):
            raise ValueError(f"{name}: expected an original 1600x900 image")
        # Match MMCV imnormalize_'s OpenCV arithmetic as well as its RGB order.
        normalized = bgr.astype(np.float32)
        cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB, normalized)
        cv2.subtract(normalized, mean.reshape(1, -1).astype(np.float64), normalized)
        cv2.multiply(normalized, 1 / std.reshape(1, -1).astype(np.float64), normalized)
        resized = cv2.resize(normalized, (800, 450), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((480, 800, 3), dtype=np.float32)
        padded[:450] = resized
        tensors.append(padded.transpose(2, 0, 1))
        projections.append(scale @ array(camera["lidar2img"], (4, 4), name))
    return (
        np.ascontiguousarray(np.stack(tensors)[None]),
        np.asarray(projections, dtype=np.float32)[None],
    )


def decode(logits, boxes, threshold=0.3):
    """Same sigmoid/top-300/range filtering as decode_detections.cpp, then display cutoff."""
    logits = array(logits, (900, 10), "logits")
    boxes = array(boxes, (900, 10), "boxes")
    if not 0 <= threshold <= 1:
        raise ValueError("Score threshold must be between zero and one")
    # Float32 matches the C++ decoder's score calculation and tie ordering.
    logits = logits.astype(np.float32)
    scores = np.exp(-np.logaddexp(np.float32(0), -logits)).ravel()
    indices = np.lexsort((np.arange(scores.size), -scores))[:300]
    result = []
    for index in indices:
        b = boxes[index // 10]
        if scores[index] < threshold or not (
            -61.2 <= b[0] <= 61.2 and -61.2 <= b[1] <= 61.2 and -10 <= b[4] <= 10
        ):
            continue
        w, length, height = np.exp(b[[2, 3, 5]])
        if not np.isfinite([w, length, height]).all():
            raise ValueError("Nonfinite decoded box dimensions")
        result.append(
            dict(
                query=int(index // 10),
                label=int(index % 10),
                score=float(scores[index]),
                cx=float(b[0]),
                cy=float(b[1]),
                z_bottom=float(b[4] - height / 2),
                w=float(w),
                l=float(length),
                h=float(height),
                yaw=float(np.arctan2(b[6], b[7])),
            )
        )
    return result


def corners(box):
    """LiDAR box convention: dimension w along local x, l along y; z is bottom."""
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    xy *= [box["w"] / 2, box["l"] / 2]
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    xy = xy @ np.array([[c, s], [-s, c]]) + [box["cx"], box["cy"]]
    return np.vstack(
        [
            np.column_stack([xy, np.full(4, box["z_bottom"])]),
            np.column_stack([xy, np.full(4, box["z_bottom"] + box["h"])]),
        ]
    )


def project_edge(p0, p1, matrix, near=0.1):
    """Clip in camera depth before perspective division; off-image clipping is by axes."""
    projected = np.column_stack([np.stack([p0, p1]), np.ones(2)]) @ np.asarray(matrix).T
    a, b = projected[0].copy(), projected[1].copy()
    if max(a[2], b[2]) < near:
        return None
    if a[2] < near:
        a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
    if b[2] < near:
        b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
    return np.stack([a[:2] / a[2], b[:2] / b[2]])
