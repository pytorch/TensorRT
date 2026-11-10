# SPDX-License-Identifier: Apache-2.0
"""Render six camera views and a metric BEV plot from saved BEVFormer outputs.

No inference or ground-truth drawing is performed here. Original image hashes
bind the pictures to the prepared inputs. Native outputs must pass a fresh
numerical comparison to the same case's eager reference before rendering.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bevformer-matplotlib")
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from recorded_scene import CAMERAS, CLASSES, corners, decode, project_edge, sha256

DISPLAY = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
EDGES = (
    [(i, (i + 1) % 4) for i in range(4)]
    + [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
    + [(i, i + 4) for i in range(4)]
)
COLORS = (
    "#76b900",
    "#2c7bb6",
    "#af7aa1",
    "#e69f00",
    "#8c564b",
    "#777777",
    "#e15759",
    "#009e73",
    "#cc79a7",
    "#f28e2b",
)


def read_output(path, shape):
    a = np.fromfile(path, dtype="<f4")
    if a.size != int(np.prod(shape)) or not np.isfinite(a).all():
        raise ValueError(f"Missing, wrong-size or nonfinite output: {path}")
    return a.reshape(shape)


def render(case, raw, output, source, threshold=0.3, extent=51.2):
    case, raw, output = Path(case), Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((case / "frame.json").read_text())
    frame = record["frame"]
    if record["input_kind"] not in ("recorded", "synthetic-test"):
        raise ValueError("Missing input provenance")
    if source == "pytorch" and raw.resolve() != case.resolve():
        raise ValueError("PyTorch source must use the prepared case outputs")
    if source == "native" and raw.resolve() == case.resolve():
        raise ValueError("Native source cannot point at eager-reference files")
    outputs, comparisons = [], []
    for i, shape in enumerate(((1, 900, 10), (1, 900, 10), (1, 2500, 256))):
        reference_path = case / f"output_{i}.bin"
        if sha256(reference_path) != record["outputs"][i]["sha256"]:
            raise ValueError("Reference changed after preparation")
        value = read_output(raw / f"output_{i}.bin", shape)
        expected = read_output(reference_path, shape)
        delta = np.abs(value.astype(np.float64) - expected)
        mismatches = int(np.count_nonzero(delta > 3e-4 + 3e-4 * np.abs(expected)))
        comparisons.append(
            dict(output=i, mismatches=mismatches, max_abs_error=float(delta.max()))
        )
        if source == "native" and mismatches:
            raise ValueError(
                f"Native output {i} differs from this case reference: {mismatches} values"
            )
        outputs.append(value)
    detections = decode(outputs[0][0], outputs[1][0], threshold)
    images = {}
    for name in CAMERAS:
        path = frame["cameras"][name]["image"]
        if sha256(path) != record["image_sha256"][name]:
            raise ValueError(f"Image no longer matches prepared inputs: {name}")
        image = cv2.imread(path)
        if image is None or image.shape != (900, 1600, 3):
            raise ValueError(f"Invalid original image: {name}")
        images[name] = image[..., ::-1]
    fig = plt.figure(figsize=(13, 15), facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[1, 1, 1.6, 1.6],
        left=0.05,
        right=0.95,
        bottom=0.10,
        top=0.91,
        hspace=0.18,
        wspace=0.04,
    )
    input_label = (
        "Recorded camera inputs"
        if record["input_kind"] == "recorded"
        else "SYNTHETIC TEST INPUTS — NOT A ROAD-SCENE RESULT"
    )
    source_label = (
        "Native ExecuTorch output" if source == "native" else "PyTorch reference output"
    )
    fig.suptitle(
        "BEVFormer-Tiny perception output",
        x=0.05,
        ha="left",
        fontsize=22,
        weight="bold",
        y=0.975,
    )
    fig.text(0.05, 0.945, f"{input_label}  |  {source_label}", fontsize=12)
    for index, name in enumerate(DISPLAY):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        ax.imshow(images[name])
        ax.axis("off")
        ax.set_title(
            name.replace("CAM_", "").replace("_", " ").title(), fontsize=12, pad=4
        )
        matrix = frame["cameras"][name]["lidar2img"]
        for box in detections:
            points = corners(box)
            for i, j in EDGES:
                edge = project_edge(points[i], points[j], matrix)
                if edge is not None:
                    ax.plot(
                        edge[:, 0],
                        edge[:, 1],
                        color=COLORS[box["label"]],
                        lw=0.9,
                        clip_on=True,
                    )
        ax.set_xlim(0, 1600)
        ax.set_ylim(900, 0)
    ax = fig.add_subplot(grid[2:, :])
    ax.set_aspect("equal")
    ax.set_title("Detected objects in bird’s-eye view", fontsize=16, pad=12)
    # Display coordinate convention: LiDAR x right, y forward (up); no ego-frame relabeling.
    for box in detections:
        points = corners(box)[:4, :2]
        ax.add_patch(
            Polygon(
                points, closed=True, fill=False, lw=1.2, edgecolor=COLORS[box["label"]]
            )
        )
    ax.scatter([0], [0], s=95, c="black", marker="^", label="LiDAR origin")
    ax.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="LiDAR x (m)",
        ylabel="LiDAR y (m)",
    )
    ax.grid(color="#dddddd", linewidth=0.6)
    labels = sorted({b["label"] for b in detections})
    handles = [
        Line2D([0], [0], color=COLORS[i], lw=2, label=CLASSES[i].replace("_", " "))
        for i in labels
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=False,
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.95,
            "No detections above the display threshold",
            transform=ax.transAxes,
            ha="center",
        )
    fig.text(
        0.05,
        0.066,
        f"Predictions, not ground truth. Score ≥ {threshold:g}; {len(detections)} boxes before plot-range clipping.",
        fontsize=11,
    )
    fig.text(
        0.05,
        0.046,
        f'Sample: {frame["sample_token"]}  |  History: {"enabled" if record["history"] else "scene reset"}',
        fontsize=9,
    )
    fig.text(
        0.05,
        0.026,
        "Camera overlays and BEV boxes use the same saved model outputs and calibration.",
        fontsize=10,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    report = dict(
        input_kind=record["input_kind"],
        source=source,
        sample_token=frame["sample_token"],
        threshold=threshold,
        detections=len(detections),
        comparison=comparisons,
        raw_output_sha256={
            f"output_{i}.bin": sha256(raw / f"output_{i}.bin") for i in range(3)
        },
        frame_metadata_sha256=sha256(case / "frame.json"),
        figure_sha256=sha256(output),
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--raw-output-dir", type=Path, required=True)
    p.add_argument("--source", choices=["native", "pytorch"], required=True)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--extent", type=float, default=51.2)
    p.add_argument("--output", type=Path, required=True, help="New PNG file")
    args = p.parse_args()
    if args.extent <= 0 or not np.isfinite(args.extent):
        p.error("--extent must be finite and positive")
    if args.output.suffix.lower() != ".png":
        p.error("Use a .png output")
    print(
        json.dumps(
            render(
                args.case_dir,
                args.raw_output_dir,
                args.output,
                args.source,
                args.score_threshold,
                args.extent,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
