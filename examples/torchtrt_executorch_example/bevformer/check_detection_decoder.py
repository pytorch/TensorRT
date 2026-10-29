# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check C++ decoding against actual pinned upstream functions on identical raw tensors.

Extract just the numerical functions via AST so the offline oracle does not need
MMCV registries/CUDA extensions. Their bodies are executed unchanged. The final
bottom-center adjustment is from BEVFormerHead.get_bboxes. This is not mAP/NDS.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch


def upstream_oracle(root):
    util = root / "projects/mmdet3d_plugin/core/bbox/util.py"
    coder = root / "projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py"
    expected = {
        util: "d24c61941209fa7a7be18d3b4c628dc06006f692fde0e7863fb54bca50282beb",
        coder: "f9cd5871541d3074b29bcdb0a4f458cd2e1164fcefe3379890e6d59bb1ea66bb",
    }
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Upstream reference changed: {path}")
    function = next(
        n
        for n in ast.parse(util.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "denormalize_bbox"
    )
    cls = next(
        n
        for n in ast.parse(coder.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "NMSFreeCoder"
    )
    decode = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "decode_single"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[function, decode], type_ignores=[]), str(coder), "exec"
        ),
        namespace,
    )
    settings = SimpleNamespace(
        max_num=300,
        num_classes=10,
        score_threshold=None,
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        post_center_range=[-61.2, -61.2, -10, 61.2, 61.2, 10],
    )
    return lambda logits, boxes: namespace["decode_single"](settings, logits, boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also check inclusive range edges and empty results",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    decode = upstream_oracle(args.upstream)
    cases = sorted(args.raw_dir.glob("frame_*")) or [args.raw_dir]
    if args.include_synthetic:
        for name in ("range_edges", "empty"):
            case = args.output_dir / ("inputs_" + name)
            case.mkdir()
            logits = np.linspace(-4, 4, 9000, dtype=np.float32).reshape(900, 10)
            boxes = np.zeros((900, 10), dtype=np.float32)
            boxes[:, 7] = 1  # cos(yaw); log dimensions zero -> unit sizes.
            if name == "empty":
                boxes[:, 0] = 70
            else:
                boxes[:, 0] = np.resize(
                    np.array([-61.3, -61.2, 0, 61.2, 61.3], dtype=np.float32), 900
                )
                boxes[:, 4] = np.resize(
                    np.array([-10.1, -10, 0, 10, 10.1, 0, 0], dtype=np.float32), 900
                )
                boxes[:, 5] = np.log(
                    16
                )  # Check range filtering happens BEFORE bottom-z adjustment.
            logits.tofile(case / "output_0.bin")
            boxes.tofile(case / "output_1.bin")
            cases.append(case)
    report = []
    for case in cases:
        logits = torch.from_numpy(
            np.fromfile(case / "output_0.bin", dtype="<f4").reshape(900, 10)
        )
        boxes = torch.from_numpy(
            np.fromfile(case / "output_1.bin", dtype="<f4").reshape(900, 10)
        )
        result = decode(logits, boxes)
        result["bboxes"][:, 2] -= result["bboxes"][:, 5] * 0.5
        path = args.output_dir / f"{case.name}.tsv"
        subprocess.run([str(args.decoder.resolve()), str(case), str(path)], check=True)
        with path.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        # Strict rank comparison for these fixtures. Tied scores can have a
        # different legal top-k order across implementations; report, don't hide.
        actual_scores = torch.tensor([float(r["score"]) for r in rows])
        actual_labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.int64)
        actual_boxes = torch.tensor(
            [
                [
                    float(r[k])
                    for k in ("cx", "cy", "z_bottom", "w", "l", "h", "yaw", "vx", "vy")
                ]
                for r in rows
            ]
        ).reshape(-1, 9)
        torch.testing.assert_close(actual_labels, result["labels"], rtol=0, atol=0)
        torch.testing.assert_close(
            actual_scores, result["scores"], rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(actual_boxes, result["bboxes"], rtol=3e-6, atol=3e-6)
        report.append(
            {
                "case": case.name,
                "detections": len(rows),
                "max_box_error": (
                    (actual_boxes - result["bboxes"]).abs().max().item()
                    if rows
                    else 0.0
                ),
            }
        )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: native decoding agrees with pinned upstream on", len(report), "cases")


if __name__ == "__main__":
    main()
