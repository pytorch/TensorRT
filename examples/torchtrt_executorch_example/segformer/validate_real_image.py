# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare a real-image fixture, then compare/visualize native SegFormer logits.

Inference in the C++ checker remains a separate, Python-free host process.
This Python tool generates the eager reference and performs offline analysis.
No ground-truth labels are provided: agreement is not segmentation accuracy.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from export_segformer import compare, load_model, write_tensor
from fetch_checkpoint import MODEL_ID, REVISION, WEIGHTS_SHA256, checksum
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Cityscapes train-ID order; visualization metadata, not ground-truth labels.
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
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
PALETTE = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_logits(path, shape):
    if len(shape) != 4 or shape[:2] != [1, 19] or min(shape) <= 0:
        raise ValueError(f"Expected positive 1x19xHxW logits, got {shape}")
    expected_bytes = int(np.prod(shape)) * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Wrong byte count: {path}")
    logits = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))
    if not torch.isfinite(logits).all():
        raise ValueError(f"Nonfinite logits: {path}")
    return logits


def prediction(logits, size):
    # Resize LOGITS before argmax; resizing class IDs bilinearly is incorrect.
    return F.interpolate(
        logits, size=size, mode="bilinear", align_corners=False
    ).argmax(1)[0]


def prepare(args):
    from transformers import SegformerImageProcessor

    artifact = json.loads((args.bundle / "report.json").read_text())
    if (artifact["model_id"], artifact["revision"], artifact["checkpoint_sha256"]) != (
        MODEL_ID,
        REVISION,
        WEIGHTS_SHA256,
    ):
        raise ValueError("Artifact and reference checkpoint differ")
    model_sha = checksum(args.bundle / "model.pte")
    if model_sha != artifact["artifacts"]["model.pte"]["sha256"]:
        raise ValueError("PTE artifact hash differs from export report")
    shape = artifact["input_shape"]
    if len(shape) != 4 or shape[:2] != [1, 3] or min(shape) <= 0:
        raise ValueError(f"Expected positive 1x3xHxW input: {shape}")
    height, width = shape[2:]
    provenance = json.loads(args.provenance.read_text())
    if checksum(args.image) != provenance["sha256"]:
        raise ValueError("Source image hash differs from provenance")
    config = json.loads((args.checkpoint / "config.json").read_text())
    if tuple(config["id2label"][str(i)] for i in range(19)) != LABELS:
        raise ValueError("Unexpected class ordering")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    processor = SegformerImageProcessor.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        size={"height": height, "width": width},
    )
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    if list(inputs.shape) != shape or inputs.dtype != torch.float32:
        raise ValueError("Unexpected preprocessed input")
    # Verify the explicit protocol instead of trusting changing processor defaults.
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(resized, dtype=np.float32)).permute(2, 0, 1)[
        None
    ]
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    torch.testing.assert_close(
        inputs, (pixels / 255.0 - mean) / std, atol=1e-6, rtol=1e-6
    )
    args.output.mkdir(parents=True, exist_ok=False)
    model, tensors = load_model(args.checkpoint)
    model = model.cuda()
    with torch.inference_mode():
        reference = model(inputs.cuda())[0].cpu()
    if not torch.isfinite(reference).all():
        raise ValueError("Nonfinite eager reference")
    write_tensor(args.output / "input_0", inputs)
    write_tensor(args.output / "output_0", reference)
    resized.save(args.output / "input_rgb.png")
    report = {
        "scope": "Single public image; numerical validation, not task accuracy",
        "source_image": provenance,
        "original_rgb_size": list(image.size),
        "preprocessing": {
            "resize": "Entire frame to export height/width; PIL bilinear, no crop/pad",
            "input_shape": shape,
            "channel_order": "RGB NCHW",
            "dtype": "float32",
            "rescale": "1/255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "processor_crosscheck": "PASS",
            "aspect_ratio_preserved": image.width * height == image.height * width,
        },
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "checkpoint_tensors": tensors,
        "model_pte_sha256": model_sha,
        "output_shape": list(reference.shape),
        "tf32": False,
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "tensorrt",
                "executorch",
                "pillow",
                "numpy",
            )
        },
        "source_sha256": checksum(Path(__file__)),
        "fixture_sha256": {
            name: checksum(args.output / name)
            for name in (
                "input_0.bin",
                "input_0.shape",
                "output_0.bin",
                "output_0.shape",
                "input_rgb.png",
            )
        },
    }
    save_json(args.output / "fixture.json", report)
    print("REAL-IMAGE REFERENCE PASS", tuple(reference.shape), flush=True)


def visualize(image, labels, output, provenance):
    colors = Image.fromarray(PALETTE[labels.numpy()])
    overlay = Image.blend(image, colors, 0.45)
    colors.save(output / "native_prediction.png")
    overlay.save(output / "native_overlay.png")
    Image.fromarray(labels.numpy().astype(np.uint8)).save(
        output / "native_class_ids.png"
    )
    # A deterministic data plot of native logits, not a generated illustration.
    canvas = Image.new("RGB", (1576, 742), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=16)
    draw.text(
        (12, 12),
        "SegFormer-B0 on DRIVE Thor | native C++ prediction",
        fill="#102330",
        font=font,
    )
    for index, (panel, title) in enumerate(
        (
            (image, "Input (resized RGB)"),
            (colors, "Native predicted classes"),
            (overlay, "Prediction overlay"),
        )
    ):
        x = 12 + index * 524
        draw.text((x, 49), title, fill="#102330", font=font)
        canvas.paste(
            panel.resize(
                (512, 512),
                Image.Resampling.NEAREST if index == 1 else Image.Resampling.BILINEAR,
            ),
            (x, 76),
        )
    for index, name in enumerate(LABELS):
        x, y = 12 + (index % 7) * 222, 604 + (index // 7) * 27
        draw.rectangle((x, y + 2, x + 17, y + 18), fill=tuple(PALETTE[index]))
        draw.text((x + 24, y), name, fill="#102330", font=small)
    credit = f"Image: {provenance['author']} ({provenance['license']}). Model predictions, not ground truth."
    draw.text((12, 699), credit, fill="#344955", font=small)
    canvas.save(output / "comparison.png")


def analyze(args):
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    for name, digest in fixture["fixture_sha256"].items():
        if checksum(args.fixture / name) != digest:
            raise ValueError(f"Fixture hash mismatch: {name}")
    shape = fixture["output_shape"]
    reference = read_logits(args.fixture / "output_0.bin", shape)
    actual = read_logits(args.native / "output_0.bin", shape)
    result = compare(actual, reference)
    size = fixture["preprocessing"]["input_shape"][2:]
    got, want = prediction(actual, size), prediction(reference, size)
    result["postprocessed_pixels"] = got.numel()
    result["postprocessed_argmax_disagreements"] = int((got != want).sum())
    result["postprocessing"] = (
        "CPU bilinear logit resize to input size, align_corners=False, then argmax"
    )
    result["accuracy_vs_ground_truth"] = None
    result["atol"] = result["rtol"] = 3e-4
    result["fixture_json_sha256"] = checksum(args.fixture / "fixture.json")
    result["model_pte_sha256"] = fixture["model_pte_sha256"]
    result["native_output_sha256"] = checksum(args.native / "output_0.bin")
    result["source_sha256"] = checksum(Path(__file__))
    result["predicted_pixels_per_class"] = {
        name: int((got == i).sum()) for i, name in enumerate(LABELS)
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.fixture / "input_rgb.png") as image:
        visualize(image.convert("RGB"), got, args.output, fixture["source_image"])
    Image.fromarray(((got != want).numpy() * 255).astype(np.uint8)).save(
        args.output / "disagreements.png"
    )
    save_json(args.output / "comparison.json", result)
    print("NATIVE REAL-IMAGE COMPARISON PASS", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    reference = modes.add_parser("prepare")
    for argument in ("image", "provenance", "checkpoint", "bundle", "output"):
        reference.add_argument("--" + argument, type=Path, required=True)
    analysis = modes.add_parser("analyze")
    for argument in ("fixture", "native", "output"):
        analysis.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    (prepare if args.mode == "prepare" else analyze)(args)


if __name__ == "__main__":
    main()
