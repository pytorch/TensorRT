# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare paired Cityscapes-val fixtures and score eager/native against labels.

Requires a user-provided, licensed Cityscapes dataset. No automatic download.
Protocol: single fixed-size input, resize logits to ORIGINAL mask resolution.
Not the checkpoint's official sliding-window or multi-scale evaluation.
"""

import argparse
import importlib.metadata
import json
import random
import re
from pathlib import Path

import numpy as np
from fetch_checkpoint import MODEL_ID, REVISION, WEIGHTS_SHA256, checksum
from PIL import Image
from segmentation_metrics import LABELS, confusion_matrix, encode_mask, summarize


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pairs(root, encoding, limit, seed):
    if limit < 0:
        raise ValueError(
            "limit must be nonnegative; 0 selects all available validation images"
        )
    images = sorted((root / "leftImg8bit/val").glob("*/*_leftImg8bit.png"))
    if not images:
        raise FileNotFoundError(f"No validation images in {root / 'leftImg8bit/val'}")
    if limit > len(images):
        raise ValueError("Requested more images than available")
    selected = (
        images if limit == 0 else sorted(random.Random(seed).sample(images, limit))
    )
    suffix = "labelIds" if encoding == "label_ids" else "labelTrainIds"
    records = []
    for image in selected:
        identifier = image.name.removesuffix("_leftImg8bit.png")
        if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
            raise ValueError(f"Invalid fixture identifier: {identifier}")
        mask = (
            root
            / "gtFine/val"
            / image.parent.name
            / f"{identifier}_gtFine_{suffix}.png"
        )
        if not mask.is_file():
            raise FileNotFoundError(f"Missing matched mask: {mask}")
        records.append((identifier, image, mask))
    if len({row[0] for row in records}) != len(records):
        raise ValueError("Duplicate fixture identifiers")
    return records, len(images)


def prepare(args):
    # Keep Torch out of module-level imports so metric/data checks run on a CPU workstation.
    import torch
    from export_segformer import load_model, write_tensor
    from transformers import SegformerImageProcessor

    records, available = pairs(
        args.data_root, args.mask_encoding, args.limit, args.seed
    )
    report = json.loads((args.bundle / "report.json").read_text())
    model_sha = checksum(args.bundle / "model.pte")
    if (report["model_id"], report["revision"], report["checkpoint_sha256"]) != (
        MODEL_ID,
        REVISION,
        WEIGHTS_SHA256,
    ):
        raise ValueError("Bundle and eager checkpoint provenance differ")
    if model_sha != report["artifacts"]["model.pte"]["sha256"]:
        raise ValueError("Bundle PTE hash mismatch")
    config = json.loads((args.checkpoint / "config.json").read_text())
    if tuple(config["id2label"][str(i)] for i in range(19)) != LABELS:
        raise ValueError("Checkpoint class order differs")
    shape = report["input_shape"]
    if len(shape) != 4 or shape[:2] != [1, 3] or min(shape) <= 0:
        raise ValueError("Expected positive batch-one RGB input shape")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, _ = load_model(args.checkpoint)
    model = model.cuda()
    processor = SegformerImageProcessor.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        size={"height": shape[2], "width": shape[3]},
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "dataset": "Cityscapes",
        "split": "val",
        "data_root": str(args.data_root.resolve()),
        "mask_encoding": args.mask_encoding,
        "available_images": available,
        "selected_images": len(records),
        "selection_seed": args.seed,
        "selection": (
            "All available"
            if args.limit == 0
            else "Seeded sample without replacement, sorted before inference"
        ),
        "input_shape": shape,
        "model_pte_sha256": model_sha,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "protocol": "Single resized frame, FP32/TF32 off; bilinear logits to original mask size before argmax; ignore ID 255",
        "official_benchmark": False,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "tensorrt", "transformers", "pillow", "numpy")
        },
        "source_sha256": checksum(Path(__file__)),
        "cases": [],
    }
    for identifier, image_path, mask_path in records:
        with Image.open(image_path) as original:
            if original.getexif().get(274, 1) != 1:
                raise ValueError(
                    "Unexpected EXIF rotation: image and mask alignment requires review"
                )
            image = original.convert("RGB")
        with Image.open(mask_path) as source_mask:
            truth = encode_mask(np.array(source_mask), args.mask_encoding)
        if truth.shape != (image.height, image.width):
            raise ValueError(f"Image/mask size mismatch: {identifier}")
        if not (truth != 255).any():
            raise ValueError(f"Mask contains no evaluated pixels: {identifier}")
        inputs = processor(images=image, return_tensors="pt")["pixel_values"]
        # Cross-check resize/normalization against the previous real-image protocol.
        resized = image.resize((shape[3], shape[2]), Image.Resampling.BILINEAR)
        pixels = (
            torch.from_numpy(np.array(resized, dtype=np.float32)).permute(2, 0, 1)[None]
            / 255
        )
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        torch.testing.assert_close(inputs, (pixels - mean) / std, atol=1e-6, rtol=1e-6)
        with torch.inference_mode():
            reference = model(inputs.cuda())[0].cpu()
        if not torch.isfinite(reference).all():
            raise ValueError(f"Nonfinite reference: {identifier}")
        case_dir = args.output / identifier
        case_dir.mkdir()
        write_tensor(case_dir / "input_0", inputs)
        write_tensor(case_dir / "output_0", reference)
        Image.fromarray(truth).save(case_dir / "ground_truth.png")
        manifest["cases"].append(
            {
                "id": identifier,
                "image": str(image_path.relative_to(args.data_root)),
                "mask": str(mask_path.relative_to(args.data_root)),
                "image_sha256": checksum(image_path),
                "mask_sha256": checksum(mask_path),
                "mask_shape": list(truth.shape),
                "logits_shape": list(reference.shape),
                "files": {
                    name: checksum(case_dir / name)
                    for name in (
                        "input_0.bin",
                        "input_0.shape",
                        "output_0.bin",
                        "output_0.shape",
                        "ground_truth.png",
                    )
                },
            }
        )
        print("REFERENCE", identifier, flush=True)
    # Write completion manifest last: partial preparation must not look scoreable.
    save_json(args.output / "manifest.json", manifest)
    print(
        "PREPARED",
        len(records),
        "paired fixtures; native inference remains",
        flush=True,
    )


def validate_manifest(manifest):
    cases = manifest.get("cases", [])
    if (
        manifest.get("schema") != 1
        or not cases
        or manifest.get("selected_images") != len(cases)
    ):
        raise ValueError("Incomplete or unsupported dataset manifest")
    identifiers = [case["id"] for case in cases]
    if len(set(identifiers)) != len(cases) or any(
        not re.fullmatch(r"[A-Za-z0-9_]+", x) for x in identifiers
    ):
        raise ValueError("Duplicate or unsafe case identifier")


def score(args):
    import torch
    from validate_real_image import prediction, read_logits

    torch.set_num_threads(2)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    validate_manifest(manifest)
    artifact_hashes = (args.native / "artifact-sha256.txt").read_text().splitlines()
    model_hashes = [
        line.split()[0]
        for line in artifact_hashes
        if line.split()[-1].endswith("/model.pte")
    ]
    if model_hashes != [manifest["model_pte_sha256"]]:
        raise ValueError("Native results use a different or unrecorded PTE")
    expected = {case["id"] for case in manifest["cases"]}
    found = {p.parent.name for p in args.native.glob("*/output_0.bin")}
    if expected != found:
        raise ValueError(
            f"Native cases differ from manifest: missing={expected-found}, extra={found-expected}"
        )
    totals = {key: np.zeros((19, 19), dtype=np.int64) for key in ("eager", "native")}
    outcomes = []
    for case in manifest["cases"]:
        directory = args.fixtures / case["id"]
        for name, digest in case["files"].items():
            if Path(name).name != name or checksum(directory / name) != digest:
                raise ValueError(f"Changed or unsafe fixture file: {case['id']}/{name}")
        with Image.open(directory / "ground_truth.png") as image:
            truth = encode_mask(np.array(image), "train_ids")
        if list(truth.shape) != case["mask_shape"]:
            raise ValueError("Ground-truth shape differs from manifest")
        reference = read_logits(directory / "output_0.bin", case["logits_shape"])
        actual_path = args.native / case["id"] / "output_0.bin"
        actual = read_logits(actual_path, case["logits_shape"])
        maps = (args.native / case["id"] / "loaded-maps.txt").read_text()
        if not maps.strip() or re.search(
            r"libtorch|libc10|libpython|/site-packages/|/venvs/", maps
        ):
            raise ValueError("Missing clean native dependency audit")
        predicted = {
            "eager": prediction(reference, truth.shape).numpy(),
            "native": prediction(actual, truth.shape).numpy(),
        }
        for key in totals:
            totals[key] += confusion_matrix(truth, predicted[key])
        outcomes.append(
            {
                "id": case["id"],
                "native_output_sha256": checksum(actual_path),
                "valid_pixels": int((truth != 255).sum()),
                "ignored_pixels": int((truth == 255).sum()),
                "logit_mismatches": int(
                    (~torch.isclose(actual, reference, atol=3e-4, rtol=3e-4)).sum()
                ),
                "max_abs_error": float((actual - reference).abs().max()),
                "prediction_disagreements_valid_pixels": int(
                    ((predicted["eager"] != predicted["native"]) & (truth != 255)).sum()
                ),
            }
        )
    summaries = {key: summarize(matrix) for key, matrix in totals.items()}
    if summaries["eager"]["mean_iou"] is None:
        raise ValueError("No valid labelled pixels; cannot report accuracy")
    result = {
        "scope": "Cityscapes validation under the recorded fixed-input protocol; not an official benchmark submission",
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "protocol": manifest["protocol"],
        "num_images": len(outcomes),
        "available_images": manifest["available_images"],
        "manifest_sha256": checksum(args.fixtures / "manifest.json"),
        "model_pte_sha256": manifest["model_pte_sha256"],
        "source_sha256": checksum(Path(__file__)),
        "metrics_source_sha256": checksum(
            Path(__file__).with_name("segmentation_metrics.py")
        ),
        "eager": summaries["eager"],
        "native": summaries["native"],
        "native_minus_eager_miou_percentage_points": 100
        * (summaries["native"]["mean_iou"] - summaries["eager"]["mean_iou"]),
        "numerical_tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "numerical_parity_pass": all(row["logit_mismatches"] == 0 for row in outcomes),
        "cases": outcomes,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "accuracy.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "num_images",
                    "native_minus_eager_miou_percentage_points",
                    "numerical_parity_pass",
                )
            }
        ),
        flush=True,
    )
    if not result["numerical_parity_pass"]:
        raise SystemExit(
            "Numerical parity failed; accuracy report retained, do not claim equivalence"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("data-root", "checkpoint", "bundle", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument(
        "--mask-encoding", choices=("label_ids", "train_ids"), default="label_ids"
    )
    prep.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Seeded subset size; 0 means all available validation images",
    )
    prep.add_argument("--seed", type=int, default=0)
    scoring = commands.add_parser("score")
    for name in ("fixtures", "native", "output"):
        scoring.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else score)(args)


if __name__ == "__main__":
    main()
