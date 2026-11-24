# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare synthetic BEVFormer correctness fixtures; Python is offline only.

Three independent cases reproduce the export checks. Three sequence cases feed
the eager model's previous BEV into its next call. The native runner must instead
feed its own preceding output for the sequence comparison. Neither is a real
driving scene or a detector-accuracy measurement.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def save_tensor(tensor, path):
    assert tensor.dtype == torch.float32
    raw = tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
    path.write_bytes(raw)
    return {
        "shape": list(tensor.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", default=Path(__file__).resolve().parent, type=Path
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--scenes",
        type=int,
        default=0,
        help="Generate varied sequences instead of the six baseline cases",
    )
    parser.add_argument("--frames-per-scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.scenes < 0 or args.frames_per_scene < 1:
        parser.error("scenes must be nonnegative and frames-per-scene must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_dir))
    from model import example_inputs, load_model

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint).cuda().eval()
    report = {
        "torch": torch.__version__,
        "scope": "synthetic FP32 tensor-core fixtures",
        "cases": {},
    }
    if args.scenes:
        manifest = []
        with torch.no_grad():
            for scene in range(args.scenes):
                previous = None
                heading = 0.0
                for frame in range(args.frames_per_scene):
                    seed = args.seed + scene * args.frames_per_scene + frame
                    name = f"scene_{scene:02d}_frame_{frame:03d}"
                    case = args.output_dir / name
                    case.mkdir()
                    history = frame > 0
                    inputs = list(example_inputs(seed=seed, history=history))
                    dx, dy, yaw = (
                        (0.0, 0.0, 0.0)
                        if not history
                        else ((0.15, -0.08, 0.0), (-0.3, 0.2, -3.0), (0.7, 0.05, 5.0))[
                            (frame - 1) % 3
                        ]
                    )
                    heading += yaw * torch.pi / 180
                    inputs[2][0, 0], inputs[2][0, 1] = dx, dy
                    inputs[2][0, -2], inputs[2][0, -1] = heading, yaw
                    if previous is not None:
                        inputs[3] = previous.cpu()
                    outputs = model(*(tensor.cuda() for tensor in inputs))
                    previous = outputs[2].detach().clone()
                    entry = {
                        "seed": seed,
                        "scene": scene,
                        "frame": frame,
                        "history": history,
                        "motion": [dx, dy, heading, yaw],
                        "inputs": [],
                        "outputs": [],
                    }
                    for kind, tensors in (("input", inputs), ("output", outputs)):
                        for i, tensor in enumerate(tensors):
                            stem = f"{kind}_{i}"
                            entry[kind + "s"].append(
                                save_tensor(tensor, case / f"{stem}.bin")
                            )
                            (case / f"{stem}.shape").write_text(
                                " ".join(map(str, tensor.shape)) + "\n"
                            )
                    report["cases"][name] = entry
                    manifest.append(f"{'continue' if history else 'reset'} {name}")
                    print("SAVED", name, "motion", entry["motion"], flush=True)
        report["scope"] = (
            "varied synthetic sequences with recurrent eager state; not recorded driving data"
        )
        report["scenes"], report["frames_per_scene"] = (
            args.scenes,
            args.frames_per_scene,
        )
        (args.output_dir / "sequence.txt").write_text("\n".join(manifest) + "\n")
        (args.output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return
    with torch.no_grad():
        for sequence in (False, True):
            previous = None
            for index, (seed, history) in enumerate(
                ((7, False), (19, True), (51, True))
            ):
                name = f"sequence_{index}" if sequence else f"seed_{seed}"
                case = args.output_dir / name
                case.mkdir()
                inputs = list(example_inputs(seed=seed, history=history))
                if sequence and previous is not None:
                    inputs[3] = previous.cpu()
                outputs = model(*(t.cuda() for t in inputs))
                previous = outputs[2].detach().clone() if sequence else None
                entry = {"seed": seed, "history": history, "inputs": [], "outputs": []}
                for kind, tensors in (("input", inputs), ("output", outputs)):
                    for i, tensor in enumerate(tensors):
                        stem = f"{kind}_{i}"
                        entry[kind + "s"].append(
                            save_tensor(tensor, case / f"{stem}.bin")
                        )
                        (case / f"{stem}.shape").write_text(
                            " ".join(map(str, tensor.shape)) + "\n"
                        )
                report["cases"][name] = entry
                print("SAVED", name, flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
