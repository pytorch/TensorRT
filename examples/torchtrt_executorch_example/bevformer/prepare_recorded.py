# SPDX-License-Identifier: Apache-2.0
"""Create real-camera fixtures for the existing native validate_pte sequence runner."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from recorded_scene import CAMERAS, load_manifest, motion_bus, preprocess, sha256


def write_tensor(tensor, path):
    value = tensor.detach().cpu().contiguous().numpy().astype("<f4")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite values: {path}")
    value.tofile(path)
    path.with_suffix(".shape").write_text(" ".join(map(str, value.shape)) + "\n")
    return dict(shape=list(value.shape), bytes=path.stat().st_size, sha256=sha256(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Model adapter directory containing model.py",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    import torch

    sys.path.insert(0, str(args.source_dir.resolve()))
    from model import load_model

    if args.device == "cuda" and not torch.cuda.is_available():
        p.error(
            "CUDA unavailable; run preparation in the authoring environment or use --device=cpu"
        )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint).to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema="bevformer-recorded-fixtures-v1",
        input_kind=manifest["input_kind"],
        manifest_sha256=sha256(args.manifest),
        checkpoint_sha256=sha256(args.checkpoint),
        source_sha256={f.name: sha256(f) for f in args.source_dir.glob("*.py")},
        torch=torch.__version__,
        reference_device=args.device,
        frames=[],
    )
    previous, previous_bev = None, None
    lines = []
    with torch.no_grad():
        for index, frame in enumerate(manifest["frames"]):
            images, projection = preprocess(frame)
            bus, history = motion_bus(frame, previous)
            if not history:
                previous_bev = torch.zeros(1, 2500, 256, device=args.device)
            inputs = [
                torch.from_numpy(x).to(args.device) for x in (images, projection, bus)
            ]
            inputs += [
                previous_bev,
                torch.full((1, 1, 1), float(history), device=args.device),
            ]
            outputs = model(*inputs)
            expected_shapes = [(1, 900, 10), (1, 900, 10), (1, 2500, 256)]
            if [tuple(t.shape) for t in outputs] != expected_shapes:
                raise ValueError("Unexpected model output shapes")
            case = args.output_dir / f"frame_{index:03d}"
            case.mkdir()
            record = dict(
                frame=frame,
                input_kind=manifest["input_kind"],
                index=index,
                camera_order=list(CAMERAS),
                history=history,
                image_sha256={n: sha256(frame["cameras"][n]["image"]) for n in CAMERAS},
                inputs=[],
                outputs=[],
            )
            for kind, tensors in [("input", inputs), ("output", outputs)]:
                for i, tensor in enumerate(tensors):
                    record[kind + "s"].append(
                        write_tensor(tensor, case / f"{kind}_{i}.bin")
                    )
            (case / "frame.json").write_text(json.dumps(record, indent=2) + "\n")
            report["frames"].append(
                dict(
                    case=case.name, sample_token=frame["sample_token"], history=history
                )
            )
            lines.append(f'{"continue" if history else "reset"} {case.name}')
            previous, previous_bev = frame, outputs[2].detach().clone()
            print("Prepared", case.name, frame["sample_token"], flush=True)
    (args.output_dir / "sequence.txt").write_text("\n".join(lines) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
