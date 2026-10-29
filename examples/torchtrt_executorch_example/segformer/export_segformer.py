# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Strict pretrained SegFormer-B0 -> TensorRT -> ExecuTorch .pte evaluation.

Inputs are already resized and ImageNet-normalized FP32 NCHW RGB tensors.
Outputs are raw quarter-resolution logits, not a Cityscapes accuracy score.
Uses the maintained Transformers implementation, not a rewritten architecture.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from fetch_checkpoint import MODEL_ID, REVISION, WEIGHTS_SHA256, checksum


class SegmentationLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return (self.model(pixel_values=pixel_values, return_dict=False)[0],)


def load_model(checkpoint):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if checksum(checkpoint / "pytorch_model.bin") != WEIGHTS_SHA256:
        raise ValueError("Unexpected checkpoint")
    config = SegformerConfig.from_json_file(checkpoint / "config.json")
    if config.hidden_sizes != [32, 64, 160, 256] or config.num_labels != 19:
        raise ValueError("Expected Cityscapes SegFormer-B0 architecture")
    model = SegformerForSemanticSegmentation(config)
    state = torch.load(
        checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    print("STRICT CHECKPOINT PASS", len(state), "tensors", flush=True)
    return SegmentationLogits(model).eval(), len(state)


def fixture(seed, height, width):
    # Synthetic RGB only: independent seeds exercise numerical consistency,
    # not semantic accuracy or representative driving content.
    image = torch.rand(
        (1, 3, height, width), generator=torch.Generator().manual_seed(seed)
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def write_tensor(path, tensor):
    value = tensor.detach().cpu().float().contiguous()
    value.numpy().tofile(path.with_suffix(".bin"))
    path.with_suffix(".shape").write_text(" ".join(map(str, value.shape)) + "\n")


def single_output(value):
    # A compiled one-output GraphModule may return a Tensor instead of the
    # wrapper's one-element tuple. Never index a Tensor and drop its batch axis.
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    raise TypeError(f"Expected a single tensor output, got {type(value)}")


def compare(actual, reference):
    got, want = actual.detach().cpu(), reference.detach().cpu()
    if got.dtype != want.dtype or got.shape != want.shape:
        raise AssertionError(
            f"Output shape/dtype mismatch: {got.shape}/{got.dtype} vs {want.shape}/{want.dtype}"
        )
    result = {
        "shape": list(got.shape),
        "elements": got.numel(),
        "max_abs_error": float((got - want).abs().max()),
        "mismatches": int((~torch.isclose(got, want, atol=3e-4, rtol=3e-4)).sum()),
        "finite": bool(torch.isfinite(got).all() and torch.isfinite(want).all()),
        "argmax_disagreements": int((got.argmax(1) != want.argmax(1)).sum()),
    }
    if not result["finite"] or result["mismatches"]:
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--eager-only", action="store_true")
    args = parser.parse_args()
    if min(args.height, args.width) < 32 or args.height % 32 or args.width % 32:
        parser.error("Input dimensions must be positive multiples of 32")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, count = load_model(args.checkpoint)
    model = model.cuda()
    report = {
        "scope": "Pretrained SegFormer-B0 Cityscapes tensor inference; synthetic inputs, no mIoU",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "checkpoint_tensors": count,
        "input_shape": [1, 3, args.height, args.width],
        "settings": {"precision": "FP32", "tf32": False, "atol": 3e-4, "rtol": 3e-4},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "tensorrt", "executorch")
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    with torch.no_grad():
        cases = []
        for seed in (7, 19, 51):
            image = fixture(seed, args.height, args.width).cuda()
            reference = model(image)[0].cpu()
            if not torch.isfinite(reference).all():
                raise ValueError("Nonfinite eager reference")
            case = args.output / f"seed_{seed}"
            case.mkdir()
            write_tensor(case / "input_0", image)
            write_tensor(case / "output_0", reference)
            cases.append((seed, image, reference))
            print("EAGER PASS", seed, tuple(reference.shape), flush=True)
        if args.eager_only:
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            return
        import torch_tensorrt
        from executorch.exir._serialize._program import deserialize_pte_binary

        inputs = (cases[0][1],)
        exported = torch.export.export(model, inputs, strict=True)
        print("STRICT EXPORT PASS", flush=True)
        compiled = torch_tensorrt.dynamo.compile(
            # Keep the GraphModule needed by save(output_format="executorch").
            # This source revision's require_full_compilation fast path returns
            # a bare engine module that save() does not accept.
            exported,
            arg_inputs=inputs,
            min_block_size=1,
            disable_tf32=True,
            enabled_precisions={torch.float32},
            workspace_size=2 << 30,
            pass_through_build_failures=True,
        )
        if not isinstance(compiled, torch.fx.GraphModule):
            raise TypeError("Expected the graph-preserving compile path")
        fallback = [
            name for name, _ in compiled.named_children() if "_run_on_gpu" in name
        ]
        if fallback:
            raise AssertionError(f"Unexpected PyTorch fallback regions: {fallback}")
        report["compiled_graph"] = str(compiled.graph)
        for seed, image, reference in cases:
            report["cases"].append(
                {
                    "seed": seed,
                    "trt_python": compare(single_output(compiled(image)), reference),
                }
            )
        path = args.output / "model.pte"
        torch_tensorrt.save(
            compiled,
            str(path),
            output_format="executorch",
            arg_inputs=inputs,
            retrace=False,
        )
        program = deserialize_pte_binary(path.read_bytes()).program
        report["delegates"] = [
            d.id for plan in program.execution_plan for d in plan.delegates
        ]
        if not report["delegates"] or set(report["delegates"]) != {"TensorRTBackend"}:
            raise AssertionError(report["delegates"])
        import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        loaded = Runtime.get().load_program(path).load_method("forward")
        for entry, (_, image, reference) in zip(report["cases"], cases):
            entry["python_pte"] = compare(loaded.execute((image.cpu(),))[0], reference)
            print("PYTHON PTE PASS", entry, flush=True)
        report["artifacts"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": checksum(p)}
            for p in args.output.iterdir()
            if p.suffix in (".pte", ".ptd")
        }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "EXPORT_PASS").write_text(
        "Native host validation is a separate step.\n"
    )
    print("EXPORT PASS; native validation remains", flush=True)


if __name__ == "__main__":
    main()
