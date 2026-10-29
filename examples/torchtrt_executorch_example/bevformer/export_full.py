# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible full BEVFormer-Tiny mixed-delegate export for the DRIVE demo.

Uses the maintained BEV model adapter without editing it. Geometry goes to AOTI;
division-rounding preservation is explicit, because it affects visibility masks.
The branch includes the generic ExecuTorch reshape-copy normalization.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import torch_tensorrt
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir._serialize._program import deserialize_pte_binary


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("Output count changed")
    results = {}
    for name, got, want in zip(("logits", "boxes", "next_bev"), actual, expected):
        got, want = got.detach().cpu(), want.detach().cpu()
        if got.shape != want.shape:
            raise AssertionError((name, got.shape, want.shape))
        results[name] = {
            "shape": list(got.shape),
            "elements": got.numel(),
            "max_abs_error": float((got - want).abs().max()),
            "mismatches": int((~torch.isclose(got, want, rtol=3e-4, atol=3e-4)).sum()),
            "finite": bool(torch.isfinite(got).all() and torch.isfinite(want).all()),
        }
    if any(r["mismatches"] or not r["finite"] for r in results.values()):
        raise AssertionError(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", default=Path(__file__).resolve().parent, type=Path
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_dir.resolve()))
    from model import Geometry, example_inputs, load_model

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "scope": "full pretrained BEVFormer-Tiny tensor computation; synthetic preprocessed inputs",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torch-tensorrt", "tensorrt", "executorch")
        },
        "torch_tensorrt_import": torch_tensorrt.__file__,
        "checkpoint_sha256": checksum(args.checkpoint),
        "source_sha256": {p.name: checksum(p) for p in args.source_dir.glob("*.py")},
        "settings": {
            "geometry": "CudaBackend/AOTInductor",
            "eager_numerics.division_rounding": True,
            "precision": "FP32",
            "disable_tf32": True,
            "rtol": 3e-4,
            "atol": 3e-4,
        },
    }
    with torch.no_grad():
        model = load_model(args.checkpoint).cuda().eval()
        inputs = tuple(t.cuda() for t in example_inputs(seed=19, history=True))
        exported = torch.export.export(model, inputs, strict=True)
        print("STRICT EXPORT PASS", flush=True)
        compiled = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=inputs,
            min_block_size=1,
            truncate_double=True,
            disable_tf32=True,
            enabled_precisions={torch.float32},
            workspace_size=2 << 30,
            pass_through_build_failures=True,
            torch_executed_modules={f"{Geometry.__module__}.{Geometry.__qualname__}"},
        )
        report["trt_plus_pytorch"] = compare(compiled(*inputs), model(*inputs))
        path = args.output_dir / "model.pte"
        with inductor_config.patch({"eager_numerics.division_rounding": True}):
            torch_tensorrt.save(
                compiled,
                str(path),
                output_format="executorch",
                arg_inputs=inputs,
                retrace=False,
                partitioners=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )
        data = args.output_dir / "aoti_cuda_blob.ptd"
        if not data.is_file():
            raise FileNotFoundError("Expected external AOTI data alongside model.pte")
        program = deserialize_pte_binary(path.read_bytes()).program
        report["delegates"] = [
            d.id for plan in program.execution_plan for d in plan.delegates
        ]
        if not {"TensorRTBackend", "CudaBackend"} <= set(report["delegates"]):
            raise AssertionError(report["delegates"])
        import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        method = Runtime.get().load_program(path, data_path=data).load_method("forward")
        report["python_cases"] = []
        for seed, history in ((7, False), (19, True), (51, True)):
            data_inputs = example_inputs(seed=seed, history=history)
            result = compare(
                method.execute(data_inputs), model(*(t.cuda() for t in data_inputs))
            )
            report["python_cases"].append(
                {"seed": seed, "history": history, "outputs": result}
            )
            print("PYTHON .PTE PASS", seed, result, flush=True)
        report["artifacts"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": checksum(p)}
            for p in (path, data)
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(
        "FULL BEVFORMER EXPORT PASS; native validation is a separate step", flush=True
    )


if __name__ == "__main__":
    main()
