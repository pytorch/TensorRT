# SPDX-License-Identifier: BSD-3-Clause
"""Export FP32 benchmark programs from checkpoints with caller-owned CPU IO."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import torch_tensorrt
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir._serialize._program import deserialize_pte_binary


class CallerIOMemoryPlanning(MemoryPlanningPass):
    def run(self, graph_module, graph_signature=None):
        # This pinned ExecuTorch version inserts D2H nodes after SpecPropPass
        # but leaves the output node's spec referring to the old GPU values.
        # Refresh that metadata before planning so alloc_graph_output=False
        # excludes the actual CPU outputs, while GPU intermediates stay planned.
        from torch.utils._pytree import tree_map

        for node in graph_module.graph.nodes:
            if node.op == "output":
                node.meta["spec"] = tree_map(
                    lambda value: (
                        value.meta["spec"] if isinstance(value, torch.fx.Node) else None
                    ),
                    node.args[0],
                )
        return super().run(graph_module, graph_signature)


def read_tensor(case, stem):
    shape = tuple(map(int, (case / f"{stem}.shape").read_text().split()))
    value = np.fromfile(case / f"{stem}.bin", dtype="<f4").reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite fixture: {case / stem}")
    return torch.from_numpy(value)


def outputs(value):
    return [value] if isinstance(value, torch.Tensor) else list(value)


def check(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("Output count mismatch")
    result = []
    for got, want in zip(actual, expected):
        got = got.detach().cpu()
        if got.shape != want.shape or got.dtype != want.dtype:
            raise ValueError("Output shape or dtype mismatch")
        bad = int((~torch.isclose(got, want, atol=3e-4, rtol=3e-4)).sum())
        if bad or not torch.isfinite(got).all():
            raise ValueError(f"Output comparison failed: {bad} mismatches")
        result.append(
            {
                "elements": got.numel(),
                "max_abs_error": float((got - want).abs().max()),
                "mismatches": bad,
            }
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--model", required=True, choices=("segformer", "bevformer"))
    p.add_argument(
        "--output", required=True, type=Path, help="New directory for this model"
    )
    a = p.parse_args()
    versions = {
        k: importlib.metadata.version(k)
        for k in ["torch", "torch-tensorrt", "executorch", "tensorrt"]
    }
    expected_versions = {
        "torch": "2.16.0.dev20261009+cu132",
        "torch-tensorrt": "2.16.0.dev0+202fba4",
        "executorch": "1.6.0.dev20261009+cu132",
        "tensorrt": "11.0.0.114",
    }
    if versions != expected_versions:
        raise RuntimeError(f"Unexpected authoring package versions: {versions}")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path(__file__).resolve().parent / "models"
    model_dir = a.bundle / a.model
    settings = dict(
        min_block_size=1,
        disable_tf32=True,
        enabled_precisions={torch.float32},
        workspace_size=2 << 30,
        pass_through_build_failures=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
        timing_cache_path=str(a.output / "timing_cache.bin"),
    )
    options = dict(
        output_format="executorch",
        retrace=False,
        backend_config=ExecutorchBackendConfig(
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=CallerIOMemoryPlanning(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        ),
    )
    if a.model == "bevformer":
        sys.path.insert(0, str(source / "bevformer"))
        from model import Geometry, load_model

        model = load_model(model_dir / "checkpoint.pth")
        settings["torch_executed_modules"] = {
            f"{Geometry.__module__}.{Geometry.__qualname__}"
        }
        settings["truncate_double"] = True
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        options["partitioners"] = [
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ]
        ni, no, delegates = 5, 3, {"TensorRTBackend", "CudaBackend"}
    else:
        sys.path.insert(0, str(source))
        from segformer import load_model

        model = load_model(model_dir / "checkpoint")
        ni, no, delegates = 1, 1, {"TensorRTBackend"}
    report = {
        "model": a.model,
        "precision": "FP32",
        "tf32": False,
        "atol": 3e-4,
        "rtol": 3e-4,
        "caller_owned_cpu_io": True,
        "cases": [],
        "versions": versions,
    }
    with torch.no_grad():
        model = model.eval().cuda()
        # case1 is history-enabled for BEVFormer; both cases are checked below.
        inputs = tuple(
            read_tensor(model_dir / "case1", f"input_{i}").cuda() for i in range(ni)
        )
        exported = torch.export.export(model, inputs, strict=True)
        (a.output / "exported-graph.txt").write_text(str(exported.graph))
        print("STRICT_EXPORT_PASS", flush=True)
        compiled = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, **settings
        )
        (a.output / "compiled-graph.txt").write_text(str(compiled.graph))
        for case in ("case0", "case1"):
            data = tuple(
                read_tensor(model_dir / case, f"input_{i}").cuda() for i in range(ni)
            )
            expected = [read_tensor(model_dir / case, f"output_{i}") for i in range(no)]
            report["cases"].append(
                {
                    "case": case,
                    "eager": check(outputs(model(*data)), expected),
                    "compiled": check(outputs(compiled(*data)), expected),
                }
            )
        print("COMPILED_OUTPUTS_PASS", flush=True)
        import torch._inductor.config as cfg

        with cfg.patch({"eager_numerics.division_rounding": True}):
            torch_tensorrt.save(
                compiled, str(a.output / "model.pte"), arg_inputs=inputs, **options
            )
    program = deserialize_pte_binary((a.output / "model.pte").read_bytes()).program
    if len(program.execution_plan) != 1:
        raise ValueError("Expected one execution plan")
    plan = program.execution_plan[0]
    if {d.id for d in plan.delegates} != delegates:
        raise ValueError(f"Unexpected delegates: {plan.delegates}")
    if len(plan.inputs) != ni or len(plan.outputs) != no:
        raise ValueError("Exported IO count changed")
    for i in plan.inputs + plan.outputs:
        if plan.values[i].val.allocation_info is not None:
            raise ValueError("Export did not preserve caller-owned IO")
    if a.model == "bevformer" and not (a.output / "aoti_cuda_blob.ptd").is_file():
        raise ValueError("Missing external AOTInductor data")
    report["delegates"] = [d.id for d in plan.delegates]
    report["artifacts"] = {}
    for f in a.output.iterdir():
        if f.suffix in (".pte", ".ptd"):
            with f.open("rb") as stream:
                report["artifacts"][f.name] = {
                    "bytes": f.stat().st_size,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
    (a.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "EXPORT_PASS; run native output checks before accepting measurements",
        flush=True,
    )


if __name__ == "__main__":
    main()
