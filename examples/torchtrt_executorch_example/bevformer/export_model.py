# SPDX-License-Identifier: Apache-2.0
"""Export the pretrained Tiny tensor core; validate every output numerically.

Preprocessed synthetic image tensors are used. This does not establish detector
accuracy. Keep the entire output directory, including any external .ptd files.
"""

import argparse
import contextlib
import json
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import torch_tensorrt
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir._serialize._program import deserialize_pte_binary
from model import Geometry, example_inputs, load_model


def compare(actual, expected, stage):
    if len(actual) != len(expected):
        raise AssertionError(f"{stage}: unexpected number of outputs")
    results = []
    for name, output, wanted in zip(
        ("class_logits", "boxes", "next_bev"), actual, expected
    ):
        output, wanted = output.cpu(), wanted.cpu()
        error = (output - wanted).abs().max().item()
        print(f"{stage}: {name} max_abs_error={error}", flush=True)
        torch.testing.assert_close(output, wanted, rtol=3e-4, atol=3e-4)
        results.append({"output": name, "max_abs_error": error})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-grid", action="store_true")
    parser.add_argument(
        "--cuda-geometry",
        action="store_true",
        help="Keep projection and previous-BEV rotation outside TensorRT",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.root / "checkpoints/bevformer_tiny_epoch_24.pth").cuda()
    cpu_inputs = example_inputs(seed=19, history=True)
    inputs = tuple(x.cuda() for x in cpu_inputs)
    with torch.no_grad():
        expected = tuple(y.cpu() for y in model(*inputs))
        exported = torch.export.export(model, inputs, strict=True)
        print(
            "PASS: strict export", len(list(exported.graph.nodes)), "nodes", flush=True
        )
        settings = dict(
            min_block_size=1,
            truncate_double=True,
            disable_tf32=True,
            enabled_precisions={torch.float32},
            workspace_size=2 << 30,
            pass_through_build_failures=True,
        )
        if args.cuda_grid:
            settings["torch_executed_ops"] = {
                torch.ops.aten.grid_sampler.default,
                torch.ops.aten.grid_sampler_2d.default,
            }
        if args.cuda_geometry:
            settings["torch_executed_modules"] = {
                f"{Geometry.__module__}.{Geometry.__qualname__}"
            }
        print("Compiling TensorRT regions", flush=True)
        compiled = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, **settings
        )
        compare(compiled(*inputs), expected, "TensorRT graph")
        print("Saving ExecuTorch program", flush=True)
        path = args.output_dir / "bevformer_tiny.pte"
        geometry_numerics = (
            inductor_config.patch({"eager_numerics.division_rounding": True})
            if args.cuda_geometry
            else contextlib.nullcontext()
        )
        with geometry_numerics:
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
        serialized = deserialize_pte_binary(path.read_bytes()).program
        delegates = [
            delegate.id
            for plan in serialized.execution_plan
            for delegate in plan.delegates
        ]
        if "TensorRTBackend" not in delegates:
            raise AssertionError(f"Expected TRT delegation, got {delegates}")
        if (args.cuda_grid or args.cuda_geometry) and "CudaBackend" not in delegates:
            raise AssertionError(f"Expected forced mixed delegation, got {delegates}")
        print("Delegates:", delegates, flush=True)
        import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        data = args.output_dir / "aoti_cuda_blob.ptd"
        data_args = {"data_path": data} if data.is_file() else {}
        method = Runtime.get().load_program(path, **data_args).load_method("forward")
        results = []
        for seed, history in ((7, False), (19, True), (51, True)):
            samples = example_inputs(seed=seed, history=history)
            wanted = model(*(x.cuda() for x in samples))
            actual = method.execute(samples)
            errors = compare(
                actual, wanted, f"ExecuTorch seed={seed} history={history}"
            )
            results.append({"seed": seed, "history": history, "errors": errors})
        torch.save(
            {"inputs": cpu_inputs, "outputs": expected},
            args.output_dir / "reference.pt",
        )
        # Diagnostic snapshot. Use prepare_native_cases.py for the complete
        # fixture format consumed by native_validation. Little-endian FP32.
        for index, tensor in enumerate(cpu_inputs):
            tensor.contiguous().numpy().tofile(args.output_dir / f"input_{index}.bin")
        for index, tensor in enumerate(expected):
            tensor.contiguous().numpy().tofile(
                args.output_dir / f"expected_{index}.bin"
            )
        report = {
            "scope": "full Tiny tensor core, synthetic preprocessed images",
            "torch": torch.__version__,
            "torch_tensorrt": torch_tensorrt.__version__,
            "forced_cuda_grid": args.cuda_grid,
            "cuda_geometry": args.cuda_geometry,
            "delegates": delegates,
            "input_shapes": [list(t.shape) for t in cpu_inputs],
            "output_shapes": [list(t.shape) for t in expected],
            "checks": results,
        }
        (args.output_dir / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print("PASS:", path, flush=True)


if __name__ == "__main__":
    main()
