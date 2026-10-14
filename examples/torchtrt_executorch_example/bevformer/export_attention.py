"""Export and validate a real pretrained attention block, not full BEVFormer.

Default: allow TensorRT to claim all supported operators. --cuda-grid is a
deliberate partitioning experiment, not evidence of a missing TRT converter.
Each export requires a fresh directory, to keep .pte/.ptd pairs together.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_tensorrt
from attention import example_inputs, load_attention
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir._serialize._program import deserialize_pte_binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=2500)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cuda-grid", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        load_attention(
            args.root / "checkpoints/bevformer_tiny_epoch_24.pth", layer=args.layer
        )
        .cuda()
        .eval()
    )
    inputs = tuple(x.cuda() for x in example_inputs(args.queries))
    with torch.no_grad():
        exported = torch.export.export(model, inputs, strict=True)
        settings = {
            "min_block_size": 1,
            "truncate_double": True,
            "disable_tf32": True,
            "enabled_precisions": {torch.float32},
        }
        if args.cuda_grid:
            settings["torch_executed_ops"] = {
                torch.ops.aten.grid_sampler.default,
                torch.ops.aten.grid_sampler_2d.default,
            }
        compiled = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, **settings
        )
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=3e-4, atol=3e-4
        )
        path = args.output_dir / "attention.pte"
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
        delegates = [d.id for plan in serialized.execution_plan for d in plan.delegates]
        if "TensorRTBackend" not in delegates:
            raise AssertionError(f"No TRT delegate: {delegates}")
        if args.cuda_grid and "CudaBackend" not in delegates:
            raise AssertionError(
                f"Forced CUDA grid did not produce a CUDA delegate: {delegates}"
            )
        print("Delegates:", delegates, flush=True)

        import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        program = Runtime.get().load_program(path)
        method = program.load_method("forward")
        results = []
        for seed in (7, 19, 51):
            cpu_inputs = example_inputs(args.queries, seed=seed)
            expected = model(*(x.cuda() for x in cpu_inputs)).cpu()
            actual = method.execute(cpu_inputs)[0].cpu().clone()
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
            results.append(
                {"seed": seed, "max_abs_error": (actual - expected).abs().max().item()}
            )
            print("PASS: complete ExecuTorch output", results[-1], flush=True)
        # A separate reference for the existing native runner's all-ones input.
        ones = tuple(torch.ones_like(x) for x in inputs)
        native_expected = model(*ones).cpu()
        report = {
            "scope": "one pretrained BEVFormer-Tiny spatial-cross-attention block",
            "layer": args.layer,
            "queries": args.queries,
            "forced_cuda_grid": args.cuda_grid,
            "delegates": delegates,
            "torch": torch.__version__,
            "torch_tensorrt": torch_tensorrt.__version__,
            "input_shapes": [list(x.shape) for x in inputs],
            "output_shape": list(native_expected.shape),
            "python_runtime_checks": results,
            "native_ones_first_eight": native_expected.flatten()[:8].tolist(),
            "native_ones_reference_only": True,
        }
        (args.output_dir / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        torch.save(
            {
                "inputs": tuple(x.cpu() for x in inputs),
                "expected": model(*inputs).cpu(),
                "native_ones_expected": native_expected,
            },
            args.output_dir / "reference.pt",
        )
    print("PASS: component export and full-output Python runtime comparison:", path)


if __name__ == "__main__":
    main()
