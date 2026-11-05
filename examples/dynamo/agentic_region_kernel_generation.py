# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _agentic_region_kernel_generation_example:

Generate a region kernel with an agent and run it inside TensorRT
=================================================================

During compilation, an agent authors a kernel for the annotated region and
Torch-TensorRT compiles, checks, and measures each proposal before installing
the accepted kernel as a plugin inside the TensorRT engine. Inference needs
neither the agent nor a Python kernel launcher.

Install and log into the selected agent CLI (``claude auth login`` or
``codex login``), and install the selected backend's toolchain: Triton, NVIDIA
cuTile (CUDA Toolkit and driver 13.2+), or CUDA Python's ``cuda-core`` for CUDA
C++. Model calls use your account and receive the annotated graph and tensor
metadata::

    python examples/dynamo/agentic_region_kernel_generation.py
    python examples/dynamo/agentic_region_kernel_generation.py --provider codex --backend cutile
    python examples/dynamo/agentic_region_kernel_generation.py --backend cuda_cpp --objective tensorrt --min-speedup 1.05
    python examples/dynamo/agentic_region_kernel_generation.py --batch-size 2 --tuning '{"num_warps": [4, 8]}' --finalists 2

Rerunning a command reuses the accepted kernel after a fresh validation, without
an agent call. See :ref:`torch_tensorrt_region_py` for the generation options,
acceptance objectives, and the commands that inspect or replay saved runs.
"""

import argparse
import json
import logging

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region import KernelGenerator, SearchPolicy, fused_kernel

# The generator is configuration, created once outside the model, so the model
# keeps its ordinary constructor. main() replaces it with the command-line options.
GENERATOR = KernelGenerator()


class Model(torch.nn.Module):
    def forward(self, x, bias, scale):
        x = x + 1.0
        # Eager and training runs execute the block as written; only
        # torch_tensorrt.compile generates and installs a kernel for it.
        with fused_kernel(generator=GENERATOR, name="bias_relu_scale"):
            x = torch.relu(x + bias) * scale
        return x - 0.5


def tuning_domains(value):
    try:
        domains = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from None
    if not isinstance(domains, dict) or not all(
        isinstance(choices, list) for choices in domains.values()
    ):
        raise argparse.ArgumentTypeError("expected a JSON object of choice arrays")
    return domains


def main():
    parser = argparse.ArgumentParser(description="Generate a region kernel")
    parser.add_argument(
        "--provider",
        default="claude",
        help="claude, codex, or a module:attribute provider adapter",
    )
    parser.add_argument(
        "--backend", choices=("triton", "cutile", "cuda_cpp"), default="triton"
    )
    parser.add_argument("--model", help="Model name understood by the agent CLI")
    parser.add_argument("--instructions", help="Additional authoring guidance")
    parser.add_argument(
        "--instruction-file", action="append", default=[], help="Guidance file"
    )
    parser.add_argument("--objective", choices=("candidate", "eager", "tensorrt"))
    parser.add_argument("--min-speedup", type=float)
    parser.add_argument("--attempts", type=int, help="Maximum author rounds")
    parser.add_argument("--timeout", type=float, help="Search budget in seconds")
    parser.add_argument("--cache-dir", help="Override the persistent kernel cache")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-no-improvement", type=int)
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--finalists", type=int)
    parser.add_argument("--tuning", type=tuning_domains, help="JSON knob domains")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--diagnostics", action="store_true", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("torch_tensorrt.dynamo.region_compilation._generation").setLevel(
        logging.INFO
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")

    def given(**options):
        # Keep the library defaults for every option that was not passed.
        return {name: value for name, value in options.items() if value is not None}

    global GENERATOR
    GENERATOR = KernelGenerator(
        provider=args.provider,
        backend=args.backend,
        instruction_files=tuple(args.instruction_file),
        search=SearchPolicy(
            **given(
                batch_size=args.batch_size,
                max_candidates=args.max_candidates,
                max_no_improvement=args.max_no_improvement,
                max_variants=args.max_variants,
                finalists=args.finalists,
                tuning=args.tuning,
                resume=args.resume,
            )
        ),
        **given(
            model=args.model,
            instructions=args.instructions,
            objective=args.objective,
            min_speedup=args.min_speedup,
            max_attempts=args.attempts,
            timeout_s=args.timeout,
            cache_dir=args.cache_dir,
            diagnostics=args.diagnostics,
        ),
    )
    torch.manual_seed(0)
    inputs = tuple(torch.randn(257, 1025, device="cuda") for _ in range(3))
    model = Model().eval().cuda()
    compiled = torch_tensorrt.compile(
        model,
        inputs=list(inputs),
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,
        require_full_compilation=True,
    )
    engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
    records = [r for r in get_region_records(compiled) if r.route == "plugin"]
    assert len(engines) == len(records) == 1
    record = records[0]
    assert compiled.get_submodule(record.owner) is engines[0]
    layers = engines[0].get_layer_info()
    assert str(record.plugin_op) in layers
    with torch.inference_mode():
        torch.testing.assert_close(compiled(*inputs), model(*inputs))
        fresh = tuple(torch.randn_like(value) for value in inputs)
        fresh[0].view(-1)[:3] = fresh[0].new_tensor([torch.nan, torch.inf, -torch.inf])
        torch.testing.assert_close(compiled(*fresh), model(*fresh), equal_nan=True)
    print("\nVerified execution inside one TensorRT engine:")
    print("  TensorRT: x + 1")
    print(f"  -> QDP plugin: agent-generated {args.backend} (add -> ReLU -> scale)")
    print("  -> TensorRT: x - 0.5")
    print(f"Plugin operator: {record.plugin_op}")
    print("Engine layers:")
    print(json.dumps(json.loads(layers)["Layers"], indent=2))
    print("The complete compiled model matches PyTorch on fresh and nonfinite inputs.")


if __name__ == "__main__":
    main()
