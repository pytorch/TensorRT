# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _agentic_region_kernel_generation_example:

Generate a region kernel with an agent and run it inside TensorRT
================================================================

Install and log into the selected Codex or Claude CLI first. Run with CUDA,
TensorRT QDP support, and the selected backend's toolchain. Triton uses
``triton_op`` prerequisites; cuTile needs CUDA Toolkit and loaded driver 13.2+
with a matching tileiras compiler. CUDA C++ needs CUDA Python's ``cuda-core`` and
``cuda-bindings`` packages plus NVRTC::

    codex login
    python examples/dynamo/agentic_region_kernel_generation.py --provider codex
    python examples/dynamo/agentic_region_kernel_generation.py --provider codex --backend cutile
    python examples/dynamo/agentic_region_kernel_generation.py --provider claude --backend cuda_cpp
    python examples/dynamo/agentic_region_kernel_generation.py --instructions "Try 256-element tiles."
    python examples/dynamo/agentic_region_kernel_generation.py --instruction-file kernel-guidance.md
    python examples/dynamo/agentic_region_kernel_generation.py --objective tensorrt --min-speedup 1.05

For Claude, use its normal interactive login, then pass ``--provider claude``.
``--model`` optionally selects a model; otherwise the CLI keeps its default.
Model calls use your account allowance and receive the annotated graph and
tensor metadata. Generation and evaluation happen during compilation.

The author receives the marked region, packaged backend guidance, any explicit
user instructions, and measured feedback. It can request bounded pages of
recorded evidence. Torch-TensorRT compiles proposals into isolated QDP engines,
checks correctness, and compares interleaved timing samples. The selected
candidate receives fresh final validation before its plugin is installed.
Inference needs neither an agent nor a Python kernel launcher.

Rerun the same command to reuse a kernel without an agent call when cache
revalidation passes. Logs identify generated source and the artifact directory
containing the task, exact instructions, per-attempt feedback, and accepted
candidate. Changed guidance changes the cache key.

The default ``candidate`` objective ranks correct generated candidates. Choose
``eager`` or ``tensorrt`` to require a speedup against that region's PyTorch or
fully compiled TensorRT baseline. ``min_speedup`` is baseline latency divided
by candidate latency, with a conservative allowance for measured timing variation.
These measurements do not establish whole-model speedup.
``cuda_cpp`` generates CUDA C++ compiled through CUDA Python, not Numba kernels.
"""

import argparse
import json
import logging

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region import KernelGenerator, fused_kernel


class Model(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, bias, scale):
        x = x + 1.0
        with fused_kernel(generator=self.generator, name="bias_relu_scale"):
            x = torch.relu(x + bias) * scale
        return x - 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("codex", "claude"), default="codex")
    parser.add_argument(
        "--backend", choices=("triton", "cutile", "cuda_cpp"), default="triton"
    )
    parser.add_argument("--model", help="Optional model name understood by your CLI")
    parser.add_argument("--instructions", help="Additional kernel-authoring guidance")
    parser.add_argument(
        "--instruction-file",
        action="append",
        default=[],
        help="Additional guidance file; repeat to include multiple files",
    )
    parser.add_argument(
        "--objective", choices=("candidate", "eager", "tensorrt"), default="candidate"
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Required baseline/candidate latency ratio for eager or tensorrt",
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="Maximum candidate proposals per region"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Search budget per region, in seconds",
    )
    parser.add_argument("--cache-dir", help="Override the persistent kernel cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("torch_tensorrt.dynamo.region_compilation._generation").setLevel(
        logging.INFO
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")
    generator = KernelGenerator(
        provider=args.provider,
        backend=args.backend,
        model=args.model,
        instructions=args.instructions,
        instruction_files=tuple(args.instruction_file),
        objective=args.objective,
        min_speedup=args.min_speedup,
        max_attempts=args.attempts,
        timeout_s=args.timeout,
        cache_dir=args.cache_dir,
    )
    torch.manual_seed(0)
    inputs = tuple(torch.randn(257, 1025, device="cuda") for _ in range(3))
    model = Model(generator).eval().cuda()
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
    assert record.owner is not None
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
