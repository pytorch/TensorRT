# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare bounded Triton configurations with their TensorRT AOT engines.

Run with a fresh TRITON_CACHE_DIR to measure cold compilation. Subsequent calls
reuse Triton's own cache. No PTX rewriting or unproven alignment assumptions.
"""

import argparse
import json
import time
from pathlib import Path

import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

import torch_tensorrt
from torch_tensorrt.kernels import triton_op
from torch_tensorrt.kernels._triton import compile_triton_to_ptx


@triton.jit
def fused(x, n, y, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0)
    tl.store(y + offsets, values * values + values, offsets < n)


def meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def elapsed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def benchmark(block, warps, sizes, repetitions):
    signature = {"x": "*fp32", "n": "i32", "y": "*fp32"}
    constants = {"BLOCK": block}
    compile_kernel = lambda: compile_triton_to_ptx(
        fused, signature, constants, num_warps=warps
    )
    artifact, first_compile = elapsed(compile_kernel)
    _, warm_compile = elapsed(compile_kernel)
    name = f"benchmark_triton::fused_b{block}_w{warps}"
    _, registration = elapsed(
        lambda: triton_op(
            name,
            fused,
            signature,
            constants,
            grid=lambda i, o: (trtp.cdiv(i[0].shape_expr.numel(), block),),
            meta_fn=meta,
            extra_args_fn=lambda i, o: [trtp.SymInt32(i[0].shape_expr.numel())],
            num_warps=warps,
        )
    )
    op = getattr(torch.ops.benchmark_triton, name.split("::")[1])

    class Model(torch.nn.Module):
        def forward(self, x):
            return op(x)

    compiled, build = elapsed(
        lambda: torch_tensorrt.compile(
            Model(),
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(min(sizes),),
                    opt_shape=(sizes[len(sizes) // 2],),
                    max_shape=(max(sizes),),
                    dtype=torch.float32,
                )
            ],
            min_block_size=1,
            require_full_compilation=True,
        )
    )
    timings = []
    for size in sizes:
        x = torch.randn(size, device="cuda")
        y = torch.empty_like(x)
        direct = lambda: fused[(triton.cdiv(size, block),)](
            x, size, y, BLOCK=block, num_warps=warps
        )
        direct()
        expected = x * x + x
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(compiled(x), expected)
        timings.append(
            dict(
                size=size,
                direct_triton_ms=do_bench(direct, warmup=10, rep=repetitions),
                tensorrt_ms=do_bench(lambda: compiled(x), warmup=10, rep=repetitions),
            )
        )
    return dict(
        block=block,
        warps=warps,
        target=str(artifact.target),
        ptx_version=artifact.ptx_version,
        first_compile_ms=first_compile,
        warm_compile_ms=warm_compile,
        registration_ms=registration,
        engine_build_ms=build,
        timings=timings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 262144, 1048576])
    parser.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--rep", type=int, default=100, help="Timing duration in milliseconds"
    )
    parser.add_argument(
        "--output", type=Path, help="Write clean JSON separately from library logs"
    )
    args = parser.parse_args()
    if min(args.sizes) <= 0 or len(set(args.sizes)) < 2:
        parser.error(
            "provide at least two distinct positive sizes for the dynamic profile"
        )
    if args.rep <= 0:
        parser.error("--rep must be positive")
    for block in args.blocks:
        if block <= 0 or block & (block - 1):
            parser.error("--blocks must be positive powers of two")
    report = json.dumps(
        [
            benchmark(b, w, sorted(args.sizes), args.rep)
            for b in args.blocks
            for w in args.warps
        ],
        indent=2,
    )
    if args.output:
        args.output.write_text(report + "\n")
    else:
        print(report)
