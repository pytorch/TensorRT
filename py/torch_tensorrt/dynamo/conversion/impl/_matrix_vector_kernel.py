# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A matrix multiply with a few rows and a constant weight, as a TensorRT plugin.

With one to 16 rows, as in a decoding step, the work is reading the weight once.
Each program reads a block of output columns over a slice of K, so the whole GPU
reads the weight in parallel. With one slice it writes the result; with several
it writes float32 partial sums, which the caller adds with a TensorRT reduce. The Triton kernel is compiled ahead of time and its PTX is
stored in the engine, so running the engine needs neither Triton nor PyTorch.
Importing this module needs Triton.
"""

import functools
import threading
from typing import Any, Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from torch_tensorrt.kernels import _common, _triton

PLUGIN_NAME = "torch_tensorrt::matrix_vector"

# tl.dot needs at least 16 rows, so the kernel computes 16 and stores the real ones.
MAX_ROWS = 16
# (BLOCK_N, num_warps, num_stages), within 3% of the best tile for every decoder
# projection measured on H100 and on DGX Spark, and within the 48 KiB of shared
# memory a TensorRT plugin kernel gets.
_TILE = (16, 2, 4)

_TRITON_DTYPES = {trt.bfloat16: "bf16", trt.float16: "fp16"}
POINTERS = {torch.bfloat16: "*bf16", torch.float16: "*fp16"}

_lock = threading.Lock()
_registered = False


@triton.jit  # type: ignore[misc]
def _matrix_vector_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    w_ptr,
    p_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    SPLIT_K: tl.constexpr = K // SPLITS
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k in range(split * SPLIT_K, split * SPLIT_K + SPLIT_K, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rows[:, None] * K + ks[None, :], mask=(rows < M)[:, None], other=0.0
        )
        w = tl.load(
            w_ptr + cols.to(tl.int64)[None, :] * K + ks[:, None],
            mask=(cols < N)[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w, acc)
    tl.store(
        p_ptr + (split * M + rows[:, None]) * N + cols[None, :],
        acc.to(p_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


def block_k(k: int) -> int:
    """The largest BLOCK_K up to 256 that divides K; K must be a multiple of 16."""
    size = 256
    while k % size:
        size //= 2
    return size


def num_splits(n: int, k: int) -> int:
    """Slices of K, as many as keep up to four programs per streaming multiprocessor."""
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    tiles = triton.cdiv(n, _TILE[0])
    size = block_k(k)
    splits = 1
    for candidate in range(2, 9):
        if k % (candidate * size) == 0 and tiles * candidate <= 4 * sms:
            splits = candidate
    return splits


@functools.lru_cache(maxsize=None)
def compile_kernel(pointer: str, m: int, n: int, k: int, splits: int) -> Any:
    block_n, num_warps, num_stages = _TILE
    artifact = _triton.compile_triton_to_ptx(
        _matrix_vector_kernel,
        {
            "a_ptr": pointer,
            "w_ptr": pointer,
            "p_ptr": "*fp32" if splits > 1 else pointer,
        },
        {
            "M": m,
            "N": n,
            "K": k,
            "SPLITS": splits,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k(k),
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if artifact.shared_mem > _common.AOT_MAX_DYNAMIC_SHARED_MEM:
        raise RuntimeError(
            f"the matrix vector kernel needs {artifact.shared_mem} bytes of shared memory, "
            f"more than the {_common.AOT_MAX_DYNAMIC_SHARED_MEM} a TensorRT plugin kernel gets"
        )
    return artifact


def _describe(a: trtp.TensorDesc, w: trtp.TensorDesc, splits: int) -> trtp.TensorDesc:
    # One slice writes the result; several write float32 partial sums for the caller to add.
    return trtp.from_shape_expr(
        (splits, a.shape_expr[0], w.shape_expr[0]),
        dtype=trt.float32 if splits > 1 else a.dtype,
    )


def _aot_impl(
    a: trtp.TensorDesc,
    w: trtp.TensorDesc,
    splits: int,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    m, k = (_common.constant_int(a.shape_expr[i]) for i in (0, 1))
    n = _common.constant_int(w.shape_expr[0])
    artifact = compile_kernel("*" + _TRITON_DTYPES[a.dtype], m, n, k, splits)
    params = trtp.KernelLaunchParams(
        grid_x=triton.cdiv(n, _TILE[0]),
        grid_y=splits,
        grid_z=1,
        block_x=artifact.num_warps * 32,
        block_y=1,
        block_z=1,
        shared_mem=artifact.shared_mem,
    )
    return artifact.kernel_name, artifact.ptx, params, trtp.SymIntExprs(0)


def plugin() -> Any:
    """The matrix vector plugin, registered with TensorRT on first use."""
    global _registered
    with _lock:
        if not _registered:
            trtp.register(PLUGIN_NAME)(_describe)
            trtp.aot_impl(PLUGIN_NAME)(_aot_impl)
            _registered = True
    return trtp.op.torch_tensorrt.matrix_vector
