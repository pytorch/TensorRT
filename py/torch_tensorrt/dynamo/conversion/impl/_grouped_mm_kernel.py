# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A grouped matrix multiply kernel for aten._grouped_mm, as a TensorRT plugin.

Every row is multiplied only by the weight of its own group. The Triton kernel is
compiled ahead of time and its PTX is stored in the engine, so running the engine
needs neither Triton nor PyTorch. Importing this module needs Triton.
"""

import functools
import logging
import threading
from typing import Any, Dict, Optional, Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from torch_tensorrt.kernels import _common, _triton

PLUGIN_NAME = "torch_tensorrt::grouped_mm"

_LOGGER = logging.getLogger(__name__)

# With 3 pipeline stages these tiles need up to 48 KiB of shared memory for 16-bit
# types, the most TensorRT gives a plugin kernel.
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 32
_NUM_WARPS = 4
_NUM_STAGES = (3, 2)

_TRITON_DTYPES = {trt.bfloat16: "bf16", trt.float16: "fp16"}

_lock = threading.Lock()
_registered = False


@triton.jit  # type: ignore[misc]
def _grouped_mm_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    offs_ptr,
    M,
    c_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUPS_POW2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program finds its group from the offsets on the device, so the
    # engine never copies them to the host. N tiles vary fastest, so the row
    # tiles of one group run together and share its weight through L2.
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile = pid // tiles_n
    pid_n = pid % tiles_n
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    group_ids = tl.arange(0, GROUPS_POW2)
    valid = group_ids < GROUPS
    ends = tl.load(offs_ptr + group_ids, mask=valid, other=0)
    starts = tl.load(offs_ptr + group_ids - 1, mask=valid & (group_ids > 0), other=0)
    ends = tl.where(valid, ends, starts)
    group_tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
    tiles_through = tl.cumsum(group_tiles, axis=0)
    total_tiles = tl.sum(group_tiles, axis=0)
    if tile >= total_tiles:
        # Rows past the last offset belong to no group. Zero them, as the dense
        # lowering does, so the output never keeps stale memory.
        rows = (
            tl.load(offs_ptr + GROUPS - 1)
            + (tile - total_tiles) * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )
        tl.store(
            c_ptr + rows.to(tl.int64)[:, None] * N + cols[None, :],
            tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty),
            mask=((rows >= 0) & (rows < M))[:, None] & col_mask[None, :],
        )
        return
    group = tl.sum((tiles_through <= tile).to(tl.int32), axis=0)
    selected = group_ids == group
    first_tile = tl.sum(tl.where(selected, tiles_through - group_tiles, 0), axis=0)
    row_start = tl.sum(tl.where(selected, starts, 0), axis=0)
    row_end = tl.sum(tl.where(selected, ends, 0), axis=0)

    rows = row_start + (tile - first_tile) * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets that count rows outside mat1 must not reach outside it or the output.
    row_mask = (rows >= 0) & (rows < tl.minimum(row_end, M))
    ks = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rows.to(tl.int64)[:, None] * K + ks[None, :]
    b_ptrs = b_ptr + group.to(tl.int64) * K * N + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = ks < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = c_ptr + rows.to(tl.int64)[:, None] * N + cols[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _describe(
    mat1: trtp.TensorDesc, mat2: trtp.TensorDesc, offs: trtp.TensorDesc
) -> trtp.TensorDesc:
    return trtp.from_shape_expr(
        (mat1.shape_expr[0], mat2.shape_expr[2]), dtype=mat1.dtype
    )


def _signature(pointer: str) -> Dict[str, str]:
    return {
        "a_ptr": pointer,
        "b_ptr": pointer,
        "offs_ptr": "*i32",
        "M": "i32",
        "c_ptr": pointer,
    }


def _compile(pointer: str, groups: int, k: int, n: int) -> Any:
    """The kernel's PTX for the current GPU, with the deepest pipeline that fits.

    Raises when the kernel cannot be built for a plugin launch. The converter's
    validator compiles it first, so a failure declines the op instead of failing
    the engine build, and the build reuses the cached result.
    """
    capability = triton.runtime.driver.active.get_current_target().arch
    artifact = _compile_for(pointer, groups, k, n, capability)
    if artifact is None:
        raise RuntimeError("the grouped GEMM kernel does not build for this shape")
    return artifact


# Failures are cached too, because the converter registry runs validators on every lookup.
@functools.lru_cache(maxsize=None)
def _compile_for(
    pointer: str, groups: int, k: int, n: int, capability: int
) -> Optional[Any]:
    try:
        return _build(pointer, groups, k, n, capability)
    except Exception as exc:
        _LOGGER.warning(
            "The grouped GEMM kernel does not build for %s with %d groups, K=%d and N=%d "
            "on sm_%d (%s). aten._grouped_mm ops of this shape follow the dense "
            "lowering's rules instead: the dense converter, or PyTorch where it declines.",
            pointer[1:],
            groups,
            k,
            n,
            capability,
            exc,
        )
        return None


def _build(pointer: str, groups: int, k: int, n: int, capability: int) -> Any:
    constexprs = {
        "N": n,
        "K": k,
        "GROUPS": groups,
        "GROUPS_POW2": triton.next_power_of_2(groups),
        "BLOCK_M": _BLOCK_M,
        "BLOCK_N": _BLOCK_N,
        "BLOCK_K": _BLOCK_K,
    }
    # From sm_100 on, Triton's tl.dot uses tensor memory, which a TensorRT
    # plugin launch cannot set up. sm_80's mma.sync runs on every later GPU.
    arch = 80 if capability >= 100 else None
    for num_stages in _NUM_STAGES:
        artifact = _triton.compile_triton_to_ptx(
            _grouped_mm_kernel,
            _signature(pointer),
            constexprs,
            num_warps=_NUM_WARPS,
            num_stages=num_stages,
            # TensorRT allocates its tensors 256-byte aligned and requires the same
            # of engine inputs and outputs. offs gets no hint: a load of a few
            # offsets gains nothing from one.
            pointer_alignment=dict.fromkeys(("a_ptr", "b_ptr", "c_ptr"), 16),
            arch=arch,
        )
        if artifact.shared_mem <= _common.AOT_MAX_DYNAMIC_SHARED_MEM:
            return artifact
    raise RuntimeError(
        f"the grouped GEMM kernel needs {artifact.shared_mem} bytes of shared memory, "
        f"more than the {_common.AOT_MAX_DYNAMIC_SHARED_MEM} a TensorRT plugin kernel gets"
    )


def _aot_impl(
    mat1: trtp.TensorDesc,
    mat2: trtp.TensorDesc,
    offs: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    groups, k, n = (_common.constant_int(dim) for dim in mat2.shape_expr)
    pointer = "*" + _TRITON_DTYPES[mat1.dtype]
    artifact = _compile(pointer, groups, k, n)

    def grid(inputs: Tuple[trtp.TensorDesc, ...], _: object) -> object:
        # Each group and the rows past the last offset can end in a partial row
        # tile, which adds at most one tile per group.
        rows = inputs[0].shape_expr[0]
        return (trtp.cdiv(rows, _BLOCK_M) + groups) * triton.cdiv(n, _BLOCK_N)

    launch = _common.make_pointer_aot_fn(
        PLUGIN_NAME,
        _triton.analyze_signature(_signature(pointer), (3, 1)),
        grid,
        (artifact.num_warps * 32, 1, 1),
        artifact.shared_mem,
        lambda inputs, _: [trtp.SymInt32(inputs[0].shape_expr[0])],
        lambda: None,
        frontend="aten._grouped_mm",
    )
    params, extra_args = launch([mat1, mat2, offs], outputs, tactic)
    return artifact.kernel_name, artifact.ptx, params, extra_args


def plugin() -> Any:
    """The plugin, registered with TensorRT on first use."""
    global _registered
    with _lock:
        if not _registered:
            trtp.register(PLUGIN_NAME)(_describe)
            trtp.aot_impl(PLUGIN_NAME)(_aot_impl)
            _registered = True
    return trtp.op.torch_tensorrt.grouped_mm
