# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Grouped matrix multiply kernels that read NVFP4 or MXFP8 expert weights, as TensorRT plugins.

They compute ``tensorrt.grouped_mm_{nvfp4,mxfp8}`` (rows grouped by offsets) and
``tensorrt.gather_mm_{nvfp4,mxfp8}`` (each row names its group): each row is
multiplied only by its own group's weight, dequantized on the fly, so a decode step
reads 4 or 8 bits per weight instead of 16. One program computes one
row against a block of output columns. That suits the few rows per group of
decoding, where reading the weights is the whole cost; at many rows per group each
row reads its group's weights again.

The packed weights and scales reach the kernel as int32 tensors holding the same
bytes, since TensorRT has no 4-bit or E8M0 tensor type a plugin can take.
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

PLUGIN_NAMES = {
    (True, False): "torch_tensorrt::grouped_mm_nvfp4",
    (False, False): "torch_tensorrt::grouped_mm_mxfp8",
    (True, True): "torch_tensorrt::gather_mm_nvfp4",
    (False, True): "torch_tensorrt::gather_mm_mxfp8",
}

_LOGGER = logging.getLogger(__name__)

# (BLOCK_N, BLOCK_K), measured on H100 for 2304x2304 and 2304x1152 expert layers.
_TILE = (16, 256)
_NUM_WARPS = 4

_TRITON_DTYPES = {trt.bfloat16: "bf16", trt.float16: "fp16"}

_lock = threading.Lock()
_registered = False


@triton.jit  # type: ignore[misc]
def _row_group(offs_ptr, row, GROUPS: tl.constexpr, GROUPS_POW2: tl.constexpr):  # type: ignore[no-untyped-def]
    # The group whose rows [offs[g - 1], offs[g]) hold this row; GROUPS past the last offset.
    group_ids = tl.arange(0, GROUPS_POW2)
    ends = tl.load(offs_ptr + group_ids, mask=group_ids < GROUPS, other=2147483647)
    return tl.sum((ends <= row).to(tl.int32), axis=0)


@triton.jit  # type: ignore[misc]
def _e2m1_to_bf16(codes):  # type: ignore[no-untyped-def]
    # Four bytes (eight E2M1 codes, code p holding K offset p) per call. Each code's sign
    # goes to bit 15 of a bf16 and its three magnitude bits to the exponent and mantissa
    # boundary, which reads the code as its value times 2^-126; a bf16x2 multiply by
    # 2^126 then gives the exact value. The two outputs hold K offsets (3, 7, 2, 6) and
    # (1, 5, 0, 4) of the four bytes, in byte order.
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 q, s, t, c, z;
        mov.b32 q, $4;
        mov.b32 c, 0x7E807E80;
        mov.b32 z, 0x80008000;
        and.b32 s, q, 0x80008000; shr.b32 t, q, 6; and.b32 t, t, 0x01C001C0; or.b32 t, t, s; fma.rn.bf16x2 $0, t, c, z;
        shl.b32 q, q, 4;
        and.b32 s, q, 0x80008000; shr.b32 t, q, 6; and.b32 t, t, 0x01C001C0; or.b32 t, t, s; fma.rn.bf16x2 $1, t, c, z;
        shl.b32 q, q, 4;
        and.b32 s, q, 0x80008000; shr.b32 t, q, 6; and.b32 t, t, 0x01C001C0; or.b32 t, t, s; fma.rn.bf16x2 $2, t, c, z;
        shl.b32 q, q, 4;
        and.b32 s, q, 0x80008000; shr.b32 t, q, 6; and.b32 t, t, 0x01C001C0; or.b32 t, t, s; fma.rn.bf16x2 $3, t, c, z;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[codes],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )


@triton.jit  # type: ignore[misc]
def _grouped_mm_nvfp4_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    w_ptr,
    s_ptr,
    gs_ptr,
    offs_ptr,
    M,
    c_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUPS_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INDEXED: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_n: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    row = pid // tiles_n
    cols = (pid % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    if row >= M:
        return
    if INDEXED:
        # offs_ptr holds each row's group index (gather_mm) rather than group offsets (grouped_mm).
        group = tl.load(offs_ptr + row).to(tl.int32)
    else:
        group = _row_group(offs_ptr, row, GROUPS, GROUPS_POW2)
    if group >= GROUPS or group < 0:
        # Rows past the last offset belong to no group. Zero them, as the dense
        # lowering does, so the output never keeps stale memory.
        tl.store(
            c_ptr + row.to(tl.int64) * N + cols,
            tl.zeros((BLOCK_N,), dtype=c_ptr.dtype.element_ty),
            mask=col_mask,
        )
        return
    w_bytes = w_ptr.to(tl.pointer_type(tl.uint8))
    s_bytes = s_ptr.to(tl.pointer_type(tl.float8e4nv))
    g = group.to(tl.int64)
    # Each thread's eight bytes are one 16-value scale block.
    KB: tl.constexpr = BLOCK_K // 16
    kb = tl.arange(0, KB)
    byte = tl.arange(0, 8)
    lane = byte % 4
    k_first = 8 * (byte // 4) + 3 - lane // 2 + 4 * (lane % 2)
    k_second = k_first - 2
    w_rows = w_bytes + g * N * (K // 2) + cols.to(tl.int64)[:, None, None] * (K // 2)
    s_rows = s_bytes + g * N * (K // 16) + cols.to(tl.int64)[:, None] * (K // 16)
    a_row = a_ptr + row.to(tl.int64) * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + kb * 16) < K
        codes = tl.load(
            w_rows + (k // 2 + kb * 8)[None, :, None] + byte[None, None, :],
            mask=col_mask[:, None, None] & k_mask[None, :, None],
            other=0,
        )
        a_first = tl.load(
            a_row + k + kb[:, None] * 16 + k_first[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        a_second = tl.load(
            a_row + k + kb[:, None] * 16 + k_second[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        v_first, v_second = _e2m1_to_bf16(codes)
        block = tl.sum(
            v_first.to(tl.float32) * a_first.to(tl.float32)[None]
            + v_second.to(tl.float32) * a_second.to(tl.float32)[None],
            axis=2,
        )
        scale = tl.load(
            s_rows + (k // 16 + kb)[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(block * scale.to(tl.float32), axis=1)
    acc = acc * tl.load(gs_ptr + group)
    tl.store(
        c_ptr + row.to(tl.int64) * N + cols,
        acc.to(c_ptr.dtype.element_ty),
        mask=col_mask,
    )


@triton.jit  # type: ignore[misc]
def _grouped_mm_mxfp8_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    w_ptr,
    s_ptr,
    offs_ptr,
    M,
    c_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUPS_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INDEXED: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_n: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    row = pid // tiles_n
    cols = (pid % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    if row >= M:
        return
    if INDEXED:
        # offs_ptr holds each row's group index (gather_mm) rather than group offsets (grouped_mm).
        group = tl.load(offs_ptr + row).to(tl.int32)
    else:
        group = _row_group(offs_ptr, row, GROUPS, GROUPS_POW2)
    if group >= GROUPS or group < 0:
        tl.store(
            c_ptr + row.to(tl.int64) * N + cols,
            tl.zeros((BLOCK_N,), dtype=c_ptr.dtype.element_ty),
            mask=col_mask,
        )
        return
    w_values = w_ptr.to(tl.pointer_type(tl.float8e4nv))
    s_bytes = s_ptr.to(tl.pointer_type(tl.uint8))
    g = group.to(tl.int64)
    KB: tl.constexpr = BLOCK_K // 32
    kb = tl.arange(0, KB)
    offset = tl.arange(0, 32)
    w_rows = w_values + g * N * K + cols.to(tl.int64)[:, None, None] * K
    s_rows = s_bytes + g * N * (K // 32) + cols.to(tl.int64)[:, None] * (K // 32)
    a_row = a_ptr + row.to(tl.int64) * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + kb * 32) < K
        w = tl.load(
            w_rows + (k + kb * 32)[None, :, None] + offset[None, None, :],
            mask=col_mask[:, None, None] & k_mask[None, :, None],
            other=0.0,
        )
        a = tl.load(
            a_row + k + kb[:, None] * 32 + offset[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        block = tl.sum(w.to(tl.float32) * a.to(tl.float32)[None], axis=2)
        exponent = tl.load(
            s_rows + (k // 32 + kb)[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=127,
        )
        # E8M0 holds a biased exponent: shifted into a float32's exponent field it is 2^(e - 127).
        acc += tl.sum(
            block * (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True), axis=1
        )
    tl.store(
        c_ptr + row.to(tl.int64) * N + cols,
        acc.to(c_ptr.dtype.element_ty),
        mask=col_mask,
    )


def _signature(pointer: str, nvfp4: bool, indexed: bool) -> Dict[str, str]:
    signature = {"a_ptr": pointer, "w_ptr": "*i32", "s_ptr": "*i32"}
    if nvfp4:
        signature["gs_ptr"] = "*fp32"
    signature.update(
        {"offs_ptr": "*i32", "M": "i32", "c_ptr": pointer}
    )
    return signature


def compile_kernel(
    pointer: str, nvfp4: bool, groups: int, k: int, n: int, indexed: bool = False
) -> Any:
    """The kernel's PTX for the current GPU. Raises when it cannot be built for a plugin launch."""
    capability = triton.runtime.driver.active.get_current_target().arch
    kernel = _compile_for(pointer, nvfp4, groups, k, n, capability, indexed)
    if kernel is None:
        raise RuntimeError(
            "the quantized grouped GEMM kernel does not build for this shape"
        )
    return kernel


@functools.lru_cache(maxsize=None)
def _compile_for(
    pointer: str,
    nvfp4: bool,
    groups: int,
    k: int,
    n: int,
    capability: int,
    indexed: bool,
) -> Optional[Any]:
    try:
        block_n, block_k = _TILE
        artifact = _triton.compile_triton_to_ptx(
            _grouped_mm_nvfp4_kernel if nvfp4 else _grouped_mm_mxfp8_kernel,
            _signature(pointer, nvfp4, indexed),
            {
                "N": n,
                "K": k,
                "GROUPS": groups,
                "GROUPS_POW2": triton.next_power_of_2(groups),
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "INDEXED": indexed,
            },
            num_warps=_NUM_WARPS,
            num_stages=1,
            pointer_alignment=dict.fromkeys(("a_ptr", "w_ptr", "s_ptr", "c_ptr"), 16),
            # Other GPUs from sm_100 on run sm_80's instructions.
            arch=80 if capability >= 100 else None,
        )
        if artifact.shared_mem > _common.AOT_MAX_DYNAMIC_SHARED_MEM:
            raise RuntimeError(f"it needs {artifact.shared_mem} bytes of shared memory")
        return artifact
    except Exception as exc:
        _LOGGER.warning(
            "The %s grouped GEMM kernel does not build for %s with %d groups, K=%d and N=%d on sm_%d (%s).",
            "NVFP4" if nvfp4 else "MXFP8",
            pointer[1:],
            groups,
            k,
            n,
            capability,
            exc,
        )
        return None


def _describe(
    mat1: trtp.TensorDesc, weight: trtp.TensorDesc, *_: trtp.TensorDesc
) -> trtp.TensorDesc:
    return trtp.from_shape_expr(
        (mat1.shape_expr[0], weight.shape_expr[1]), dtype=mat1.dtype
    )


def _aot(
    nvfp4: bool, indexed: bool, mat1: trtp.TensorDesc, weight: trtp.TensorDesc
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    groups, n = (_common.constant_int(dim) for dim in weight.shape_expr[:2])
    k = _common.constant_int(mat1.shape_expr[1])
    artifact = compile_kernel(
        "*" + _TRITON_DTYPES[mat1.dtype], nvfp4, groups, k, n, indexed
    )
    params = trtp.KernelLaunchParams(
        grid_x=mat1.shape_expr[0] * triton.cdiv(n, _TILE[0]),
        grid_y=1,
        grid_z=1,
        block_x=artifact.num_warps * 32,
        block_y=1,
        block_z=1,
        shared_mem=artifact.shared_mem,
    )
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(mat1.shape_expr[0])
    return artifact.kernel_name, artifact.ptx, params, extra_args


_AotResult = Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]


def _aot_grouped_nvfp4(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    scale_2: trtp.TensorDesc,
    offs: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> _AotResult:
    return _aot(True, False, mat1, weight)


def _aot_grouped_mxfp8(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    offs: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> _AotResult:
    return _aot(False, False, mat1, weight)


def _aot_gather_nvfp4(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    scale_2: trtp.TensorDesc,
    index: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> _AotResult:
    return _aot(True, True, mat1, weight)


def _aot_gather_mxfp8(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    index: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> _AotResult:
    return _aot(False, True, mat1, weight)


def _describe_grouped_nvfp4(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    scale_2: trtp.TensorDesc,
    offs: trtp.TensorDesc,
) -> trtp.TensorDesc:
    return _describe(mat1, weight)


def _describe_grouped_mxfp8(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    offs: trtp.TensorDesc,
) -> trtp.TensorDesc:
    return _describe(mat1, weight)


def _describe_gather_nvfp4(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    scale_2: trtp.TensorDesc,
    index: trtp.TensorDesc,
) -> trtp.TensorDesc:
    return _describe(mat1, weight)


def _describe_gather_mxfp8(
    mat1: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    scale: trtp.TensorDesc,
    index: trtp.TensorDesc,
) -> trtp.TensorDesc:
    return _describe(mat1, weight)


_CALLBACKS = {
    (True, False): (_describe_grouped_nvfp4, _aot_grouped_nvfp4),
    (False, False): (_describe_grouped_mxfp8, _aot_grouped_mxfp8),
    (True, True): (_describe_gather_nvfp4, _aot_gather_nvfp4),
    (False, True): (_describe_gather_mxfp8, _aot_gather_mxfp8),
}


def plugin(nvfp4: bool, indexed: bool) -> Any:
    """The plugin for one weight format and row-to-group form, registered with TensorRT on first use."""
    global _registered
    with _lock:
        if not _registered:
            for key, (describe, aot) in _CALLBACKS.items():
                trtp.register(PLUGIN_NAMES[key])(describe)
                trtp.aot_impl(PLUGIN_NAMES[key])(aot)
            _registered = True
    return getattr(
        trtp.op.torch_tensorrt, PLUGIN_NAMES[(nvfp4, indexed)].split("::")[1]
    )
