# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Attention for a single query row, as two TensorRT plugins.

Decoding attends one new token to the whole key and value cache, so the work is
reading the cache once. The first kernel splits the cache into chunks that run in
parallel and writes a partial result per chunk; the second merges them. A TensorRT
AOT plugin launches one kernel, hence two plugins. The Triton kernels are compiled
ahead of time and their PTX is stored in the engine, so running the engine needs
neither Triton nor PyTorch. Importing this module needs Triton.
"""

import functools
import math
import threading
from typing import Any, Optional, Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl
from torch_tensorrt.kernels import _common, _triton

SPLIT_PLUGIN_NAME = "torch_tensorrt::decode_attention_split"
MASKED_SPLIT_PLUGIN_NAME = "torch_tensorrt::decode_attention_split_masked"
COMBINE_PLUGIN_NAME = "torch_tensorrt::decode_attention_combine"

# (BLOCK_N, num_warps, num_stages), fastest first on an H100. Later entries need
# less shared memory, for head sizes where the first does not fit the 48 KiB a
# TensorRT plugin kernel gets.
_SPLIT_CONFIGS = ((64, 4, 1), (32, 4, 1), (16, 4, 1))

_TRITON_DTYPES = {trt.bfloat16: "bf16", trt.float16: "fp16", trt.float32: "fp32"}
_OUTPUT_DTYPES = {"bf16": trt.bfloat16, "fp16": trt.float16}

_lock = threading.Lock()
_registered = False


@triton.jit  # type: ignore[misc]
def _split_kernel(  # type: ignore[no-untyped-def]
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    S,
    o_ptr,
    lse_ptr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    MASK_HEADS: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_SCALE: tl.constexpr,
):
    # One program per (chunk, batch * key head). It serves all GROUP query heads
    # of its key head, so each key and value row is read once.
    split = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HKV
    heads = (bh % HKV) * GROUP + tl.arange(0, GROUP_ROWS)
    head_valid = heads < (bh % HKV + 1) * GROUP
    dims = tl.arange(0, D_POW2)
    dim_valid = dims < D

    q = tl.load(
        q_ptr + (b * HQ + heads)[:, None] * D + dims[None, :],
        mask=head_valid[:, None] & dim_valid[None, :],
        other=0.0,
    )
    chunk = tl.cdiv(tl.cdiv(S, SPLITS), BLOCK_N) * BLOCK_N
    start = split * chunk
    end = tl.minimum(start + chunk, S)
    kv_base = bh.to(tl.int64) * S * D
    if mask_ptr is not None:
        # The mask broadcasts over batch and heads when those dims are 1.
        if MASK_HEADS == 1:
            mask_heads = heads * 0
        else:
            mask_heads = heads
        mask_rows = (
            mask_ptr
            + (b * MASK_BATCH * MASK_HEADS + mask_heads)[:, None].to(tl.int64) * S
        )

    m = tl.full((GROUP_ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((GROUP_ROWS,), tl.float32)
    acc = tl.zeros((GROUP_ROWS, D_POW2), tl.float32)
    for n in range(start, end, BLOCK_N):
        cols = n + tl.arange(0, BLOCK_N)
        valid = cols < end
        if mask_ptr is not None:
            bias = (
                tl.load(
                    mask_rows + cols[None, :],
                    mask=head_valid[:, None] & valid[None, :],
                    other=float("-inf"),
                ).to(tl.float32)
                * 1.4426950408889634
            )
            # Masked loads move no data, so cache rows no query head may see,
            # such as the empty end of a static cache, cost nothing.
            valid = valid & (tl.max(bias, axis=0) > float("-inf"))
        kv_offs = kv_base + cols[:, None] * D + dims[None, :]
        kv_mask = valid[:, None] & dim_valid[None, :]
        k = tl.load(k_ptr + kv_offs, mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * QK_SCALE
        if mask_ptr is not None:
            s += bias
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        # A row that has seen only masked keys keeps max -inf; exp2(-inf - -inf) is NaN.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_offs, mask=kv_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    # Partials are normalized, with their log-sum-exp in base 2. A chunk with no
    # visible key gets weight zero in the merge.
    seen = l > 0
    rows = (b * HQ + heads) * SPLITS + split
    tl.store(
        o_ptr + rows[:, None] * D + dims[None, :],
        acc / tl.where(seen, l, 1.0)[:, None],
        mask=head_valid[:, None] & dim_valid[None, :],
    )
    tl.store(
        lse_ptr + rows,
        tl.where(seen, m + tl.math.log2(l), float("-inf")),
        mask=head_valid,
    )


@triton.jit  # type: ignore[misc]
def _combine_kernel(  # type: ignore[no-untyped-def]
    o_ptr,
    lse_ptr,
    out_ptr,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, D_POW2)
    dim_valid = dims < D
    lse = tl.load(lse_ptr + row * SPLITS + splits)
    top = tl.max(lse, axis=0)
    # A row that sees no key at all ends up 0 / 0, NaN, as in PyTorch.
    weights = tl.math.exp2(lse - tl.where(top == float("-inf"), 0.0, top))
    o = tl.load(
        o_ptr + (row * SPLITS + splits)[:, None] * D + dims[None, :],
        mask=dim_valid[None, :],
        other=0.0,
    )
    out = tl.sum(weights[:, None] * o, axis=0) / tl.sum(weights, axis=0)
    tl.store(out_ptr + row * D + dims, out.to(out_ptr.dtype.element_ty), mask=dim_valid)


def num_splits(batch: int, kv_heads: int, seq_len: Optional[int]) -> int:
    """How many chunks each key head's cache is split into.

    About two programs per SM keep every SM busy while each still reads a long
    enough run of the cache, and no chunk is shorter than one tile.
    """
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    splits = 1 << max(0, (2 * sms // (batch * kv_heads)).bit_length() - 1)
    if seq_len is not None:
        splits = min(splits, triton.next_power_of_2(triton.cdiv(seq_len, 64)))
    return splits


def compile_kernels(
    pointer: str,
    mask_pointer: Optional[str],
    heads: int,
    kv_heads: int,
    head_dim: int,
    mask_batch: bool,
    mask_heads: int,
    splits: int,
    scale: float,
) -> None:
    """Build both kernels now, so a failure can still decline the op.

    The engine build then reuses the cached result.
    """
    _split_artifact(
        pointer,
        mask_pointer,
        heads,
        kv_heads,
        head_dim,
        mask_batch,
        mask_heads,
        splits,
        scale,
    )
    _combine_artifact(pointer, head_dim, splits)


@functools.lru_cache(maxsize=None)
def _split_artifact(
    pointer: str,
    mask_pointer: Optional[str],
    heads: int,
    kv_heads: int,
    head_dim: int,
    mask_batch: bool,
    mask_heads: int,
    splits: int,
    scale: float,
) -> Any:
    group = heads // kv_heads
    signature = {
        "q_ptr": pointer,
        "k_ptr": pointer,
        "v_ptr": pointer,
        "mask_ptr": mask_pointer,
        "S": "i32",
        "o_ptr": "*fp32",
        "lse_ptr": "*fp32",
    }
    constexprs = {
        "HQ": heads,
        "HKV": kv_heads,
        "D": head_dim,
        "D_POW2": triton.next_power_of_2(head_dim),
        "GROUP": group,
        # tl.dot needs at least 16 rows.
        "GROUP_ROWS": max(16, triton.next_power_of_2(group)),
        "MASK_HEADS": mask_heads,
        "MASK_BATCH": int(mask_batch),
        "SPLITS": splits,
        "QK_SCALE": scale * math.log2(math.e),
    }
    if mask_pointer is None:
        del signature["mask_ptr"]
        constexprs["mask_ptr"] = None
    for block_n, num_warps, num_stages in _SPLIT_CONFIGS:
        artifact = _triton.compile_triton_to_ptx(
            _split_kernel,
            signature,
            {**constexprs, "BLOCK_N": block_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if artifact.shared_mem <= _common.AOT_MAX_DYNAMIC_SHARED_MEM:
            return artifact
    raise RuntimeError(
        f"the decode attention kernel needs {artifact.shared_mem} bytes of shared "
        f"memory for head size {head_dim}, more than the "
        f"{_common.AOT_MAX_DYNAMIC_SHARED_MEM} a TensorRT plugin kernel gets"
    )


@functools.lru_cache(maxsize=None)
def _combine_artifact(pointer: str, head_dim: int, splits: int) -> Any:
    return _triton.compile_triton_to_ptx(
        _combine_kernel,
        {"o_ptr": "*fp32", "lse_ptr": "*fp32", "out_ptr": pointer},
        {
            "D": head_dim,
            "D_POW2": triton.next_power_of_2(head_dim),
            "SPLITS": splits,
        },
        num_warps=4,
    )


def _describe_split(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    scale: float,
    splits: int,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    batch, heads, head_dim = (q.shape_expr[i] for i in (0, 1, 3))
    return (
        trtp.from_shape_expr((batch, heads, splits, head_dim), dtype=trt.float32),
        trtp.from_shape_expr((batch, heads, splits), dtype=trt.float32),
    )


def _describe_masked_split(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    mask: trtp.TensorDesc,
    scale: float,
    splits: int,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    return _describe_split(q, k, v, scale, splits)


def _describe_combine(
    o: trtp.TensorDesc, lse: trtp.TensorDesc, dtype: str
) -> trtp.TensorDesc:
    batch, heads, head_dim = (o.shape_expr[i] for i in (0, 1, 3))
    return trtp.from_shape_expr(
        (batch, heads, 1, head_dim), dtype=_OUTPUT_DTYPES[dtype]
    )


def _split_launch(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    mask: Optional[trtp.TensorDesc],
    scale: float,
    splits: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    heads, head_dim = (_common.constant_int(q.shape_expr[i]) for i in (1, 3))
    kv_heads = _common.constant_int(k.shape_expr[1])
    mask_pointer, mask_batch, mask_heads = None, False, 1
    if mask is not None:
        mask_pointer = "*" + _TRITON_DTYPES[mask.dtype]
        mask_batch = _common.constant_int(mask.shape_expr[0]) != 1
        mask_heads = _common.constant_int(mask.shape_expr[1])
    artifact = _split_artifact(
        "*" + _TRITON_DTYPES[q.dtype],
        mask_pointer,
        heads,
        kv_heads,
        head_dim,
        mask_batch,
        mask_heads,
        splits,
        scale,
    )
    params = trtp.KernelLaunchParams(
        grid_x=splits,
        grid_y=q.shape_expr[0] * kv_heads,
        grid_z=1,
        block_x=artifact.num_warps * 32,
        block_y=1,
        block_z=1,
        shared_mem=artifact.shared_mem,
    )
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(k.shape_expr[2])
    return artifact.kernel_name, artifact.ptx, params, extra_args


def _aot_split(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    scale: float,
    splits: int,
    outputs: Tuple[trtp.TensorDesc, trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    return _split_launch(q, k, None, scale, splits)


def _aot_masked_split(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    mask: trtp.TensorDesc,
    scale: float,
    splits: int,
    outputs: Tuple[trtp.TensorDesc, trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    return _split_launch(q, k, mask, scale, splits)


def _aot_combine(
    o: trtp.TensorDesc,
    lse: trtp.TensorDesc,
    dtype: str,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    splits, head_dim = (_common.constant_int(o.shape_expr[i]) for i in (2, 3))
    artifact = _combine_artifact("*" + dtype, head_dim, splits)
    params = trtp.KernelLaunchParams(
        grid_x=o.shape_expr[0] * o.shape_expr[1],
        grid_y=1,
        grid_z=1,
        block_x=artifact.num_warps * 32,
        block_y=1,
        block_z=1,
        shared_mem=artifact.shared_mem,
    )
    return artifact.kernel_name, artifact.ptx, params, trtp.SymIntExprs(0)


def plugins() -> Tuple[Any, Any, Any]:
    """The split, masked split and combine plugins, registered with TensorRT on first use."""
    global _registered
    with _lock:
        if not _registered:
            trtp.register(SPLIT_PLUGIN_NAME)(_describe_split)
            trtp.aot_impl(SPLIT_PLUGIN_NAME)(_aot_split)
            trtp.register(MASKED_SPLIT_PLUGIN_NAME)(_describe_masked_split)
            trtp.aot_impl(MASKED_SPLIT_PLUGIN_NAME)(_aot_masked_split)
            trtp.register(COMBINE_PLUGIN_NAME)(_describe_combine)
            trtp.aot_impl(COMBINE_PLUGIN_NAME)(_aot_combine)
            _registered = True
    ops = trtp.op.torch_tensorrt
    return (
        ops.decode_attention_split,
        ops.decode_attention_split_masked,
        ops.decode_attention_combine,
    )
