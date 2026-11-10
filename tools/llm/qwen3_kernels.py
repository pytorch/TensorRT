# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-written Triton kernels for Qwen3 batch-1 decode blocks.

Every kernel computes one output vector of a decoder block for a single token:
the activation is a contiguous FP16 row vector, weights are contiguous
row-major FP16 ``[N, K]`` matrices (``nn.Linear`` layout), products accumulate
in FP32, and outputs are rounded to FP16 once.

The kernels are launched as TensorRT AOT plugins through
``torch_tensorrt.region.fused_kernel``. That path compiles them with
``triton.compiler.ASTSource`` without Triton's usual pointer-alignment
specialization, so the GEMV and norm loads carry explicit 16-byte
``tl.multiple_of`` hints: without them Triton emits 2-byte loads. (The small
per-head loads of the qk_rope kernel are unhinted.) Torch, TensorRT and Myelin
allocate every tensor these blocks see on (at least) 256-byte boundaries and
every row is a multiple of 16 bytes long.

Weight tiles load with ``EVICT=1`` (``ld.global.L1::evict_first.L2::cache_hint``
with an L2 evict_first policy). With the default policy the streamed weights
evict the L2-resident KV cache and small tensors that TensorRT's own kernels
rely on: a default-policy lm_head kernel alone cost +97 us/token (1.280 vs
1.183 ms) although the kernel itself was as fast as TensorRT's GEMV.

Runtime parameter order follows the plugin ABI:
``(input pointers..., i32 scalars..., output pointers...)``; there are no
scalars, all sizes are ``tl.constexpr``.

The kernels sit inside ``EDITABLE`` fences: ``tools/llm/agentic_decode.py``
declares this file as an editable file, so an Auto-Tuner session may edit
them (each edit is compiled, checked against the FP32 reference and
measured in the model like any other candidate).
"""

import triton
import triton.language as tl

# EDITABLE-START


@triton.jit
def _vec(ptr, offs):
    """Load a contiguous FP16 vector slice as FP32 (16-byte aligned)."""
    return tl.load(tl.multiple_of(ptr + offs, [16])).to(tl.float32)


@triton.jit
def _tile(w_ptr, rows, cols, K: tl.constexpr, EVICT: tl.constexpr):
    """Load a ``[len(rows), len(cols)]`` FP16 tile of a row-major ``[N, K]`` matrix."""
    ptrs = tl.multiple_of(w_ptr + rows[:, None] * K + cols[None, :], [16, 16])
    if EVICT:
        return tl.load(ptrs, eviction_policy="evict_first")
    else:
        return tl.load(ptrs)


@triton.jit
def _rmsnorm(x, nw, K: tl.constexpr, EPS: tl.constexpr):
    """RMSNorm of an FP32 row (HF Qwen3RMSNorm, kept in FP32)."""
    var = tl.sum(x * x, axis=0) / K
    return x * tl.rsqrt(var + EPS) * nw


# --------------------------------------------------------------------------
# RMSNorm(x) @ W^T                                   (final norm + lm_head,
#                                                     input norm + fused QKV)
# --------------------------------------------------------------------------
@triton.jit
def norm_gemv_kernel(
    x_ptr,
    nw_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    # Issue the weight stream first; the norm overlaps its latency.
    w = _tile(w_ptr, rows, cols, K, EVICT)
    x = _vec(x_ptr, cols)
    nw = _vec(nw_ptr, cols)
    xn = _rmsnorm(x, nw, K, EPS)
    acc = tl.sum(w.to(tl.float32) * xn[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


# --------------------------------------------------------------------------
# RMSNorm(x) @ [Wq; Wk; Wv]^T as three outputs          (input norm + QKV)
#
# Separate q/k/v outputs instead of one fused qkv vector: TensorRT cannot fuse
# the split of a plugin output into the q/k norms that follow, and inserts a
# slice-copy kernel for each.
# --------------------------------------------------------------------------
@triton.jit
def _norm_rows(
    x_ptr,
    nw_ptr,
    w_ptr,
    y_ptr,
    row0,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    rows = row0 + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    w = _tile(w_ptr, rows, cols, K, EVICT)
    xn = _rmsnorm(_vec(x_ptr, cols), _vec(nw_ptr, cols), K, EPS)
    acc = tl.sum(w.to(tl.float32) * xn[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


@triton.jit
def norm_qkv_kernel(
    x_ptr,
    nw_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    PQ: tl.constexpr = NQ // BLOCK_N
    PK: tl.constexpr = NKV // BLOCK_N
    if pid < PQ:
        _norm_rows(x_ptr, nw_ptr, wq_ptr, q_ptr, pid * BLOCK_N, K, BLOCK_N, EPS, EVICT)
    elif pid < PQ + PK:
        _norm_rows(
            x_ptr, nw_ptr, wk_ptr, k_ptr, (pid - PQ) * BLOCK_N, K, BLOCK_N, EPS, EVICT
        )
    else:
        _norm_rows(
            x_ptr,
            nw_ptr,
            wv_ptr,
            v_ptr,
            (pid - PQ - PK) * BLOCK_N,
            K,
            BLOCK_N,
            EPS,
            EVICT,
        )


@triton.jit
def _add_norm_rows(
    a_ptr,
    b_ptr,
    nw_ptr,
    w_ptr,
    r_ptr,
    y_ptr,
    row0,
    STORE_R: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    rows = row0 + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    w = _tile(w_ptr, rows, cols, K, EVICT)
    r = (_vec(a_ptr, cols) + _vec(b_ptr, cols)).to(tl.float16)
    if STORE_R:
        tl.store(tl.multiple_of(r_ptr + cols, [16]), r)
    xn = _rmsnorm(r.to(tl.float32), _vec(nw_ptr, cols), K, EPS)
    acc = tl.sum(w.to(tl.float32) * xn[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


@triton.jit
def add_norm_qkv_kernel(
    a_ptr,
    b_ptr,
    nw_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    r_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    PQ: tl.constexpr = NQ // BLOCK_N
    PK: tl.constexpr = NKV // BLOCK_N
    if pid == 0:
        _add_norm_rows(
            a_ptr, b_ptr, nw_ptr, wq_ptr, r_ptr, q_ptr, 0, True, K, BLOCK_N, EPS, EVICT
        )
    elif pid < PQ:
        _add_norm_rows(
            a_ptr,
            b_ptr,
            nw_ptr,
            wq_ptr,
            r_ptr,
            q_ptr,
            pid * BLOCK_N,
            False,
            K,
            BLOCK_N,
            EPS,
            EVICT,
        )
    elif pid < PQ + PK:
        _add_norm_rows(
            a_ptr,
            b_ptr,
            nw_ptr,
            wk_ptr,
            r_ptr,
            k_ptr,
            (pid - PQ) * BLOCK_N,
            False,
            K,
            BLOCK_N,
            EPS,
            EVICT,
        )
    else:
        _add_norm_rows(
            a_ptr,
            b_ptr,
            nw_ptr,
            wv_ptr,
            r_ptr,
            v_ptr,
            (pid - PQ - PK) * BLOCK_N,
            False,
            K,
            BLOCK_N,
            EPS,
            EVICT,
        )


# --------------------------------------------------------------------------
# r = a + b ; RMSNorm(r) @ W^T          (residual add + norm + GEMV; r kept)
# --------------------------------------------------------------------------
@triton.jit
def add_norm_gemv_kernel(
    a_ptr,
    b_ptr,
    nw_ptr,
    w_ptr,
    r_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    w = _tile(w_ptr, rows, cols, K, EVICT)
    # The residual stream is FP16 in the model: round the sum once.
    r = (_vec(a_ptr, cols) + _vec(b_ptr, cols)).to(tl.float16)
    if pid == 0:
        tl.store(tl.multiple_of(r_ptr + cols, [16]), r)
    xn = _rmsnorm(r.to(tl.float32), _vec(nw_ptr, cols), K, EPS)
    acc = tl.sum(w.to(tl.float32) * xn[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


# RMSNorm(a + b) @ W^T without keeping the sum      (last residual + final norm
#                                                     + lm_head)
@triton.jit
def add_norm_gemv_out_kernel(
    a_ptr,
    b_ptr,
    nw_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    w = _tile(w_ptr, rows, cols, K, EVICT)
    r = (_vec(a_ptr, cols) + _vec(b_ptr, cols)).to(tl.float16)
    xn = _rmsnorm(r.to(tl.float32), _vec(nw_ptr, cols), K, EPS)
    acc = tl.sum(w.to(tl.float32) * xn[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


# --------------------------------------------------------------------------
# act = silu(RMSNorm(x) @ Wg^T) * (RMSNorm(x) @ Wu^T)        (MLP input half)
# --------------------------------------------------------------------------
@triton.jit
def norm_gateup_kernel(
    x_ptr,
    nw_ptr,
    wg_ptr,
    wu_ptr,
    act_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    wg = _tile(wg_ptr, rows, cols, K, EVICT)
    wu = _tile(wu_ptr, rows, cols, K, EVICT)
    xn = _rmsnorm(_vec(x_ptr, cols), _vec(nw_ptr, cols), K, EPS)
    g = tl.sum(wg.to(tl.float32) * xn[None, :], axis=1)
    u = tl.sum(wu.to(tl.float32) * xn[None, :], axis=1)
    act = g * tl.sigmoid(g) * u
    tl.store(act_ptr + rows, act.to(tl.float16))


@triton.jit
def add_norm_gateup_kernel(
    a_ptr,
    b_ptr,
    nw_ptr,
    wg_ptr,
    wu_ptr,
    r_ptr,
    act_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, K)
    wg = _tile(wg_ptr, rows, cols, K, EVICT)
    wu = _tile(wu_ptr, rows, cols, K, EVICT)
    r = (_vec(a_ptr, cols) + _vec(b_ptr, cols)).to(tl.float16)
    if pid == 0:
        tl.store(tl.multiple_of(r_ptr + cols, [16]), r)
    xn = _rmsnorm(r.to(tl.float32), _vec(nw_ptr, cols), K, EPS)
    g = tl.sum(wg.to(tl.float32) * xn[None, :], axis=1)
    u = tl.sum(wu.to(tl.float32) * xn[None, :], axis=1)
    act = g * tl.sigmoid(g) * u
    tl.store(act_ptr + rows, act.to(tl.float16))


# --------------------------------------------------------------------------
# y = res + a @ W^T                    (o_proj + residual, down + residual)
# --------------------------------------------------------------------------
@triton.jit
def gemv_add_kernel(
    a_ptr,
    w_ptr,
    res_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        w = _tile(w_ptr, rows, cols, K, EVICT)
        a = _vec(a_ptr, cols)
        acc += w.to(tl.float32) * a[None, :]
    y = tl.sum(acc, axis=1)
    y = y + tl.load(res_ptr + rows).to(tl.float32)
    tl.store(y_ptr + rows, y.to(tl.float16))


# --------------------------------------------------------------------------
# q/k per-head RMSNorm + rotary embedding                     (attention prep)
#
# One program per q or k head. Outputs are the rotated q and k in
# [heads, head_dim] order (= [1, heads, 1, head_dim] for one token); the KV
# cache write and attention stay in TensorRT.
# --------------------------------------------------------------------------
@triton.jit
def _norm_rope_head(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + d).to(tl.float32)
    x2 = tl.load(x_ptr + HALF + d).to(tl.float32)
    w1 = tl.load(w_ptr + d).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
    c1 = tl.load(cos_ptr + d).to(tl.float32)
    c2 = tl.load(cos_ptr + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + d).to(tl.float32)
    s2 = tl.load(sin_ptr + HALF + d).to(tl.float32)
    r = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + EPS)
    n1 = x1 * r * w1
    n2 = x2 * r * w2
    tl.store(o_ptr + d, ((n1 * c1 - n2 * s1) * SCALE).to(tl.float16))
    tl.store(o_ptr + HALF + d, ((n2 * c2 + n1 * s2) * SCALE).to(tl.float16))


@triton.jit
def qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    QSCALE: tl.constexpr,
):
    """QSCALE pre-multiplies q (the attention softmax scale folded into q)."""
    pid = tl.program_id(0)
    if pid < HQ:
        _norm_rope_head(
            q_ptr + pid * D, qw_ptr, cos_ptr, sin_ptr, qo_ptr + pid * D, D, EPS, QSCALE
        )
    else:
        _norm_rope_head(
            k_ptr + (pid - HQ) * D,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            ko_ptr + (pid - HQ) * D,
            D,
            EPS,
            1.0,
        )


# EDITABLE-END
