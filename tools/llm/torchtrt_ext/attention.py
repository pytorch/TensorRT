# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# copied from https://gitlab-master.nvidia.com/cuda-python/cuda-python-tile-compiler/-/raw/main/test/kernels/attention.py?ref_type=heads

import math

import cuda.tile as ct
import numpy as np
from cuda.tile.numeric_semantics import RoundingMode as RMd

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel(occupancy=2)
def fmha_kernel(
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    input_pos: int,
    TILE_D: ct.Constant[int],  # TILE_D = hidden_size
    H: ct.Constant[int],
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    QUERY_GROUP_SIZE: ct.Constant[int],
    CAUSAL: ct.Constant[bool],
    EVEN_K: ct.Constant[bool],
):
    bid_x = ct.bid(0)  # int
    bid_y = ct.bid(1)  # int
    batch_idx = bid_y // H  # int
    head_idx = bid_y % H  # int
    off_kv_h = head_idx // QUERY_GROUP_SIZE  # int
    qk_scale = qk_scale * INV_LOG_2
    # init offset
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # initialize m, l, acc
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    # load q
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)
    ).reshape(
        (TILE_M, TILE_D)
    )  # [TILE_M, TILE_D]

    # loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M  # int
    k_seqlen = K.shape[2]  # int
    if CAUSAL:
        # when kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # when kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdivi(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdivi(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    for j in range(0, Tc):
        # -- compute qk ----
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool)
            # out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -np.inf)  # [TILE_M, TILE_N]
            qk += mask
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # scale acc
        acc = acc * alpha  # [TILE_M, TILE_N]
        # compute pv
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape(
            (TILE_N, TILE_D)
        )  # [TILE_N, TILE_D]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)
