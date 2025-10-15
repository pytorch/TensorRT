# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# copied from https://gitlab-master.nvidia.com/cuda-python/cuda-python-tile-compiler/-/blob/main/test/kernels/matmul.py?ref_type=heads

import cuda.tile as ct
from cuda.tile.by_target import ByTarget

ConstInt = ct.Constant[int]


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    bid = ct.bid(0)
    num_bid_m = ct.cdivi(M, tm)
    num_bid_n = ct.cdivi(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ByTarget(sm_100=2))
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    num_tiles = ct.dim(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingValue.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_value=zero_pad).astype(
            dtype
        )
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_value=zero_pad).astype(
            dtype
        )
        sum = ct.mma(a, b, sum)

    sum = ct.astype(sum, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=sum)


@ct.kernel
def matmul_split_k_kernel(
    A, B, C, LOCKS, COUNTS, tm: ConstInt, tn: ConstInt, tk: ConstInt, SPLIT_K: ConstInt
):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    bidz = ct.bid(1)

    num_tiles = ct.dim(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingValue.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(bidz, num_tiles, SPLIT_K):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_value=zero_pad).astype(
            dtype
        )
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_value=zero_pad).astype(
            dtype
        )
        sum = ct.mma(a, b, sum)

    sum = ct.astype(sum, C.dtype)
    lock_offset = ct.bid(0)
    count_offset = lock_offset
    while (
        ct.atomic_cas(LOCKS, lock_offset, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE)
        == 1
    ):
        pass
    count = ct.load_offset(COUNTS, count_offset)
    if count == 0:
        ct.store(C, index=(bidx, bidy), tile=sum)
    else:
        curr = ct.load(C, index=(bidx, bidy), shape=(tm, tn))
        ct.store(C, index=(bidx, bidy), tile=(curr + sum))
    ct.store_offset(COUNTS, count_offset, (count + 1) % SPLIT_K)
    ct.atomic_xchg(LOCKS, lock_offset, 0, memory_order=ct.MemoryOrder.RELEASE)
