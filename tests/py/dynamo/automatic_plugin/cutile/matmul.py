# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

# Define a type alias for Constant integers.
# This makes kernel signatures cleaner and indicates that these parameters
# are compile-time constants, which cuTile uses for optimization.
ConstInt = ct.Constant[int]


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(
    A,
    B,
    C,
    tm: ConstInt,  # Tile size along M dimension (rows of C)
    tn: ConstInt,  # Tile size along N dimension (columns of C)
    tk: ConstInt,
):  # Tile size along K dimension (inner product dimension)
    """
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each CUDA thread block (CTA)
    computes a `tm` x `tn` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `tk`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `tk` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(
            dtype
        )

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(
            dtype
        )

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.kernel
def matmul_split_k_kernel(
    A, B, C, LOCKS, COUNTS, tm: ConstInt, tn: ConstInt, tk: ConstInt, SPLIT_K: ConstInt
):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    bidz = ct.bid(1)

    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(bidz, num_tiles, SPLIT_K):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(
            dtype
        )
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(
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
    count = ct.gather(COUNTS, count_offset)
    if count == 0:
        ct.store(C, index=(bidx, bidy), tile=sum)
    else:
        curr = ct.load(C, index=(bidx, bidy), shape=(tm, tn))
        ct.store(C, index=(bidx, bidy), tile=(curr + sum))
    ct.scatter(COUNTS, count_offset, (count + 1) % SPLIT_K)
    ct.atomic_xchg(LOCKS, lock_offset, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Each thread block computes one (tm x tn) tile for a specific batch item.
    The grid is 3D: (Batch_idx, M_tile_idx, N_tile_idx).
    """
    pid_batch = ct.bid(0)  # Batch dimension
    pidx = ct.bid(1)  # M dimension
    pidy = ct.bid(2)  # N dimension

    # Calculate number of K tiles
    # A is (Batch, M, K), so K is axis 2
    # Use A.shape[2] for the total K dimension and ct.cdiv for ceiling division
    num_k_tiles = ct.cdiv(A.shape[2], tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        # A is (Batch, M, K), load (1, tm, tk) tile
        a = ct.load(
            A, index=(pid_batch, pidx, k), shape=(1, tm, tk), padding_mode=zero_pad
        )
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D for ct.mma

        # B is (Batch, K, N), load (1, tk, tn) tile
        b = ct.load(
            B, index=(pid_batch, k, pidy), shape=(1, tk, tn), padding_mode=zero_pad
        )
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D for ct.mma

        accumulator = ct.mma(a, b, acc=accumulator)

    # Convert to output dtype and store
    result = ct.astype(accumulator, C.dtype)
    # Store with 3D index and 3D shape, C is (Batch, M, N)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, pidx, pidy), tile=result_3d)
