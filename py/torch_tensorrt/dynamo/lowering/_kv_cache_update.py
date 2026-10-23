# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# mypy: disallow-untyped-decorators=False

"""``tensorrt::kv_cache_update``: a KV-cache write with a position per batch row.

``IKVCacheUpdateLayer`` writes ``update[b]`` from its own ``writeIndices[b]`` in
every batch row ``b``, but the aten writes the KV-cache fast path matches cannot
say that: ``slice_scatter`` takes one constant start and ``index_copy`` one index
shared by every row. Batched decoding, where each sequence is at its own
position, writes its cache with this op instead::

    # cache [b, num_heads, s_max, head_dim], update [b, num_heads, s, head_dim],
    # write_indices [b] (int32 / int64): positions write_indices[b] ..
    # write_indices[b] + s - 1 of row b take update[b].
    new_cache = torch.ops.tensorrt.kv_cache_update(cache, update, write_indices)

The op is functional. When ``cache`` is an input of the engine -- a cache passed
to the compiled module, or a mutated buffer lifted to an input -- it converts to
an ``IKVCacheUpdateLayer``, which writes the cache in place (aliased I/O), so a
buffer written as ``self.cache.copy_(torch.ops.tensorrt.kv_cache_update(...))``
needs no copy back. Otherwise it converts to a scatter.

Positions must satisfy ``write_indices[b] + s <= s_max``; like the layer, the op
does not check them.
"""

import torch


def _check_args(
    cache: torch.Tensor, update: torch.Tensor, write_indices: torch.Tensor
) -> None:
    torch._check(
        cache.dim() == 4,
        lambda: f"kv_cache_update: cache must be 4-D [b, num_heads, s_max, head_dim], "
        f"got {tuple(cache.shape)}",
    )
    torch._check(
        update.dim() == 4
        and update.shape[0] == cache.shape[0]
        and update.shape[1] == cache.shape[1]
        and update.shape[3] == cache.shape[3],
        lambda: f"kv_cache_update: update must be [b, num_heads, s, head_dim] like the "
        f"cache {tuple(cache.shape)}, got {tuple(update.shape)}",
    )
    torch._check(
        update.dtype == cache.dtype,
        lambda: f"kv_cache_update: update dtype {update.dtype} differs from the "
        f"cache's {cache.dtype}",
    )
    torch._check(
        write_indices.dim() == 1
        and write_indices.shape[0] == cache.shape[0]
        and write_indices.dtype in (torch.int32, torch.int64),
        lambda: "kv_cache_update: write_indices must be an int32 / int64 tensor of "
        f"shape [{cache.shape[0]}], got {write_indices.dtype} {tuple(write_indices.shape)}",
    )


@torch.library.custom_op("tensorrt::kv_cache_update", mutates_args=())
def kv_cache_update(
    cache: torch.Tensor, update: torch.Tensor, write_indices: torch.Tensor
) -> torch.Tensor:
    """``cache`` with ``update[b]`` written at positions ``write_indices[b]`` ..
    ``write_indices[b] + s - 1`` of row ``b`` (dim 2)."""
    _check_args(cache, update, write_indices)
    positions = write_indices.to(torch.int64)[:, None] + torch.arange(
        update.shape[2], device=update.device
    )
    return cache.scatter(2, positions[:, None, :, None].expand_as(update), update)


@kv_cache_update.register_fake
def _(
    cache: torch.Tensor, update: torch.Tensor, write_indices: torch.Tensor
) -> torch.Tensor:
    _check_args(cache, update, write_indices)
    return torch.empty_like(cache)


tensorrt_kv_cache_update_op = torch.ops.tensorrt.kv_cache_update.default
