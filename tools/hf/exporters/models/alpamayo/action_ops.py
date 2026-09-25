"""Action-engine operations required by TensorRT-Edge-LLM's runtime ABI."""

from __future__ import annotations

import torch
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor


@torch.library.custom_op("edge_export::action_kv_cache_update", mutates_args=())
def action_kv_cache_update(
    cache: torch.Tensor,
    values: torch.Tensor,
    write_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference linear KV-cache update with one start index per batch row."""
    output = cache.clone()
    update_length = int(values.shape[2])
    for batch_index in range(int(cache.shape[0])):
        start = int(write_indices[batch_index].item())
        output[
            batch_index,
            :,
            start : start + update_length,
            :,
        ] = values[batch_index]
    return output


@action_kv_cache_update.register_fake
def _action_kv_cache_update_fake(
    cache: torch.Tensor,
    values: torch.Tensor,
    write_indices: torch.Tensor,
) -> torch.Tensor:
    del values, write_indices
    return torch.empty_like(cache)


@dynamo_tensorrt_converter(
    torch.ops.edge_export.action_kv_cache_update.default,
    supports_dynamic_shapes=True,
)
def convert_action_kv_cache_update(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
):
    """Lower action KV updates to TensorRT's aliased IKVCacheUpdateLayer."""
    del target, kwargs
    from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
        emit_kv_cache_update_layer,
    )

    cache = get_trt_tensor(ctx, args[0], f"{name}_cache")
    values = get_trt_tensor(ctx, args[1], f"{name}_values")
    write_indices = get_trt_tensor(
        ctx,
        args[2],
        f"{name}_write_indices",
        dtype=torch.int32,
    )
    output = emit_kv_cache_update_layer(
        ctx,
        name,
        cache,
        values,
        write_indices,
    )
    if output is None:
        raise RuntimeError(
            "Action KV cache update requires the cache to be a direct engine input"
        )
    return output
