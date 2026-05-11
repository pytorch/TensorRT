"""
TensorRT converter for ViT attention plugin ops.

This module contains the TensorRT converter for the tensorrt_edge_llm::xqa_attn
custom op. It is kept in a separate file from plugin_utils.py for maintainability.
"""

import numpy as np
import tensorrt as trt

from plugin_utils_vit import get_vit_plugin_config, register_vit_plugin_op
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)

from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

register_vit_plugin_op()

import torch # noqa: E402 (must be after register_vit_plugin_op so the op exists)


def _as_int(value, default):
    """Return value as a Python int when export preserved it as a scalar."""
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_mask_type(mask_arg, default):
    """Infer cu_seqlens mode when the fourth plugin input is an INT32 tensor."""
    dtype = getattr(mask_arg, "dtype", None)
    dtype_name = str(dtype).lower()
    if (
        dtype == torch.int32
        or dtype == trt.int32
        or dtype == trt.DataType.INT32
        or "int32" in dtype_name
    ):
        return 1
    return default


@dynamo_tensorrt_converter(
    torch.ops.tensorrt_vit.attention.default, supports_dynamic_shapes=True
)
def convert_vit_attention(ctx: ConversionContext, target, args, kwargs, name):
    """Convert tensorrt_vit::attention to TensorRT ViTAttentionPlugin."""
    qkv, cos, sin, mask_or_cu_seqlens, num_heads, head_dim = args[:6]
    qkv_fused = args[6] if len(args) > 6 else kwargs.get("qkv_fused", 1)
    mask_type = args[7] if len(args) > 7 else kwargs.get("mask_type", 0)
    max_seq_len = args[8] if len(args) > 8 else kwargs.get("max_seq_len", 0)

    creator = trt.get_plugin_registry().get_plugin_creator(
        "ViTAttentionPlugin", "1", ""
    )
    if creator is None:
        raise RuntimeError(
            "ViTAttentionPlugin not found in TensorRT plugin registry!"
        )

    qkv_tensor = (
        get_trt_tensor(ctx, qkv, f"{name}_qkv")
        if not isinstance(qkv, trt.ITensor)
        else qkv
    )
    cos_tensor = (
        get_trt_tensor(ctx, cos, f"{name}_cos")
        if not isinstance(cos, trt.ITensor)
        else cos
    )
    sin_tensor = (
        get_trt_tensor(ctx, sin, f"{name}_sin")
        if not isinstance(sin, trt.ITensor)
        else sin
    )
    mask_tensor = (
        get_trt_tensor(ctx, mask_or_cu_seqlens, f"{name}_mask")
        if not isinstance(mask_or_cu_seqlens, trt.ITensor)
        else mask_or_cu_seqlens
    )

    config = get_vit_plugin_config()
    num_heads_val = config.get("num_attention_heads", num_heads)
    head_dim_val = config.get("head_dim", head_dim)
    qkv_fused_val = _as_int(qkv_fused, 1)
    inferred_mask_type = _infer_mask_type(mask_tensor, config.get("mask_type", 0))
    mask_type_val = _as_int(mask_type, inferred_mask_type)
    if inferred_mask_type == 1:
        mask_type_val = 1
    max_seq_len_val = _as_int(max_seq_len, config.get("max_seq_len", 0))
    field_list = [
        trt.PluginField(
            "num_heads",
            np.array([num_heads_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "head_size",
            np.array([head_dim_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "qkv_fused",
            np.array([qkv_fused_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "mask_type",
            np.array([mask_type_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "max_seq_len",
            np.array([max_seq_len_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
    ]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    layer = ctx.net.add_plugin_v2([qkv_tensor, cos_tensor, sin_tensor, mask_tensor], plugin)
    layer.name = name
    return layer.get_output(0)
