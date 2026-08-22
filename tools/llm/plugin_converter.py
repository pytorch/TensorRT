"""
TensorRT converter for Edge-LLM attention plugin ops.

This module contains the TensorRT converter for the tensorrt_edge_llm::xqa_attn
custom op. It is kept in a separate file from plugin_utils.py for maintainability.
"""

import numpy as np
import tensorrt as trt
from plugin_utils import get_plugin_config, register_plugin_op
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

# Ensure the custom op is registered before the converter decorator runs
register_plugin_op()

import torch  # noqa: E402 (must be after register_plugin_op so the op exists)


@dynamo_tensorrt_converter(
    torch.ops.tensorrt_edge_llm.xqa_attn.default, supports_dynamic_shapes=True
)
def convert_attn(ctx: ConversionContext, target, args, kwargs, name):
    """
    Convert tensorrt_edge_llm::xqa_attn op to TensorRT AttentionPlugin.

    TensorRT-Edge-LLM (0.4.0) plugin requires 5 inputs:
    - qkv, kv, ctx_len, rope, kv_cache_start_idx

    Plugin fields:
    - num_q_heads, num_kv_heads, head_size, enable_tree_attention, enable_delta_kv_output
    """
    # args: qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d
    qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d = args[:8]

    creator = trt.get_plugin_registry().get_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry!")

    # Get config from global settings
    config = get_plugin_config()
    if config:
        nq_val = config["num_attention_heads"]
        nkv_val = config["num_key_value_heads"]
        d_val = config["head_dim"]
    else:
        # Fallback to values from args (may not work correctly)
        nq_val = nq if isinstance(nq, int) else 14
        nkv_val = nkv if isinstance(nkv, int) else 2
        d_val = d if isinstance(d, int) else 64

    # Plugin fields for TensorRT-Edge-LLM AttentionPlugin
    # Required: num_q_heads, num_kv_heads, head_size, enable_tree_attention
    # enable_delta_kv_output=1 enables delta KV output for Python/torch_tensorrt compatibility
    field_list = [
        trt.PluginField(
            field_name, np.array([field_val], dtype=np.int32), trt.PluginFieldType.INT32
        )
        for field_name, field_val in [
            ("num_q_heads", nq_val),
            ("num_kv_heads", nkv_val),
            ("head_size", d_val),
            ("enable_tree_attention", 0),
            ("enable_delta_kv_output", 1),
        ]
    ]

    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)

    # 5 inputs for release version: qkv, kv, ctx_len, rope, kv_cache_start_idx
    inputs = [
        (
            get_trt_tensor(ctx, i, f"{name}_i{idx}")
            if not isinstance(i, trt.ITensor)
            else i
        )
        for idx, i in enumerate([qkv, kv, ctx_len, rope, kv_cache_start_idx])
    ]

    # Handle ctx_len shape if needed (squeeze if [B, 1] -> [B])
    if len(inputs[2].shape) == 2 and inputs[2].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[2])
        shuffle_layer.reshape_dims = (inputs[2].shape[0],)
        inputs[2] = shuffle_layer.get_output(0)

    # Handle kv_cache_start_idx shape if needed (squeeze if [B, 1] -> [B])
    if len(inputs[4].shape) == 2 and inputs[4].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[4])
        shuffle_layer.reshape_dims = (inputs[4].shape[0],)
        inputs[4] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    return layer.get_output(0), layer.get_output(1)
