"""
TensorRT converters for Edge-LLM plugin custom ops.

Attention: separate Q/K/V into AttentionPlugin (no fused-qkv slice), plus
``context_attention_mask_type``. Also lowers ``trt::causal_conv1d``,
``trt::update_ssm_state``, and ``trt::nvfp4_moe_plugin`` onto the matching
IPluginV3 creators from libNvInfer_edgellm_plugin.so.
"""

import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion._ConversionContext import AliasedOutput, AliasKind
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

from .attention import ContextAttentionMaskType
from .plugin_utils import (
    attention_plugin_uses_paged_kv,
    get_trt_plugin_creator,
    mamba_plugin_uses_ragged,
)


def _creator_is_v3(creator) -> bool:
    return "V3" in type(creator).__name__


def _create_trt_plugin(
    creator, name: str, field_list: list
) -> trt.IPluginV2 | trt.IPluginV3:
    fields = trt.PluginFieldCollection(field_list)
    if _creator_is_v3(creator):
        return creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
    return creator.create_plugin(name, fields)


def _plugin_is_v3(plugin) -> bool:
    return "V3" in type(plugin).__name__


def _add_plugin_layer(ctx: ConversionContext, inputs: list, plugin, name: str):
    layer = (
        ctx.net.add_plugin_v3(inputs, [], plugin)
        if _plugin_is_v3(plugin)
        else ctx.net.add_plugin_v2(inputs, plugin)
    )
    layer.name = name
    return layer


@dynamo_tensorrt_converter(
    torch.ops.trt.attention_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_llm_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, kv, ctx_len, rope, kv_cache_start_idx = args[:7]
    num_q_heads = args[7]
    num_kv_heads = args[8]
    enable_tree_attention = args[9]
    head_size = args[10]
    enable_fp8_kv_cache = args[11]
    sliding_window_size = args[12] if len(args) > 12 else -1
    context_attention_mask_type = (
        int(args[13]) if len(args) > 13 else int(ContextAttentionMaskType.CAUSAL)
    )
    attention_mask = args[14] if len(args) > 14 else None
    position_ids = args[15] if len(args) > 15 else None
    qkv_scales = args[16] if len(args) > 16 else None

    creator = get_trt_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry")
    if attention_plugin_uses_paged_kv():
        raise NotImplementedError(
            "The loaded Edge-LLM AttentionPlugin uses the >=0.10 paged-KV API, which this "
            "converter does not support yet; use an Edge-LLM <=0.9.1 plugin build."
        )
    if context_attention_mask_type not in (
        int(ContextAttentionMaskType.CAUSAL),
        int(ContextAttentionMaskType.SLIDING_OR_CHUNKED_CAUSAL),
    ):
        raise NotImplementedError(
            f"{name}: context_attention_mask_type="
            f"{ContextAttentionMaskType(context_attention_mask_type).name} needs Edge-LLM "
            ">=0.10 (context mask selector); the <=0.9.1 AttentionPlugin always applies "
            "a causal context mask."
        )

    field_list = [
        trt.PluginField(
            field_name,
            np.array([field_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        )
        for field_name, field_val in [
            ("num_q_heads", int(num_q_heads)),
            ("num_kv_heads", int(num_kv_heads)),
            ("head_size", int(head_size)),
            ("enable_tree_attention", int(enable_tree_attention)),
            ("enable_fp8_kv_cache", int(enable_fp8_kv_cache)),
            ("sliding_window_size", int(sliding_window_size)),
        ]
    ]
    if bool(enable_fp8_kv_cache) and qkv_scales is not None:
        field_list.append(
            trt.PluginField(
                "qkv_scales",
                np.array(list(qkv_scales), dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            )
        )

    plugin = _create_trt_plugin(creator, name, field_list)
    if plugin is None:
        raise RuntimeError("Failed to create AttentionPlugin")

    plugin_inputs = [q, k, v, kv, ctx_len, rope, kv_cache_start_idx]
    if bool(enable_tree_attention):
        plugin_inputs.extend([attention_mask, position_ids])

    inputs = [
        (
            get_trt_tensor(ctx, tensor, f"{name}_i{idx}")
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        for idx, tensor in enumerate(plugin_inputs)
    ]

    kv_cache_start_idx_input_idx = 6
    if (
        len(inputs[kv_cache_start_idx_input_idx].shape) == 2
        and inputs[kv_cache_start_idx_input_idx].shape[1] == 1
    ):
        shuffle_layer = ctx.net.add_shuffle(inputs[kv_cache_start_idx_input_idx])
        shuffle_layer.reshape_dims = (inputs[kv_cache_start_idx_input_idx].shape[0],)
        inputs[kv_cache_start_idx_input_idx] = shuffle_layer.get_output(0)

    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.attention_plugin_paged.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_llm_attention_plugin_paged(
    ctx: ConversionContext, target, args, kwargs, name
):
    """Edge-LLM >=0.10 AttentionPlugin: packed QKV, paged KV pool, ragged metadata."""
    del target
    args = list(args) + [kwargs[k] for k in ("sliding_window_size",) if k in kwargs]
    qkv, kv_pool, query_lengths, rope_rows, kv_start, page_table, selector = args[:7]
    offsets, seq_lengths, phase_marker, ctx_carrier = args[7:11]
    num_q_heads, num_kv_heads, head_size = (int(a) for a in args[11:14])
    sliding_window_size = int(args[14]) if len(args) > 14 else -1

    creator = get_trt_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry")
    if not attention_plugin_uses_paged_kv():
        raise NotImplementedError(
            "trt::attention_plugin_paged needs an Edge-LLM >=0.10 AttentionPlugin build"
        )

    fields = [
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_size", head_size),
        ("enable_context_mask_selector", int(selector is not None)),
    ]
    if sliding_window_size > 0:
        fields.append(("sliding_window_size", sliding_window_size))
    field_list = [
        trt.PluginField(n, np.array([v], dtype=np.int32), trt.PluginFieldType.INT32)
        for n, v in fields
    ]
    plugin = _create_trt_plugin(creator, name, field_list)
    if plugin is None:
        raise RuntimeError("Failed to create AttentionPlugin")

    plugin_inputs = [qkv, kv_pool, query_lengths, rope_rows, kv_start, page_table]
    if selector is not None:
        plugin_inputs.append(selector)
    plugin_inputs += [offsets, seq_lengths, phase_marker, ctx_carrier]
    inputs = [
        (
            tensor
            if isinstance(tensor, trt.ITensor)
            else get_trt_tensor(ctx, tensor, f"{name}_i{idx}")
        )
        for idx, tensor in enumerate(plugin_inputs)
    ]
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.vit_attention_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_vit_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, cu_seqlens, max_seqlen_carrier = args[:5]
    num_heads = args[5]
    head_size = args[6]

    creator = get_trt_plugin_creator("ViTAttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("ViTAttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            "num_heads",
            np.array([int(num_heads)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "head_size",
            np.array([int(head_size)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
    ]
    plugin = _create_trt_plugin(creator, name, field_list)
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    inputs = []
    for idx, tensor in enumerate([q, k, v, cu_seqlens, max_seqlen_carrier]):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not trt_tensor.name:
            trt_tensor.name = tensor_name
        inputs.append(trt_tensor)

    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    output = layer.get_output(0)
    if not output.name:
        output.name = f"{name}_output"
    return output


def _int_field(name: str, value: int) -> trt.PluginField:
    return trt.PluginField(
        name, np.array([int(value)], dtype=np.int32), trt.PluginFieldType.INT32
    )


def _float_field(name: str, value: float) -> trt.PluginField:
    return trt.PluginField(
        name, np.array([float(value)], dtype=np.float32), trt.PluginFieldType.FLOAT32
    )


def _as_plugin_inputs(ctx: ConversionContext, tensors: list, name: str) -> list:
    inputs = []
    for idx, tensor in enumerate(tensors):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not getattr(trt_tensor, "name", None):
            trt_tensor.name = tensor_name
        inputs.append(trt_tensor)
    return inputs


@dynamo_tensorrt_converter(
    torch.ops.trt.causal_conv1d.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_causal_conv1d(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    hidden_states, weight, bias, conv_state, context_lengths = args[:5]
    stride, padding, dilation, groups = args[5:9]

    creator = get_trt_plugin_creator("causal_conv1d", "1", "")
    if creator is None:
        raise RuntimeError("causal_conv1d plugin not found in TensorRT plugin registry")

    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _int_field("stride", stride),
            _int_field("padding", padding),
            _int_field("dilation", dilation),
            _int_field("groups", groups),
            _int_field("use_mtp", 0),
            _int_field("use_ddtree", 0),
        ],
    )
    if plugin is None:
        raise RuntimeError("Failed to create causal_conv1d plugin")

    inputs = _as_plugin_inputs(
        ctx, [hidden_states, weight, bias, conv_state, context_lengths], name
    )
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.update_ssm_state.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_update_ssm_state(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    hidden_states, ssm_a, ssm_b, ssm_c, ssm_d, dt, dt_bias, state, context_lengths = (
        args[:9]
    )
    dt_softplus, ngroups, nheads, head_dim, dstate = args[9:14]

    creator = get_trt_plugin_creator("update_ssm_state", "1", "")
    if creator is None:
        raise RuntimeError(
            "update_ssm_state plugin not found in TensorRT plugin registry"
        )

    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _int_field("dim", head_dim),
            _int_field("dstate", dstate),
            _int_field("nheads", nheads),
            _int_field("ngroups", ngroups),
            _int_field("dt_softplus", dt_softplus),
        ],
    )
    if plugin is None:
        raise RuntimeError("Failed to create update_ssm_state plugin")

    inputs = _as_plugin_inputs(
        ctx,
        [
            hidden_states,
            ssm_a,
            ssm_b,
            ssm_c,
            ssm_d,
            dt,
            dt_bias,
            state,
            context_lengths,
        ],
        name,
    )
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


def _alias_resident_state(
    ctx: ConversionContext, output: trt.ITensor, state: trt.ITensor, name: str
) -> None:
    """Bind a >=0.11 Mamba plugin's state output to its state input's storage.

    These plugins update resident state in place and fail ``enqueue`` unless the
    output pointer equals the input pointer. They do not declare the alias to TRT
    (the Edge-LLM runtime binds both to one pool), so declare it to Torch-TensorRT.
    """
    if not state.is_network_input:
        raise RuntimeError(
            f"{name}: resident state must be an engine input to alias it in place"
        )
    ctx.aliased_outputs.append(AliasedOutput(output, state.name, AliasKind.USER))


@dynamo_tensorrt_converter(
    torch.ops.trt.causal_conv1d_ragged.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_causal_conv1d_ragged(ctx: ConversionContext, target, args, kwargs, name):
    """Edge-LLM >=0.11 ``causal_conv1d``: token-major rows plus ragged metadata."""
    del target, kwargs
    args = list(args)
    tensors = args[:9]
    stride, padding, dilation, groups = args[9:13]

    creator = get_trt_plugin_creator("causal_conv1d", "1", "")
    if creator is None:
        raise RuntimeError("causal_conv1d plugin not found in TensorRT plugin registry")
    if not mamba_plugin_uses_ragged():
        raise NotImplementedError(
            "trt::causal_conv1d_ragged needs an Edge-LLM >=0.11 causal_conv1d build"
        )
    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _int_field("stride", stride),
            _int_field("padding", padding),
            _int_field("dilation", dilation),
            _int_field("groups", groups),
            _int_field("use_mtp", 0),
            _int_field("use_ddtree", 0),
        ],
    )
    if plugin is None:
        raise RuntimeError("Failed to create causal_conv1d plugin")
    inputs = _as_plugin_inputs(ctx, tensors, name)
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    _alias_resident_state(ctx, layer.get_output(1), inputs[3], name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.update_ssm_state_ragged.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_update_ssm_state_ragged(ctx: ConversionContext, target, args, kwargs, name):
    """Edge-LLM >=0.11 ``update_ssm_state``: token-major rows plus ragged metadata."""
    del target, kwargs
    args = list(args)
    tensors = args[:13]
    dt_softplus, ngroups, nheads, head_dim, dstate = args[13:18]

    creator = get_trt_plugin_creator("update_ssm_state", "1", "")
    if creator is None:
        raise RuntimeError(
            "update_ssm_state plugin not found in TensorRT plugin registry"
        )
    if not mamba_plugin_uses_ragged():
        raise NotImplementedError(
            "trt::update_ssm_state_ragged needs an Edge-LLM >=0.11 update_ssm_state build"
        )
    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _int_field("dim", head_dim),
            _int_field("dstate", dstate),
            _int_field("nheads", nheads),
            _int_field("ngroups", ngroups),
            _int_field("dt_softplus", dt_softplus),
        ],
    )
    if plugin is None:
        raise RuntimeError("Failed to create update_ssm_state plugin")
    inputs = _as_plugin_inputs(ctx, tensors, name)
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    _alias_resident_state(ctx, layer.get_output(1), inputs[7], name)
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.kimi_kda_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_kimi_kda_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    tensors = args[:9]
    lower_bound = float(args[9])
    use_lower_bound = int(bool(args[10]))

    creator = get_trt_plugin_creator("kimi_kda", "1", "")
    if creator is None:
        raise RuntimeError("kimi_kda not found in TensorRT plugin registry")

    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _float_field("lower_bound", lower_bound),
            _int_field("use_lower_bound", use_lower_bound),
        ],
    )
    if plugin is None:
        raise RuntimeError("Failed to create kimi_kda plugin")

    inputs = _as_plugin_inputs(ctx, tensors, name)
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0), layer.get_output(1)


def _convert_nvfp4_moe(ctx: ConversionContext, args, name: str, plugin_name: str):
    args = list(args)
    tensors = args[:11]
    num_experts = args[11]
    top_k = args[12]
    hidden_size = args[13]
    moe_inter_size = args[14]
    activation_type = args[15]
    n_group = args[16]
    topk_group = args[17]
    norm_topk_prob = args[18]
    routed_scaling_factor = args[19]
    routing_mode = args[20]
    backend = args[21]
    io_dtype = args[22]
    max_routed_rows = args[23]

    creator = get_trt_plugin_creator(plugin_name, "1", "")
    if creator is None:
        raise RuntimeError(f"{plugin_name} not found in TensorRT plugin registry")

    plugin = _create_trt_plugin(
        creator,
        name,
        [
            _int_field("num_experts", num_experts),
            _int_field("top_k", top_k),
            _int_field("hidden_size", hidden_size),
            _int_field("moe_inter_size", moe_inter_size),
            _int_field("activation_type", activation_type),
            _int_field("n_group", n_group),
            _int_field("topk_group", topk_group),
            _int_field("norm_topk_prob", norm_topk_prob),
            _float_field("routed_scaling_factor", routed_scaling_factor),
            _int_field("routing_mode", routing_mode),
            _int_field("backend", backend),
            _int_field("io_dtype", io_dtype),
            _int_field("max_routed_rows", max_routed_rows),
        ],
    )
    if plugin is None:
        raise RuntimeError(f"Failed to create {plugin_name}")

    inputs = _as_plugin_inputs(ctx, tensors, name)
    layer = _add_plugin_layer(ctx, inputs, plugin, name)
    return layer.get_output(0)


@dynamo_tensorrt_converter(
    torch.ops.trt.nvfp4_moe_plugin.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_nvfp4_moe_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    return _convert_nvfp4_moe(ctx, args, name, "Nvfp4MoePlugin")


@dynamo_tensorrt_converter(
    torch.ops.trt.nvfp4_moe_plugin_geforce.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def convert_nvfp4_moe_plugin_geforce(
    ctx: ConversionContext, target, args, kwargs, name
):
    del target, kwargs
    return _convert_nvfp4_moe(ctx, args, name, "NvFP4MoEPluginGeforce")
