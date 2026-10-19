# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# mypy: disallow-untyped-decorators=False

import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import tensorrt as trt
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    turing_rejects_forward_convolution,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    args_bounds_check,
    enforce_tensor_types,
)
from torch_tensorrt.dynamo.lowering._kv_cache_update import (
    tensorrt_kv_cache_update_op,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    tensorrt_fused_nccl_all_gather_op,
    tensorrt_fused_nccl_all_reduce_op,
    tensorrt_fused_nccl_all_to_all_op,
    tensorrt_fused_nccl_gather_op,
    tensorrt_fused_nccl_reduce_scatter_op,
    tensorrt_fused_nccl_scatter_op,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
    tensorrt_conv_asym_pad_op,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


def conv_asym_pad_capability_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    """Apply the aten.convolution capability guards to the fused pad+conv op.

    ``fuse_pad_into_convolution`` rewrites constant_pad_nd -> aten.convolution into
    this op during lowering, which erases the aten.convolution node that
    ``convolution_capability_validator`` would otherwise have rejected. Without this,
    the pass smuggles a 3D convolution past that guard onto Turing (SM 7.5), where it
    surfaces at runtime as a null execution context. The pass runs in the normal
    compile path, so this is not confined to tests.

    Spatial rank comes from the argument list rather than node meta, so the guard holds
    whether or not the caller traced with dynamo.
    """
    # conv_asym_pad(source, weight, bias, stride, pre_padding, post_padding, dilation,
    #               groups) -- no transposed argument; the pass skips those outright.
    _ARG_STRIDE = 3
    stride = args_bounds_check(node.args, _ARG_STRIDE)
    spatial_rank = len(stride) if stride is not None else None
    return not turing_rejects_forward_convolution(node, spatial_rank, settings)


@dynamo_tensorrt_converter(
    tensorrt_conv_asym_pad_op,
    capability_validator=conv_asym_pad_capability_validator,
    supports_dynamic_shapes=True,
)
def conv_asym_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    """Folded pad+conv with independent pre/post padding per spatial dim."""
    del kwargs
    source, weight, bias, stride, pre_padding, post_padding, dilation, groups = args
    return impl.conv.convNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        is_conv1d=False,
        input=source,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=None,
        dilation=dilation,
        groups=groups,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )


def _kv_cache_update_shapes_supported(
    node: Node, cache: Optional[Argument] = None
) -> bool:
    """Static shapes, a 4-D cache, and an update no longer than the cache. The op's
    fake kernel has already checked the rest when the graph was traced: matching
    batch / head / head_dim sizes and dtypes, and ``write_indices`` of shape
    ``[b]``. ``cache`` overrides the node read as the cache."""
    args = list(node.args[:3])
    if len(args) != 3:
        return False
    if cache is not None:
        args[0] = cache
    shapes = []
    for arg in args:
        val = arg.meta.get("val") if isinstance(arg, Node) else None
        if not isinstance(val, torch.Tensor):
            return False
        shapes.append(tuple(val.shape))
    if not all(isinstance(size, int) for shape in shapes for size in shape):
        return False
    cache_shape, update_shape = shapes[0], shapes[1]
    return (
        len(cache_shape) == 4
        and len(update_shape) == 4
        and update_shape[2] <= cache_shape[2]
    )


def kv_cache_update_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    """Writes the converter cannot lower (dynamic shapes) run in PyTorch."""
    return _kv_cache_update_shapes_supported(node)


def kv_cache_update_will_alias(
    node: Node, *, input_node: Optional[Node] = None
) -> bool:
    """Whether the converter will emit an ``IKVCacheUpdateLayer`` for this write:
    it converts, its cache is a network input, and ``lift_mutated_buffers`` did not
    file it copy-back.

    ``input_node`` overrides the node taken as the cache. The pre-conversion
    classifier in ``lowering/_buffer_lifting.py`` passes the node lowering will leave
    behind, which is not always ``node.args[0]``.
    """
    if node.meta.get("_trt_no_kv_alias"):
        return False
    cache = node.args[0] if input_node is None else input_node
    if not (isinstance(cache, Node) and cache.op == "placeholder"):
        return False
    return _kv_cache_update_shapes_supported(node, cache)


@dynamo_tensorrt_converter(
    tensorrt_kv_cache_update_op,
    capability_validator=kv_cache_update_validator,
    supports_dynamic_shapes=False,
)
@enforce_tensor_types({0: (trt.ITensor,), 1: (trt.ITensor,), 2: (trt.ITensor,)})
def tensorrt_kv_cache_update(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    """KV-cache write with a position per batch row: an ``IKVCacheUpdateLayer``
    when the cache is a network input, else a scatter."""
    del kwargs
    cache, update, write_indices = args[:3]
    return impl.kv_cache_update.kv_cache_update(
        ctx, target, SourceIR.ATEN, name, cache, update, write_indices
    )


if ENABLED_FEATURES.native_trt_collectives:
    # Use native TensorRT DistCollective API (no TensorRT-LLM dependency)
    _LOGGER.info("Using native TensorRT DistCollective API for distributed operations")

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_all_gather_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_all_gather(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """All-gather using native TensorRT DistCollective API"""
        return impl.nccl_ops.nccl_all_gather_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_reduce_scatter_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_reduce_scatter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """Reduce-scatter using native TensorRT DistCollective API"""
        # Fused node args: (inp, reduce_op, group_size, group_name)
        reduce_op = args[1] if len(args) > 1 else "sum"
        return impl.nccl_ops.nccl_reduce_scatter_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
            reduce_op=reduce_op,
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_all_reduce_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_all_reduce(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """All-reduce using native TensorRT DistCollective API"""
        reduce_op = args[1] if len(args) > 1 else "sum"
        return impl.nccl_ops.nccl_all_reduce_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
            reduce_op=reduce_op,
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_all_to_all_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_all_to_all(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """All-to-all using native TensorRT DistCollective API."""
        return impl.nccl_ops.nccl_all_to_all_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_scatter_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_scatter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """Scatter using native TensorRT DistCollective API."""
        root = args[1] if len(args) > 1 else 0
        return impl.nccl_ops.nccl_scatter_native(
            ctx, target, SourceIR.ATEN, name, [args[0]], root=root
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_gather_op,
        requires_native_multidevice=True,
        supports_dynamic_shapes=True,
    )
    def fused_nccl_gather(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """Gather using native TensorRT DistCollective API."""
        root = args[1] if len(args) > 1 else 0
        return impl.nccl_ops.nccl_gather_native(
            ctx, target, SourceIR.ATEN, name, [args[0]], root=root
        )


# Conditionally register NCCL converters only if TensorRT-LLM plugin is available.
# We use an `if` statement instead of @needs_trtllm_for_nccl decorator because
# @dynamo_tensorrt_converter ALWAYS registers at import time regardless of decorator
# order. Conditional registration prevents registration when TRTLLM is unavailable,
# allowing fallback to PyTorch execution for NCCL ops.

# Order 1: @needs_trtllm_for_nccl followed by registering the converter leads to plugin registry not finding nccl ops plugins since we register the bare converter, without the decorator
# Order 2: registering the converter first followed by @needs_trtllm_for_nccl leads to  "NotImplementedError: TensorRT-LLM plugin for NCCL is not available :TensorRT-LLM plugin for NCCL is not available" and no fall back to pytorch
elif ENABLED_FEATURES.trtllm_for_nccl:
    _LOGGER.debug(
        "TensorRT-LLM plugin for NCCL is available. Registering NCCL converters."
    )

    @dynamo_tensorrt_converter(tensorrt_fused_nccl_all_gather_op)
    def fused_nccl_all_gather(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        return impl.nccl_ops.nccl_all_gather(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(tensorrt_fused_nccl_reduce_scatter_op)
    def fused_nccl_reduce_scatter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        return impl.nccl_ops.nccl_reduce_scatter(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(tensorrt_fused_nccl_all_reduce_op)
    def fused_nccl_all_reduce(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        return impl.nccl_ops.nccl_all_reduce(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

else:
    _LOGGER.info(
        "TensorRT-LLM plugin for NCCL is not available. "
        "NCCL operations will fall back to PyTorch execution."
    )
