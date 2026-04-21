# mypy: disallow-untyped-decorators=False

import logging
from typing import Dict, Sequence, Tuple, Union

from torch.fx.node import Argument, Target
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    tensorrt_fused_nccl_all_gather_op,
    tensorrt_fused_nccl_all_reduce_op,
    tensorrt_fused_nccl_reduce_scatter_op,
)

import tensorrt as trt

_LOGGER: logging.Logger = logging.getLogger(__name__)

if ENABLED_FEATURES.native_trt_collectives:
    # Use native TensorRT DistCollective API (no TensorRT-LLM dependency)
    _LOGGER.info("Using native TensorRT DistCollective API for distributed operations")

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_all_gather_op, requires_native_multidevice=True
    )
    def fused_nccl_gather(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """All-gather using native TensorRT DistCollective API"""
        return impl.nccl_ops.nccl_gather_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_reduce_scatter_op, requires_native_multidevice=True
    )
    def fused_nccl_reduce_scatter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        """Reduce-scatter using native TensorRT DistCollective API"""
        return impl.nccl_ops.nccl_reduce_scatter_native(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            [args[0]],
        )

    @dynamo_tensorrt_converter(
        tensorrt_fused_nccl_all_reduce_op, requires_native_multidevice=True
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
    def fused_nccl_gather(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
        return impl.nccl_ops.nccl_gather(
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
