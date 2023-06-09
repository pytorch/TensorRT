import operator
import warnings
from typing import Optional, cast, Any

import numpy as np

import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
)

from torch_tensorrt.fx.converters.impl.elementwise import get_trt_tensor

def embedding(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
    max_norm: None,
    norm_type: None,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TRTTensor:

    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `embedding` function should be called with explicit batch dimension."
        )

    indices_tensor = input
    embedding_tensor = weight
    if isinstance(indices_tensor, torch.Tensor) and indices_tensor.dtype == torch.int64:
        indices_tensor = indices_tensor.to(torch.int32)
        warnings.warn(
            "Embedding op has indices_tensor dtype=int64. Reduce it to int32 to run on TRT. Accuracy may not be correct!"
        )
    if (
        isinstance(embedding_tensor, torch.Tensor)
        and embedding_tensor.dtype == torch.int64
    ):
        embedding_tensor = embedding_tensor.to(torch.int32)
        warnings.warn(
            "Embedding op has embedding_tensor dtype=int64. Reduce it to int32 to run on TRT. Accuracy may not be correct!"
        )
    indices_tensor = get_trt_tensor(network, indices_tensor, f"{name}_indices_tensor")
    embedding_tensor = get_trt_tensor(
        network, embedding_tensor, f"{name}_embedding_tensor"
    )

    # unsupported parameters
    # ignore padding_idx since it is meaningful for training only

    if max_norm is not None:
        raise RuntimeError(
            f"Currently we don't support specifying max_norm, got {max_norm}."
        )

    if norm_type != 2.0:
        raise RuntimeError(
            f"Currently we don't support specifying max_norm, got {norm_type} for norm_type."
        )

    if scale_grad_by_freq:
        raise RuntimeError(
            "Currently we don't support scale gradient by word frequency."
        )

    if sparse:
        raise RuntimeError("Currently we don't support sparse gradient.")

    # Implement embedding lookup with gather layer
    gather_layer = network.add_gather(embedding_tensor, indices_tensor, axis=0)
    set_layer_name(gather_layer, target, name + "_gather")
    return gather_layer.get_output(0)