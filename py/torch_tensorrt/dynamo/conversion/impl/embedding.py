from typing import Optional

import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def embedding(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TRTTensor:
    indices_tensor = input
    embedding_tensor = weight
    if isinstance(indices_tensor, torch.Tensor) and indices_tensor.dtype == torch.int64:
        raise RuntimeError(
            "The `embedding` op has indices_tensor dtype=int64. This is incorrect since it has to be int32 to run on TRT."
        )
    indices_tensor = get_trt_tensor(network, indices_tensor, f"{name}_indices_tensor")
    embedding_tensor = get_trt_tensor(
        network, embedding_tensor, f"{name}_embedding_tensor"
    )
    # unsupported parameters
    # ignore padding_idx since it is meaningful for training only

    if scale_grad_by_freq:
        raise RuntimeError(
            "Currently we don't support scale gradient by word frequency."
        )

    if sparse:
        raise RuntimeError("Currently we don't support sparse gradient.")

    # Implement embedding lookup with gather layer
    gather_layer = network.add_gather(embedding_tensor, indices_tensor, axis=0)
    set_layer_name(gather_layer, target, name + "_gather", source_ir)
    return gather_layer.get_output(0)
