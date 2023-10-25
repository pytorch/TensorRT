from __future__ import annotations

import logging
from typing import Any, Tuple

import torch
from torch.export import Dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import (
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    default_device,
)
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.utils import get_torch_inputs, set_log_level, to_torch_device

logger = logging.getLogger(__name__)


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if "debug" in kwargs and kwargs["debug"]:
        set_log_level(logger.parent, logging.DEBUG)

    device = to_torch_device(kwargs.get("device", default_device()))
    torch_inputs = get_torch_inputs(inputs, device)
    dynamic_shapes = {}
    for input in inputs:
        if input.shape_mode == Input._ShapeMode.DYNAMIC:
            min_shape = input.shape["min_shape"]
            opt_shape = input.shape["opt_shape"]
            max_shape = input.shape["max_shape"]
            assert len(min_shape) == len(opt_shape) == len(max_shape)
            dynamic_dims = {}
            for dim in range(len(min_shape)):
                if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                    continue
                else:
                    dynamic_dims[dim] = Dim(
                        input.name + "_" + str(dim),
                        min=min_shape[dim],
                        max=max_shape[dim],
                    )

            dynamic_shapes[input.name] = dynamic_dims

    experimental_decompositions = kwargs.get(
        "enable_experimental_decompositions", ENABLE_EXPERIMENTAL_DECOMPOSITIONS
    )

    exp_program = export(
        model, tuple(torch_inputs), dynamic_shapes=dynamic_shapes
    ).run_decompositions(get_decompositions(experimental_decompositions))

    return exp_program
