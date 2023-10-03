from __future__ import annotations

import logging
from typing import Any, List, Tuple

import torch
from torch._export import dynamic_dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.utils import get_torch_inputs, set_log_level, to_torch_device

logger = logging.getLogger(__name__)


def get_random_tensor(
    shape: List[Any], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.int32 or dtype == torch.int64:
        return torch.randint(2, 10, shape, dtype=dtype, device=device)
    elif dtype in (torch.float64, torch.float32, torch.float16):
        return torch.randn(shape, dtype=dtype, device=device)
    else:
        logger.critical(
            "Invalid dtype detected in creating input tensors for tracing the graph."
        )
        raise


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if "debug" in kwargs and kwargs["debug"]:
        set_log_level(logger.parent, logging.DEBUG)

    # Determine the dynamic dimension and setup constraints to input dimensions as dictated by TensorRT
    # Torch dynamo does not allow 0/1 value for dynamic dimensions
    # for inputs during tracing. Hence we create new inputs for export
    device = to_torch_device(kwargs.get("device", default_device()))
    torch_inputs = get_torch_inputs(inputs, device)
    trace_inputs = []
    constraints = []
    for idx, input in enumerate(inputs):
        if input.shape_mode == Input._ShapeMode.DYNAMIC:
            min_shape = input.shape["min_shape"]
            opt_shape = input.shape["opt_shape"]
            max_shape = input.shape["max_shape"]
            assert len(min_shape) == len(opt_shape) == len(max_shape)

            constraint_dims = []
            new_shape = []
            for dim in range(len(min_shape)):
                if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                    new_shape.append(torch_inputs[idx].shape[dim])
                else:
                    constraint_dims.append(dim)
                    if torch_inputs[idx].shape[dim] == 1:
                        new_shape.append(torch_inputs[idx].shape[dim] + 1)
                    else:
                        new_shape.append(torch_inputs[idx].shape[dim])

            trace_input = get_random_tensor(new_shape, torch_inputs[idx].dtype, device)

            for dim in constraint_dims:
                if min_shape[dim] > 1:
                    constraints.append(min_shape[dim] <= dynamic_dim(trace_input, dim))
                if max_shape[dim] > 1:
                    constraints.append(dynamic_dim(trace_input, dim) <= max_shape[dim])
            trace_inputs.append(trace_input)
        else:
            trace_inputs.append(torch_inputs[idx])

    experimental_decompositions = kwargs.get(
        "enable_experimental_decompositions", False
    )

    exp_program = export(
        model, tuple(trace_inputs), constraints=constraints
    ).run_decompositions(get_decompositions(experimental_decompositions))

    return exp_program
