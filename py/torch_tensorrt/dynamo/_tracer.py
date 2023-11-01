from __future__ import annotations

import logging
import unittest.mock
from typing import Any, List, Optional, Tuple, Union

import torch
from torch._export import dynamic_dim, export
from torch_tensorrt._Device import Device
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    DEVICE,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    default_device,
)
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
    mod: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    device: Optional[Union[Device, torch.device, str]] = DEVICE,
    debug: bool = DEBUG,
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    **kwargs: Any,
) -> torch.export.ExportedProgram:
    """Exports a ``torch.export.ExportedProgram`` from a ``torch.nn.Module`` or ``torch.fx.GraphModule`` specifically targeting being compiled with Torch-TensorRT

    Exports a ``torch.export.ExportedProgram`` from either a ``torch.nn.Module`` or torch.fx.GraphModule``. Runs specific operator decompositions geared towards
    compilation by Torch-TensorRT's dynamo frontend.

    Arguments:
        mod (torch.nn.Module | torch.fx.GraphModule): Source module to later be compiled by Torch-TensorRT's dynamo fronted
        inputs (Tuple[Any, ...]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]
    Keyword Arguments:
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        debug (bool): Enable debuggable engine
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the grap easier to covert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if debug:
        set_log_level(logger.parent, logging.DEBUG)
    device = to_torch_device(device if device else default_device())

    # Determine the dynamic dimension and setup constraints to input dimensions as dictated by TensorRT
    # Torch dynamo does not allow 0/1 value for dynamic dimensions
    # for inputs during tracing. Hence we create new inputs for export
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

    with unittest.mock.patch(
        "torch._export.DECOMP_TABLE",
        get_decompositions(enable_experimental_decompositions),
    ):
        exp_program = export(mod, tuple(trace_inputs), constraints=constraints)

    return exp_program
