from __future__ import annotations

import logging
from typing import Any, Tuple

import torch
from torch.export import Dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import DEBUG, default_device
from torch_tensorrt.dynamo.utils import get_torch_inputs, set_log_level, to_torch_device

logger = logging.getLogger(__name__)


def trace(
    mod: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
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
        device (Union(torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch.device("cuda:0")

        debug (bool): Enable debuggable engine
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the grap easier to covert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    debug = kwargs.get("debug", DEBUG)
    if debug:
        set_log_level(logger.parent, logging.DEBUG)

    device = to_torch_device(kwargs.get("device", default_device()))
    torch_inputs = get_torch_inputs(inputs, device)
    dynamic_shapes = {}
    for input in inputs:
        if isinstance(input, Input) and input.shape_mode == Input._ShapeMode.DYNAMIC:
            if not input.name:
                raise AssertionError(
                    f"Expected a name for a dynamic input with shape {input.shape} but found none"
                )
            min_shape = input.shape["min_shape"]  # type: ignore
            opt_shape = input.shape["opt_shape"]  # type: ignore
            max_shape = input.shape["max_shape"]  # type: ignore
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

    exp_program = export(mod, tuple(torch_inputs), dynamic_shapes=dynamic_shapes)

    return exp_program
