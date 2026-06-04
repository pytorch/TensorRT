from __future__ import annotations

import logging
from inspect import signature
from typing import Any, Optional, Tuple, Union

import torch
from torch.export import Dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo.utils import get_torch_inputs, to_torch_device

logger = logging.getLogger(__name__)


def trace(
    mod: torch.nn.Module | torch.fx.GraphModule,
    inputs: Optional[Tuple[Any, ...]] = None,
    *,
    arg_inputs: Optional[Tuple[Any, ...]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
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
        arg_inputs (Tuple[Any, ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[Any, ...]): Optional, kwarg inputs to the module forward function.
        device (Union(torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch.device("cuda:0")

        debug (bool): Enable debuggable engine
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if arg_inputs is None and inputs is None:
        raise AssertionError("'arg_inputs' and 'inputs' should not both be None.")

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )
    arg_inputs = inputs if inputs is not None else arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    device = to_torch_device(kwargs.get("device", default_device()))
    torch_arg_inputs = get_torch_inputs(arg_inputs, device)
    torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
    # Build dynamic shapes from the Input objects. Inputs carrying name_dims
    # share a Dim across inputs via the registry; the rest get an independent
    # per-input Dim.
    dim_registry = build_dim_registry(arg_inputs, kwarg_inputs)
    dynamic_shapes = get_dynamic_shapes_args(mod, arg_inputs, dim_registry)
    dynamic_shapes.update(get_dynamic_shapes_kwargs(kwarg_inputs, dim_registry))
    exp_program = export(
        mod,
        tuple(torch_arg_inputs),
        kwargs=torch_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=kwargs.get("strict", False),
    )

    return exp_program


def _collect_inputs(obj: Any) -> list[Input]:
    """Flatten an arg/kwarg input structure into a list of Input objects."""
    if isinstance(obj, Input):
        return [obj]
    elif isinstance(obj, dict):
        collected: list[Input] = []
        for v in obj.values():
            collected.extend(_collect_inputs(v))
        return collected
    elif isinstance(obj, (list, tuple)):
        collected = []
        for v in obj:
            collected.extend(_collect_inputs(v))
        return collected
    return []


def build_dim_registry(arg_inputs: Any, kwarg_inputs: Any) -> dict[str, Any]:
    """Build a ``{name: torch.export.Dim}`` registry from Input.name_dims.

    The same name appearing on multiple inputs yields a single shared ``Dim``
    instance, so ``torch.export`` treats those axes as one symbol. Conflicting
    (min, max) ranges for the same name are rejected.
    """
    registry: dict[str, Any] = {}
    bounds: dict[str, tuple[int, int]] = {}
    for inp in _collect_inputs(arg_inputs) + _collect_inputs(kwarg_inputs):
        name_dims = getattr(inp, "name_dims", None)
        if not name_dims or inp.shape_mode != Input._ShapeMode.DYNAMIC:
            continue
        assert isinstance(inp.shape, dict)
        min_shape = inp.shape["min_shape"]
        max_shape = inp.shape["max_shape"]
        for axis, dim_name in name_dims.items():
            lo, hi = int(min_shape[axis]), int(max_shape[axis])
            if dim_name in bounds:
                if bounds[dim_name] != (lo, hi):
                    raise ValueError(
                        f"Dimension name '{dim_name}' is used with conflicting ranges "
                        f"{bounds[dim_name]} and {(lo, hi)}. A shared named dimension "
                        f"must have identical (min, max) on every input that uses it."
                    )
            else:
                bounds[dim_name] = (lo, hi)
                registry[dim_name] = Dim(dim_name, min=lo, max=hi)
    return registry


def get_dynamic_shapes_kwargs(
    inputs: Any, dim_registry: Optional[dict[str, Any]] = None
) -> Union[dict[str, Any], list[Any]]:
    if isinstance(inputs, dict):
        dynamic_shapes_kwarg = {}
        for k, v in inputs.items():
            dynamic_shapes_kwarg[k] = get_dynamic_shapes_kwargs(v, dim_registry)
        return dynamic_shapes_kwarg

    elif isinstance(inputs, Input):
        return get_dynamic_shapes(inputs, dim_registry)

    elif isinstance(inputs, (list, tuple)):
        dynamic_shapes = []
        for input in inputs:
            dynamic_shapes.append(get_dynamic_shapes(input, dim_registry))
        return dynamic_shapes

    raise TypeError(f"Unknown type {type(inputs)}.")


def get_dynamic_shapes_args(
    mod: torch.nn.Module, inputs: Any, dim_registry: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    # dynamic_shape is a dict and cannot work without keys. Here we use position argument name
    # in forward function as the name
    args = list(signature(mod.forward).parameters.keys())
    dynamic_shapes = {}
    for input, input_name in zip(inputs, args[: len(inputs)]):
        dynamic_shapes[input_name] = get_dynamic_shapes(input, dim_registry)
    return dynamic_shapes


def get_dynamic_shapes(
    input: Input, dim_registry: Optional[dict[str, Any]] = None
) -> dict[Any, Any]:
    if not isinstance(input, Input):
        # If the input is torch.Tensor, no dynamic is needed. Return empty dict
        return {}
    else:
        dynamic_dims = {}
        if input.shape_mode == Input._ShapeMode.DYNAMIC:
            assert isinstance(input.shape, dict)
            min_shape = input.shape["min_shape"]
            opt_shape = input.shape["opt_shape"]
            max_shape = input.shape["max_shape"]
            name_dims = getattr(input, "name_dims", None) or {}
            assert len(min_shape) == len(opt_shape) == len(max_shape)
            for dim in range(len(min_shape)):
                if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                    continue
                elif dim_registry is not None and dim in name_dims:
                    # Named axis: reuse the shared Dim so axes with the same
                    # name across inputs become a single exported symbol.
                    dynamic_dims[dim] = dim_registry[name_dims[dim]]
                else:
                    dynamic_dims[dim] = Dim(
                        input.name + "_" + str(dim),
                        min=min_shape[dim],
                        max=max_shape[dim],
                    )
        return dynamic_dims
