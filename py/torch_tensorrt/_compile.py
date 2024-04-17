from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Set

import torch
import torch.fx
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.fx import InputTensorSpec
from torch_tensorrt.fx.lower import compile as fx_compile
from torch_tensorrt.fx.utils import LowerPrecision
from typing_extensions import TypeGuard

if ENABLED_FEATURES.torchscript_frontend:
    import torch_tensorrt.ts
    from torch_tensorrt.ts._compiler import compile as torchscript_compile
    from torch_tensorrt.ts._compiler import (
        convert_method_to_trt_engine as ts_convert_method_to_trt_engine,
    )

if ENABLED_FEATURES.dynamo_frontend:
    from torch._export import ExportedProgram
    from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
    from torch_tensorrt.dynamo._compiler import (
        convert_module_to_trt_engine as dynamo_convert_module_to_trt_engine,
    )
    from torch_tensorrt.dynamo._tracer import trace as dynamo_trace

logger = logging.getLogger(__name__)

__all__ = [
    "compile",
    "convert_method_to_trt_engine",
]


def _non_fx_input_interface(
    inputs: Sequence[Input | torch.Tensor | InputTensorSpec],
) -> TypeGuard[List[Input | torch.Tensor]]:
    return all(isinstance(i, (torch.Tensor, Input)) for i in inputs)


def _fx_input_interface(
    inputs: Sequence[Input | torch.Tensor | InputTensorSpec],
) -> TypeGuard[List[InputTensorSpec | torch.Tensor]]:
    return all(isinstance(i, (torch.Tensor, InputTensorSpec)) for i in inputs)


class _IRType(Enum):
    """Enum to determine the type of IR selected for model compilation"""

    ts = 0
    fx = 1
    dynamo = 2
    torch_compile = 3
    exported_program = 4


class _ModuleType(Enum):
    """Enum to determine the type of model provided as input"""

    nn = 0
    ts = 1
    fx = 2
    ep = 3


def _parse_module_type(module: Any) -> _ModuleType:
    if any(
        isinstance(module, t)
        for t in [torch.jit.ScriptModule, torch.jit.ScriptFunction]
    ):
        return _ModuleType.ts
    elif isinstance(module, torch.fx.GraphModule):
        return _ModuleType.fx
    elif isinstance(module, ExportedProgram):
        return _ModuleType.ep
    elif isinstance(module, torch.nn.Module):
        return _ModuleType.nn
    else:
        raise RuntimeError("Module is an unknown format")


def _get_target_fe(module_type: _ModuleType, ir: str) -> _IRType:
    module_is_tsable = any(module_type == t for t in [_ModuleType.nn, _ModuleType.ts])
    module_is_fxable = any(module_type == t for t in [_ModuleType.nn, _ModuleType.fx])
    module_is_exportable = module_type == _ModuleType.ep

    ir_targets_torchscript = any(ir == opt for opt in ["torchscript", "ts"])
    ir_targets_fx = ir == "fx"
    ir_targets_dynamo = ir == "dynamo"
    ir_targets_torch_compile = ir == "torch_compile"

    if module_is_tsable and ir_targets_torchscript:
        if ENABLED_FEATURES.torchscript_frontend:
            return _IRType.ts
        else:
            raise ValueError(
                "Requested using the TS frontend but the TS frontend is not available in this build of Torch-TensorRT"
            )
    elif module_is_fxable and ir_targets_fx:
        if ENABLED_FEATURES.fx_frontend:
            return _IRType.fx
        else:
            raise ValueError(
                "Requested using the FX frontend but the FX frontend is not available in this build of Torch-TensorRT"
            )
    elif (module_is_fxable or module_is_exportable) and ir_targets_dynamo:
        if ENABLED_FEATURES.dynamo_frontend:
            return _IRType.dynamo
        else:
            raise ValueError(
                "Requested using the Dynamo frontend but the Dynamo frontend is not available in this build of Torch-TensorRT"
            )
    elif module_is_fxable and ir_targets_torch_compile:
        if ENABLED_FEATURES.dynamo_frontend:
            return _IRType.torch_compile
        else:
            raise ValueError(
                "Requested using the Torch-TensorRT torch.compile backend but the Torch-TensorRT torch.compile backend is not available in this build of Torch-TensorRT"
            )
    else:
        if ir == "default":
            # Options are listed in order of preference
            if ENABLED_FEATURES.dynamo_frontend and module_is_fxable:
                logger.info("ir was set to default, using dynamo frontend")
                return _IRType.dynamo
            elif ENABLED_FEATURES.torchscript_frontend and module_is_tsable:
                if ENABLED_FEATURES.dynamo_frontend:
                    logger.warning(
                        "Input is a torchscript module but the ir was not specified (default=dynamo), please set ir=torchscript to suppress the warning."
                    )
                return _IRType.ts
            elif ENABLED_FEATURES.dynamo_frontend and module_is_exportable:
                logger.info("ir was set to default, using dynamo frontend")
                return _IRType.dynamo
            else:
                raise ValueError(
                    f"Module was provided in an unsupported format\nInstalled frontends:\n\tDynamo - {ENABLED_FEATURES.dynamo_frontend}\n\tTorchScript - {ENABLED_FEATURES.torchscript_frontend}\n\tFX - {ENABLED_FEATURES.fx_frontend})"
                )
        else:
            raise ValueError("Unknown ir was requested")


def compile(
    module: Any,
    ir: str = "default",
    inputs: Optional[Sequence[Input | torch.Tensor | InputTensorSpec]] = None,
    enabled_precisions: Optional[Set[torch.dtype | dtype]] = None,
    **kwargs: Any,
) -> (
    torch.nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule | Callable[..., Any]
):
    """Compile a PyTorch module for NVIDIA GPUs using TensorRT

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
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

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """
    input_list = inputs if inputs is not None else []
    enabled_precisions_set: Set[dtype | torch.dtype] = (
        enabled_precisions
        if enabled_precisions is not None
        else _defaults.ENABLED_PRECISIONS
    )

    module_type = _parse_module_type(module)
    target_ir = _get_target_fe(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logger.info(
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        assert _non_fx_input_interface(input_list)
        compiled_ts_module: torch.jit.ScriptModule = torchscript_compile(
            ts_mod,
            inputs=input_list,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
        return compiled_ts_module
    elif target_ir == _IRType.fx:
        if (
            torch.float16 in enabled_precisions_set
            or torch_tensorrt.dtype.half in enabled_precisions_set
        ):
            lower_precision = LowerPrecision.FP16
        elif (
            torch.float32 in enabled_precisions_set
            or torch_tensorrt.dtype.float in enabled_precisions_set
        ):
            lower_precision = LowerPrecision.FP32
        else:
            raise ValueError(f"Precision {enabled_precisions_set} not supported on FX")

        assert _fx_input_interface(input_list)
        compiled_fx_module: torch.nn.Module = fx_compile(
            module,
            input_list,
            lower_precision=lower_precision,
            explicit_batch_dimension=True,
            dynamic_batch=False,
            **kwargs,
        )
        return compiled_fx_module
    elif target_ir == _IRType.dynamo:
        # Prepare torch and torchtrt inputs
        import collections.abc

        from torch_tensorrt.dynamo.utils import prepare_inputs

        if not isinstance(input_list, collections.abc.Sequence):
            input_list = [input_list]

        # Export the module
        torchtrt_inputs = prepare_inputs(input_list)
        exp_program = dynamo_trace(module, torchtrt_inputs, **kwargs)
        trt_graph_module = dynamo_compile(
            exp_program,
            inputs=torchtrt_inputs,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
        return trt_graph_module
    elif target_ir == _IRType.torch_compile:
        return torch_compile(
            module, enabled_precisions=enabled_precisions_set, **kwargs
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def torch_compile(module: torch.nn.Module, **kwargs: Any) -> Any:
    """
    Returns a boxed model which is the output of torch.compile.
    This does not compile the model to TRT. Execute this model on
    sample inputs to compile the model to TRT.
    """
    from torch_tensorrt.dynamo.backend import torch_tensorrt_backend

    # TODO: Remove dynamic=False when SymInt Dynamic shape support is ready
    boxed_fn = torch.compile(
        module, backend=torch_tensorrt_backend, dynamic=False, options={**kwargs}
    )

    return boxed_fn


def convert_method_to_trt_engine(
    module: Any,
    method_name: str = "forward",
    inputs: Optional[Sequence[Input | torch.Tensor]] = None,
    ir: str = "default",
    enabled_precisions: Optional[Set[torch.dtype | dtype]] = None,
    **kwargs: Any,
) -> bytes:
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
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

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else {torch.float}
    )

    module_type = _parse_module_type(module)
    target_ir = _get_target_fe(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logger.info(
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        serialized_engine: bytes = ts_convert_method_to_trt_engine(
            ts_mod,
            inputs=inputs,
            method_name=method_name,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
        return serialized_engine
    elif target_ir == _IRType.fx:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=fx"
        )
    elif target_ir == _IRType.dynamo:
        return dynamo_convert_module_to_trt_engine(  # type: ignore[no-any-return]
            module,
            inputs=inputs,
            method_name=method_name,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
    elif target_ir == _IRType.torch_compile:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=torch_compile"
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")
