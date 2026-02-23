from __future__ import annotations

import collections.abc
import logging
import platform
import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES, needs_cross_compile
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)
from typing_extensions import TypeGuard

if ENABLED_FEATURES.fx_frontend:
    import torch.fx
    from torch_tensorrt.fx import InputTensorSpec
    from torch_tensorrt.fx.lower import compile as fx_compile
    from torch_tensorrt.fx.utils import LowerPrecision

    InputType = Union[Input, torch.Tensor]
else:
    InputType = Union[Input, torch.Tensor]  # type: ignore

if ENABLED_FEATURES.torchscript_frontend:
    import torch_tensorrt.ts
    from torch_tensorrt.ts._compiler import compile as torchscript_compile
    from torch_tensorrt.ts._compiler import (
        convert_method_to_trt_engine as ts_convert_method_to_trt_engine,
    )

if ENABLED_FEATURES.dynamo_frontend:
    from torch.export import ExportedProgram
    from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
    from torch_tensorrt.dynamo._compiler import (
        convert_exported_program_to_serialized_trt_engine as dynamo_convert_exported_program_to_serialized_trt_engine,
    )
    from torch_tensorrt.dynamo._compiler import (
        cross_compile_for_windows as dynamo_cross_compile_for_windows,
    )
    from torch_tensorrt.dynamo._compiler import (
        load_cross_compiled_exported_program as dynamo_load_cross_compiled_exported_program,
    )
    from torch_tensorrt.dynamo._compiler import (
        save_cross_compiled_exported_program as dynamo_save_cross_compiled_exported_program,
    )
    from torch_tensorrt.dynamo._defaults import default_device
    from torch_tensorrt.dynamo._tracer import (
        get_dynamic_shapes_args,
        get_dynamic_shapes_kwargs,
    )
    from torch_tensorrt.dynamo._tracer import trace as dynamo_trace
    from torch_tensorrt.dynamo.utils import get_torch_inputs

logger = logging.getLogger(__name__)

__all__ = [
    "compile",
    "cross_compile_for_windows",
    "load_cross_compiled_exported_program",
    "convert_method_to_trt_engine",
    "save",
    "load",
]


def _non_fx_input_interface(
    inputs: Sequence[Input | torch.Tensor],
) -> TypeGuard[List[Input | torch.Tensor]]:
    return all(isinstance(i, (torch.Tensor, Input)) for i in inputs)


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
        warnings.warn(
            "FX frontend is deprecated. Please use the Dynamo frontend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    inputs: Optional[Sequence[InputType]] = None,
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[Dict[str, Any]] = None,
    enabled_precisions: Optional[Set[Union[torch.dtype, dtype]]] = None,
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

                inputs=[
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
        arg_inputs (Tuple[Any, ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[Any, ...]): Optional, kwarg inputs to the module forward function.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """

    input_list = inputs if inputs is not None else []
    enabled_precisions_set: Set[Union[torch.dtype, dtype]] = (
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
        warnings.warn(
            "FX frontend is deprecated. Please use the Dynamo frontend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not ENABLED_FEATURES.fx_frontend:
            raise RuntimeError(
                "FX frontend is not enabled, cannot compile with target_ir=fx"
            )

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

        def _fx_input_interface(
            inputs: Sequence[Input | torch.Tensor | InputTensorSpec],
        ) -> TypeGuard[List[InputTensorSpec | torch.Tensor]]:
            return all(isinstance(i, (torch.Tensor, InputTensorSpec)) for i in inputs)

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
        if arg_inputs is None and inputs is None:
            raise AssertionError("'arg_inputs' and 'inputs' should not both be None.")

        elif arg_inputs is not None and inputs is not None:
            raise AssertionError(
                "'arg_inputs' and 'inputs' should not be used at the same time."
            )
        if inputs is not None:
            arg_inputs = inputs

        if kwarg_inputs is None:
            kwarg_inputs = {}

        from torch_tensorrt.dynamo.utils import prepare_inputs

        if not isinstance(arg_inputs, collections.abc.Sequence):
            arg_inputs = [arg_inputs]  # type: ignore

        torchtrt_arg_inputs = prepare_inputs(arg_inputs)
        torchtrt_kwarg_inputs = prepare_inputs(kwarg_inputs)

        if module_type == _ModuleType.ep:
            exp_program = module
        else:
            exp_program = dynamo_trace(
                module,
                torchtrt_arg_inputs,
                kwarg_inputs=torchtrt_kwarg_inputs,
                **kwargs,
            )
        trt_graph_module = dynamo_compile(
            exp_program,
            arg_inputs=torchtrt_arg_inputs,
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


@needs_cross_compile  # type: ignore[misc]
def cross_compile_for_windows(
    module: torch.nn.Module,
    file_path: str,
    inputs: Optional[Sequence[Input | torch.Tensor]] = None,
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    enabled_precisions: Optional[Set[Union[torch.dtype, dtype]]] = None,
    **kwargs: Any,
) -> None:
    """Compile a PyTorch module using TensorRT in Linux for Inference in Windows

    Takes an existing PyTorch module and a set of settings to configure the compiler
    and it will convert methods to AOT graphs which call equivalent TensorRT serialized
    engine info into the disk in the specified file_path user provided.
    It will then allow user to load the deserialized model from the disk in Windows.
    Note: the model cross compiled for windows in Linux environmen can only be loaded
    in Windows.

    Argument:
        module (torch.nn.Module): Source module
        file_path (str): the file path to store the serialized module into the disk

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                inputs=[
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
        arg_inputs (Tuple[Any, ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[Any, ...]): Optional, kwarg inputs to the module forward function.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    """

    if platform.system() != "Linux" or platform.architecture()[0] != "64bit":
        raise RuntimeError(
            f"Cross compile for windows is only supported on x86-64 Linux architecture, current platform: {platform.system()=}, {platform.architecture()[0]=}"
        )

    if not file_path:
        raise ValueError("File path cannot be empty. Please provide a valid file path")

    enabled_precisions_set: Set[dtype | torch.dtype] = (
        enabled_precisions
        if enabled_precisions is not None
        else _defaults.ENABLED_PRECISIONS
    )

    # Prepare torch and torchtrt inputs
    if arg_inputs is None and inputs is None:
        raise AssertionError("'arg_inputs' and 'inputs' should not both be None.")

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )

    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    from torch_tensorrt.dynamo.utils import prepare_inputs

    if not isinstance(arg_inputs, collections.abc.Sequence):
        arg_inputs = [arg_inputs]  # type: ignore

    # Export the module
    torchtrt_arg_inputs = prepare_inputs(arg_inputs)
    torchtrt_kwarg_inputs = prepare_inputs(kwarg_inputs)

    exp_program = dynamo_trace(
        module, torchtrt_arg_inputs, kwarg_inputs=torchtrt_kwarg_inputs, **kwargs
    )
    logger.debug("successfully exported the module")

    # Compile and save the module
    trt_gm = dynamo_cross_compile_for_windows(
        exp_program,
        arg_inputs=torchtrt_arg_inputs,
        enabled_precisions=enabled_precisions_set,
        **kwargs,
    )

    dynamo_save_cross_compiled_exported_program(trt_gm, file_path)
    logger.debug("successfully compiled and saved the module for windows")


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
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    ir: str = "default",
    enabled_precisions: Optional[Set[Union[torch.dtype, dtype]]] = None,
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

        arg_inputs (Tuple[Any, ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[Any, ...]): Optional, kwarg inputs to the module forward function.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else {torch.float}
    )

    if arg_inputs is None and inputs is None:
        raise AssertionError("'arg_inputs' and 'inputs' should not both be None.")

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )
    arg_inputs = arg_inputs or inputs

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
            inputs=arg_inputs,
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
        # Prepare torch and torchtrt inputs
        if kwarg_inputs is None:
            kwarg_inputs = {}

        from torch_tensorrt.dynamo.utils import prepare_inputs

        if not isinstance(arg_inputs, collections.abc.Sequence):
            arg_inputs = [arg_inputs]  # type: ignore

        # Export the module
        torchtrt_arg_inputs = prepare_inputs(arg_inputs)
        torchtrt_kwarg_inputs = prepare_inputs(kwarg_inputs)

        exp_program = torch_tensorrt.dynamo.trace(
            module, torchtrt_arg_inputs, kwarg_inputs=torchtrt_kwarg_inputs, **kwargs
        )

        return dynamo_convert_exported_program_to_serialized_trt_engine(
            exp_program,
            arg_inputs=tuple(arg_inputs),
            kwarg_inputs=torchtrt_kwarg_inputs,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
    elif target_ir == _IRType.torch_compile:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=torch_compile"
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def load_cross_compiled_exported_program(file_path: str = "") -> Any:
    """
    Load an ExportedProgram file in Windows which was previously cross compiled in Linux

    Arguments:
        file_path (str): Path to file on the disk

    Raises:
        ValueError: If the api is not called in windows or there is no file or the file is not a valid ExportedProgram file
    """
    return dynamo_load_cross_compiled_exported_program(file_path)


def load(file_path: str = "") -> Any:
    """
    Load either a Torchscript model or ExportedProgram.

    Loads a TorchScript or ExportedProgram file from disk. File type will be detect the type using try, except.

    Arguments:
        file_path (str): Path to file on the disk

    Raises:
        ValueError: If there is no file or the file is not either a TorchScript file or ExportedProgram file
    """

    try:
        logger.debug(f"Loading the provided file {file_path} using torch.export.load()")
        exp_program = torch.export.load(file_path)
        return exp_program
    except Exception:
        logger.info(
            f"Loading the provided file {file_path} via torch.export.load() failed with the following error",
            exc_info=True,
        )
        pass

    try:
        logger.debug(f"Loading the provided file {file_path} using torch.jit.load()")
        ts_module = torch.jit.load(file_path)
        return ts_module
    except Exception:
        logger.info(
            f"Loading the provided file {file_path} via torch.jit.load() (after failing to load with torch.export.load()) failed with the following error",
            exc_info=True,
        )
        raise ValueError(
            f"The file {file_path} doesn't correspond to a valid Torchscript module or ExportedProgram. Please verify the file path."
        )


def save(
    module: Any,
    file_path: str = "",
    *,
    output_format: str = "exported_program",
    inputs: Optional[Sequence[torch.Tensor | Input]] = None,
    arg_inputs: Optional[Sequence[torch.Tensor | Input]] = None,
    kwarg_inputs: Optional[Dict[str, Any]] = None,
    retrace: bool = True,
    use_legacy_exporter: Optional[bool] = None,
    pickle_protocol: int = 2,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Save the model to disk in the specified output format.

    Arguments:
        module (Optional(torch.jit.ScriptModule | torch.export.ExportedProgram | torch.fx.GraphModule | CudaGraphsTorchTensorRTModule)): Compiled Torch-TensorRT module
        inputs (Union[torch.Tensor, torch_tensorrt.Input]): Torch input tensors or Input specifications
        arg_inputs (Tuple[Union[torch.Tensor, torch_tensorrt.Input], ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[str, Union[torch.Tensor, torch_tensorrt.Input]]): Optional, kwarg inputs to the module forward function.
        output_format (str): Format to save the model. Options include exported_program | torchscript | aot_inductor.
        retrace (bool): When the module type is a fx.GraphModule, this option re-exports the graph using torch.export.export(strict=False) to save it.

                For TRT-compiled modules with dynamic shapes, both retrace=True and retrace=False are supported:

                - **retrace=True**: Automatically detects symbolic shape metadata in the TRT module and preserves it
                  without retracing. This is the recommended approach as it maintains the exact symbolic shapes
                  from the original compilation.

                - **retrace=False**: Directly serializes the existing graph metadata without any re-export.
                  This is faster but may not be compatible with all torch.export consumers.

                For static shape models, retrace=True performs a standard torch.export.export() call.

        use_legacy_exporter (Optional[bool]): Override the exporter used when serializing a torch.fx.GraphModule.
                By default (None) the choice is made automatically:

                - ``retrace=False`` always uses the legacy exporter (pure graph surgery, no re-execution).
                - ``retrace=True`` with dynamic shapes uses ``torch.export.export`` on the inlined graph,
                  which produces a fully standards-compliant ExportedProgram.

                Set to ``True`` to force the legacy exporter regardless of ``retrace``.
                Set to ``False`` to force ``torch.export.export`` on the inlined graph; this requires
                example inputs and a live CUDA device.

        pickle_protocol (int): The pickle protocol to use to save the model. Default is 2. Increase this to 4 or higher for large models
        dynamic_shapes (Optional[Union[dict[str, Any], tuple[Any, ...]]]): Dynamic shape specifications for re-exporting the model.

                **Method 1: Explicit dynamic_shapes (torch.export style)**

                Provide explicit torch.export.Dim specifications::

                    # For a single input with dynamic batch dimension
                    dyn_batch = torch.export.Dim("batch", min=1, max=32)
                    dynamic_shapes = {"x": {0: dyn_batch}}
                    torch_tensorrt.save(model, "model.ep", arg_inputs=[example_tensor], dynamic_shapes=dynamic_shapes)

                    # For multiple inputs
                    dynamic_shapes = ({"x": {0: dyn_batch}}, {"y": {0: dyn_batch}})

                **Method 2: Inferred from torch_tensorrt.Input**

                Pass torch_tensorrt.Input objects with min/opt/max shapes in arg_inputs/kwarg_inputs,
                and dynamic_shapes will be inferred automatically::

                    inputs = [
                        torch_tensorrt.Input(
                            min_shape=(1, 3, 224, 224),
                            opt_shape=(8, 3, 224, 224),
                            max_shape=(32, 3, 224, 224),
                            name="x"  # Optional: name for better dim naming
                        )
                    ]
                    torch_tensorrt.save(model, "model.ep", arg_inputs=inputs)  # dynamic_shapes inferred!

                **Important Limitations:**

                - Automatic inference creates **separate Dim objects for each input**. If your model requires
                  multiple inputs to share the same dimension (e.g., matching batch sizes), you MUST use
                  Method 1 with explicit shared Dim objects::

                      batch = torch.export.Dim("batch", min=1, max=8)
                      dynamic_shapes = {"x": {0: batch}, "mask": {0: batch}}  # Shared batch dimension

                - Automatic inference is **disabled for mixed Input/Tensor inputs** to avoid spurious
                  equality constraints. Use explicit dynamic_shapes for these cases.

                - If both dynamic_shapes and Input objects are provided, the explicit dynamic_shapes
                  parameter takes precedence.
    """
    if isinstance(module, CudaGraphsTorchTensorRTModule):
        module = module.compiled_module
    module_type = _parse_module_type(module)
    accepted_formats = {"exported_program", "torchscript", "aot_inductor"}
    if arg_inputs is not None and not all(
        isinstance(input, (torch.Tensor, Input)) for input in arg_inputs
    ):
        raise ValueError(
            "Not all inputs provided are torch.Tensor or torch_tensorrt.Input objects. Please provide inputs of a valid type"
        )
    if arg_inputs and inputs:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )

    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    if kwarg_inputs and any(value is None for value in kwarg_inputs.values()):
        raise ValueError("kwargs should not include None.")

    def _all_are_input_objects(obj: Any) -> bool:
        """Recursively check if all elements in nested collections are Input objects."""
        if isinstance(obj, Input):
            return True
        elif isinstance(obj, (list, tuple)):
            return all(_all_are_input_objects(item) for item in obj)
        elif isinstance(obj, dict):
            return all(_all_are_input_objects(value) for value in obj.values())
        else:
            # Not an Input object or collection
            return False

    all_inputs_are_input_objects = _all_are_input_objects(arg_inputs)
    if kwarg_inputs:
        all_inputs_are_input_objects = (
            all_inputs_are_input_objects and _all_are_input_objects(kwarg_inputs)
        )

    # Infer dynamic_shapes from Input objects if not explicitly provided
    # Only infer if ALL inputs are Input objects (not mixed with Tensors)
    #
    # Why? When we have mixed Input/Tensor inputs, torch.export may detect that
    # a dynamic Input's dimension always equals a static Tensor's dimension during
    # tracing, and enforce an equality constraint. Since we create separate Dim
    # objects for each input, this causes a constraint violation. Users must use
    # explicit dynamic_shapes for these cases.

    # Warn if user provides both dynamic_shapes and Input objects with dynamic shapes

    arg_tensors: Tuple[torch.Tensor | int, ...] = ()
    kwarg_tensors: Dict[str, Any] = {}

    if all_inputs_are_input_objects:
        if dynamic_shapes is not None:
            has_dynamic_input_objects = any(
                isinstance(inp, Input) and inp.shape_mode == Input._ShapeMode.DYNAMIC
                for inp in arg_inputs  # type: ignore[union-attr]
            )
            if kwarg_inputs:
                has_dynamic_input_objects = has_dynamic_input_objects or any(
                    isinstance(inp, Input)
                    and inp.shape_mode == Input._ShapeMode.DYNAMIC
                    for inp in kwarg_inputs.values()
                )
            if has_dynamic_input_objects:
                logger.warning(
                    "Both explicit dynamic_shapes and torch_tensorrt.Input objects with min/opt/max shapes were provided. "
                    "The explicit dynamic_shapes parameter takes precedence and Input shape specifications will be ignored."
                )
        else:
            inferred_dynamic_shapes = get_dynamic_shapes_args(module, arg_inputs)
            inferred_dynamic_shapes.update(get_dynamic_shapes_kwargs(kwarg_inputs))

            if inferred_dynamic_shapes is not None:
                dynamic_shapes = inferred_dynamic_shapes
                logger.info(
                    f"Inferred dynamic_shapes from torch_tensorrt.Input objects with min/opt/max specifications: {dynamic_shapes}"
                )

        arg_tensors = tuple(get_torch_inputs(arg_inputs, default_device()))  # type: ignore
        kwarg_tensors = get_torch_inputs(kwarg_inputs, default_device())  # type: ignore

    else:
        # Mixed case: some inputs are Tensors, some are Input objects
        # Extract tensors from Input objects and use provided tensors as-is
        def _extract_tensor(obj: Any) -> Any:
            """Recursively extract tensors from Input objects or pass through tensors."""
            if isinstance(obj, Input):
                if (
                    obj.shape_mode == Input._ShapeMode.DYNAMIC
                    and dynamic_shapes is None
                ):
                    logger.warning(
                        "Mixed torch.Tensor and torch_tensorrt.Input objects provided in the example arguments without explicit dynamic_shapes. "
                        "We cannot infer the dynamic shape specs from these mixed cases "
                        "Consider providing explicit dynamic_shapes parameter or using Input objects for all inputs."
                    )
                return obj.example_tensor()
            elif isinstance(obj, torch.Tensor):
                return obj
            elif isinstance(obj, (list, tuple)):
                extracted = [_extract_tensor(item) for item in obj]
                return type(obj)(extracted)
            elif isinstance(obj, dict):
                return {key: _extract_tensor(value) for key, value in obj.items()}
            else:
                raise TypeError(
                    f"Unsupported input type: {type(obj)}. Expected torch.Tensor or torch_tensorrt.Input"
                )

        arg_tensors = _extract_tensor(arg_inputs) if arg_inputs is not None else ()
        kwarg_tensors = (
            _extract_tensor(kwarg_inputs) if kwarg_inputs is not None else {}
        )

    # Extract tensors from Input objects for actual execution
    # When inferring dynamic shapes, use different sizes for args vs kwargs to avoid
    # torch.export detecting spurious equality constraints

    if output_format not in accepted_formats:
        raise ValueError(
            f"Provided output_format {output_format} is not supported. Supported options are exported_program | torchscript"
        )
    if output_format == "aot_inductor" and platform.system() != "Linux":
        raise ValueError(
            f"The AOT Inductor format is only supported on Linux, {platform.system()} is not a supported platform for this format"
        )
    if not file_path:
        raise ValueError("File path cannot be empty. Please provide a valid file path")

    if module_type == _ModuleType.nn:
        raise ValueError(
            "Input model is of type nn.Module. Saving nn.Module directly is not supported. Supported model types torch.jit.ScriptModule | torch.fx.GraphModule | torch.export.ExportedProgram."
        )
    elif module_type == _ModuleType.ts:
        if not all(output_format == f for f in ["exported_program", "aot_inductor"]):
            raise ValueError(
                "Provided model is a torch.jit.ScriptModule but the output_format specified is not torchscript. Other output formats are not supported"
            )
        else:
            torch.jit.save(module, file_path)
    elif module_type == _ModuleType.ep:
        if output_format == "torchscript":
            raise ValueError(
                "Provided model is a torch.export.ExportedProgram but the output_format specified is torchscript. Please verify the output_format"
            )
        else:
            if arg_inputs is not None:
                logger.warning(
                    "Provided model is a torch.export.ExportedProgram, inputs or arg_inputs is not necessary during save, it uses the inputs or arg_inputs provided during export and compile"
                )
            if output_format == "exported_program":
                torch.export.save(module, file_path, pickle_protocol=pickle_protocol)
            elif output_format == "aot_inductor":
                inductor_configs = {}
                if "inductor_configs" in kwargs:
                    inductor_configs = kwargs["inductor_configs"]

                torch._inductor.aoti_compile_and_package(
                    module,
                    inductor_configs=inductor_configs,
                    package_path=file_path,
                )
            else:
                raise RuntimeError(
                    "Attempted to serialize an exported program with an unsupported format. Exported programs support exported_program and aot_inductor"
                )
    elif module_type == _ModuleType.fx:
        # The module type is torch.fx.GraphModule
        if output_format == "torchscript":
            module_ts = torch.jit.trace(
                module, arg_inputs, example_kwarg_inputs=kwarg_inputs
            )
            torch.jit.save(module_ts, file_path)
        else:
            if not retrace:
                from torch_tensorrt.dynamo._exporter import export

                if arg_inputs is not None:
                    logger.warning(
                        "Provided model is a torch.fx.GraphModule and retrace is False, inputs or arg_inputs is not necessary during save."
                    )

                # Default for retrace=False is the legacy exporter (pure graph surgery,
                # no re-execution). Override with use_legacy_exporter if provided.
                _use_legacy = (
                    use_legacy_exporter if use_legacy_exporter is not None else True
                )
                exp_program = export(
                    module,
                    arg_inputs=arg_tensors,
                    kwarg_inputs=kwarg_tensors,
                    dynamic_shapes=dynamic_shapes,
                    use_legacy_exporter=_use_legacy,
                )
                if output_format == "exported_program":
                    torch.export.save(
                        exp_program, file_path, pickle_protocol=pickle_protocol
                    )
                elif output_format == "aot_inductor":
                    inductor_configs = {}
                    if "inductor_configs" in kwargs:
                        inductor_configs = kwargs["inductor_configs"]

                    torch._inductor.aoti_compile_and_package(
                        exp_program,
                        inductor_configs=inductor_configs,
                        package_path=file_path,
                    )
                else:
                    raise RuntimeError(
                        "Attempted to serialize an exported program with an unsupported format. Exported programs support exported_program and aot_inductor"
                    )
            else:
                # When retrace=True with a TRT-compiled GraphModule that has dynamic shapes,
                # use torch.export.export on the inlined graph to get a fully
                # standards-compliant ExportedProgram. Override with use_legacy_exporter
                # if provided.
                has_symbolic_metadata = any(
                    isinstance(dim, torch.SymInt)
                    for node in module.graph.nodes
                    if node.op == "placeholder" and "val" in node.meta
                    for dim in getattr(node.meta["val"], "shape", [])
                )
                if has_symbolic_metadata and dynamic_shapes is not None:
                    from torch_tensorrt.dynamo._exporter import export

                    if arg_inputs is not None:
                        logger.info(
                            "Provided model is a torch.fx.GraphModule with dynamic shapes and retrace is True. "
                            "Using existing symbolic metadata instead of retracing. Input specs are not necessary."
                        )
                    # Default for this path is the non-legacy exporter.
                    _use_legacy = (
                        use_legacy_exporter
                        if use_legacy_exporter is not None
                        else False
                    )
                    exp_program = export(
                        module,
                        arg_inputs=arg_tensors,
                        kwarg_inputs=kwarg_tensors,
                        dynamic_shapes=dynamic_shapes,
                        use_legacy_exporter=_use_legacy,
                    )
                else:
                    # Regular GraphModule or no dynamic shapes - retrace normally
                    if has_symbolic_metadata:
                        logger.warning(
                            "The provided module has symbolic metadata and retrace is True, however there is no dynamic shapes information available either explicitly or derived from arg/kwarg inputs (torch_tensorrt.Input) "
                            "This may lead to incorrect tracing and overly restrictive shape guards when the exported program is loaded. Please specify the dynamic shapes either explicitly or derived from arg/kwarg inputs"
                        )

                    if (arg_inputs is None or arg_inputs == ()) and (
                        kwarg_tensors is None or kwarg_tensors == {}
                    ):
                        raise ValueError(
                            "Provided model is a torch.fx.GraphModule without existing shape metadata and retrace is True, however no inputs specs were provided. "
                            "Please provide valid torch.Tensors or torch_tensorrt.Input objects as inputs to retrace and save the model"
                        )

                    exp_program = torch.export.export(
                        module,
                        args=tuple(arg_tensors),
                        kwargs=kwarg_tensors,
                        dynamic_shapes=dynamic_shapes,
                        strict=False,
                    )

                if output_format == "exported_program":
                    torch.export.save(
                        exp_program, file_path, pickle_protocol=pickle_protocol
                    )
                elif output_format == "aot_inductor":
                    inductor_configs = {}
                    if "inductor_configs" in kwargs:
                        inductor_configs = kwargs["inductor_configs"]

                    torch._inductor.aoti_compile_and_package(
                        exp_program,
                        inductor_configs=inductor_configs,
                        package_path=file_path,
                    )
                else:
                    raise RuntimeError(
                        "Attempted to serialize an exported program with an unsupported format. Exported programs support exported_program and aot_inductor"
                    )
