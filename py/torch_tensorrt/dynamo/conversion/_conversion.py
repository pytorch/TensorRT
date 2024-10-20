from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import tensorrt as trt
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    TRTInterpreter,
    TRTInterpreterResult,
)
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

# from torch_tensorrt.dynamo.utils import (
#     get_model_device,
#     get_torch_inputs,
#     unwrap_tensor_shape,
# )

logger = logging.getLogger(__name__)


def get_output_dtypes(output: Any, truncate_doulbe: bool = False) -> List[dtype]:
    output_dtypes = []
    if isinstance(output, torch.fx.node.Node):
        if "val" in output.meta:
            output_meta = output.meta["val"]
            if isinstance(output_meta, (FakeTensor, torch.Tensor)):
                if truncate_doulbe and output_meta.dtype == torch.float64:
                    output_dtypes.append(dtype.float32)
                else:
                    output_dtypes.append(dtype._from(output_meta.dtype))
        else:
            raise ValueError(
                "meta['val'] does not exist, expect meta['val'] exists for each output node"
            )
    elif isinstance(output, tuple):
        for ele in output:
            output_dtypes.extend(get_output_dtypes(ele))
    else:
        raise ValueError(
            f"got type {type(output)}, expected type is a torch.fx.node.Node or a tuple of torch.fx.node.Node"
        )
    return output_dtypes


def infer_module_output_dtypes(
    module: torch.fx.GraphModule,
    # inputs: Sequence[Input],
    # device: Device,
    # kwarg_inputs: Optional[dict[str, Any]] = None,
    truncate_double: bool = False,
) -> List[dtype]:
    """
    This function performs model inference to determine the output shapes and output dtypes
    and truncates them accordingly. inputs can be either arg_inputs or flattened input list.
    If it is flattened list, kwarg_inputs should be None, as it is already included in the flattened input.
    """
    outputs = [node for node in module.graph.nodes if node.op == "output"]
    outputs = outputs[0].args
    return get_output_dtypes(outputs, truncate_double)

    # TODO: We can also determine output dtypes from the module.graph based on node metadata.
    # However, our converter tests use fx.symbolic_trace which sometimes does not provide metadata,
    # so we stick to the model inference approach currently.
    # with unset_fake_temporarily():
    #     # Get the device on which the model exists
    #     # For large models, this can be done on CPU to save GPU memory allocation for TRT.
    #     device = get_model_device(module)
    #     torch_inputs = get_torch_inputs(inputs, device)
    #     if kwarg_inputs is None:
    #         kwarg_inputs = {}
    #     torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
    #     module_outputs = module(*torch_inputs, **torch_kwarg_inputs)
    #     if not isinstance(module_outputs, (list, tuple)):
    #         module_outputs = [module_outputs]

    # Int64 outputs can sometimes be generated from within other operators
    # # such as aten.sum - such outputs can be truncated
    # output_dtypes_ret = []

    # for output_dtype in output_dtypes:

    #     if truncate_double and output_dtype == dtype.float64:
    #         output_dtypes_ret.append(dtype.float32)
    #     else:
    #         output_dtypes_ret.append(dtype._from(output_dtype))

    # return output_shapes, output_dtypes


def interpret_module_to_result(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    arg_inputs: Optional[Sequence[Input]] = None,
    kwarg_inputs: Optional[dict[str, Any]] = None,
    engine_cache: Optional[BaseEngineCache] = None,
) -> TRTInterpreterResult:
    """Interpret an FX module to the output shapes and a TRTInterpreterResult
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of FLATTENED Tensors representing inputs to the module. It should include both
                arg_inputs and kwarg_inputs, if applicable.
        arg_inputs: Sequence of Tensors representing inputs to the module.
        kwarg_inputs: A dictionary of Tensors representing inputs to the module.
        settings: Compilation settings
        engine_cache: Engine cache instance
    Returns:
        (TRTInterpreterResult, List[Tuple[int]])
    """
    output_dtypes = infer_module_output_dtypes(
        module, truncate_double=settings.truncate_double
    )

    interpreter = TRTInterpreter(
        module,
        inputs,
        logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
        output_dtypes=output_dtypes,
        compilation_settings=settings,
        engine_cache=engine_cache,
    )

    interpreter_result = interpreter.run()
    return interpreter_result


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    name: str = "",
    engine_cache: Optional[BaseEngineCache] = None,
) -> PythonTorchTensorRTModule | TorchTensorRTModule:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
        name: TRT engine name
        engine_cache: Engine cache instance
    Returns:
        PythonTorchTensorRTModule or TorchTensorRTModule
    """

    interpreter_result = interpret_module_to_result(
        module, inputs, settings, engine_cache=engine_cache
    )

    rt_cls = PythonTorchTensorRTModule

    if ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime:

        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        rt_cls = TorchTensorRTModule

    elif (
        not ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime
    ):

        logger.info(
            "Since Torch-TensorRT runtime is not available, using Python Runtime, some features may not be available"
        )

    return rt_cls(
        serialized_engine=interpreter_result.serialized_engine,
        input_binding_names=list(interpreter_result.input_names),
        output_binding_names=list(interpreter_result.output_names),
        name=name,
        settings=settings,
        weight_name_map=interpreter_result.weight_name_map,
    )
