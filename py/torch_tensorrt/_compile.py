from typing import List, Dict, Any
from torch_tensorrt import _enums
import torch_tensorrt.ts
from torch_tensorrt import logging
import torch
from torch import fx
from enum import Enum


class _IRType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout
    """
    ts = 0
    fx = 1


def _module_ir(module: Any, ir: str) -> _IRType.ts:
    # Possible module types
    module_is_tsable = any(
        isinstance(module, t) for t in [torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction])
    module_is_fxable = any(isinstance(module, t) for t in [torch.nn.Module, torch.fx.GraphModule])

    ir_targets_torchscript = any([ir == opt for opt in ["torchscript", "ts"]])
    ir_targets_fx = ir == "fx"

    if module_is_tsable and ir_targets_torchscript:
        return _IRType.ts
    elif module_is_fxable and ir_targets_fx:
        if isinstance(module, torch.fx.GraphModule):
            raise ValueError("Was given a torch.fx.GraphModule, fx is not currently supported by Torch-TensorRT")
        elif ir_targets_fx:
            raise ValueError("Preferred ir was set to \"fx\" which is currently not supported by Torch-TensorRT")
        else:
            raise ValueError("Torch-TensorRT currently does not support fx")
        # return _IRType.fx
    else:
        if ir == "default":
            # Options are listed in order of preference
            if module_is_tsable:
                logging.log(logging.Level.Info, "ir was set to default, using TorchScript as ir")
                return _IRType.ts
            elif module_is_fxable:
                raise ValueError("Was given a torch.fx.GraphModule, fx is not currently supported by Torch-TensorRT")
                #logging.log(logging.Level.Info, "ir was set to default, using TorchScript as fx")
                #return _IRType.fx
            else:
                raise ValueError("Module was provided with in an unsupported format")
        else:
            raise ValueError("Unknown ir was requested")


def compile(module: Any, ir="default", inputs=[], enabled_precisions=set([_enums.dtype.float]), **kwargs):
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
    target_ir = _module_ir(module, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if isinstance(module, torch.nn.Module):
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.compile(ts_mod, inputs=inputs, enabled_precisions=enabled_precisions, **kwargs)
    elif target_ir == _IRType.fx:
        raise RuntimeError("fx is currently not supported")
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def convert_method_to_trt_engine(module: Any,
                                 method_name: str,
                                 ir="default",
                                 inputs=[],
                                 enabled_precisions=set([_enums.dtype.float]),
                                 **kwargs):
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
    target_ir = _module_ir(module, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if isinstance(module, torch.nn.Module):
            logging.log(
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.convert_method_to_trt_engine(ts_mod,
                                                              method_name,
                                                              inputs=inputs,
                                                              enabled_precisions=enabled_precisions,
                                                              **kwargs)
    elif target_ir == _IRType.fx:
        raise RuntimeError("fx is currently not supported")
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")
