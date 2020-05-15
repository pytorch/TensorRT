from typing import List, Dict, Any
import torch
import trtorch._C
from trtorch._extra_info import _parse_extra_info
from trtorch._version import __version__

def compile(module: torch.jit.ScriptModule, extra_info: Any) -> torch.jit.ScriptModule:
    """Compile a TorchScript module for NVIDIA GPUs using TensorRT

    Takes a existing TorchScript module and a set of settings to configure the compiler
    and will convert methods to JIT Graphs which call equivalent TensorRT engines

    Converts specifically the forward method of a TorchScript Module

    Args:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        extra_info (dict): Compilation settings including operating precision, target device, etc.
            One key is required which is ``input_shapes``, describing the input sizes or ranges for inputs
            to the graph. All other keys are optional

            .. code-block:: py

                ExtraInfo = {
                    "input_shapes": [
                        (1, 3, 224, 224), # Static input shape for input #1
                        {
                            "min": (1, 3, 224, 224),
                            "opt": (1, 3, 512, 512),
                            "max": (1, 3, 1024, 1024)
                        } # Dynamic input shape for input #2
                    ],
                    "op_precision": torch.half, # Operating precision set to FP16
                    "refit": false, # enable refit
                    "debug": false, # enable debuggable engine
                    "strict_types": false, # kernels should strictly run in operating precision
                    "allow_gpu_fallback": false, # (DLA only) Allow layers unsupported on DLA to run on GPU
                    "device": torch.device("cuda"), # Type of device to run engine on (for DLA use trtorch.DeviceType.DLA)
                    "capability": trtorch.EngineCapability.DEFAULT, # Restrict kernel selection to safe gpu kernels or safe dla kernels
                    "num_min_timing_iters": 2, # Number of minimization timing iterations used to select kernels
                    "num_avg_timing_iters": 1, # Number of averaging timing iterations used to select kernels
                    "workspace_size": 0, # Maximum size of workspace given to TensorRT
                    "max_batch_size": 0, # Maximum batch size (must be >= 1 to be set, 0 means not set)
                }

            Input Sizes can be specified as torch sizes, tuples or lists. Op precisions can be specified using
            torch datatypes or trtorch datatypes and you can use either torch devices or the trtorch device type enum
            to select device type.

    Returns:
        torch.jit.ScriptModule: Compiled TorchScript Module, when run it will execute via TensorRT
    """
    compiled_cpp_mod = trtorch._C._compile_graph(module._c, _parse_extra_info(extra_info))
    compiled_module = torch.jit._recursive.wrap_cpp_module(compiled_cpp_mod)
    return compiled_module

def convert_method_to_trt_engine(module: torch.jit.ScriptModule, method_name: str, extra_info: Any) -> str:
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Args:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        method_name (str): Name of method to convert
        extra_info (dict): Compilation settings including operating precision, target device, etc.
            One key is required which is ``input_shapes``, describing the input sizes or ranges for inputs
            to the graph. All other keys are optional

            .. code-block:: py

                ExtraInfo = {
                    "input_shapes": [
                        (1, 3, 224, 224), # Static input shape for input #1
                        {
                            "min": (1, 3, 224, 224),
                            "opt": (1, 3, 512, 512),
                            "max": (1, 3, 1024, 1024)
                        } # Dynamic input shape for input #2
                    ],
                    "op_precision": torch.half, # Operating precision set to FP16
                    "refit": false, # enable refit
                    "debug": false, # enable debuggable engine
                    "strict_types": false, # kernels should strictly run in operating precision
                    "allow_gpu_fallback": false, # (DLA only) Allow layers unsupported on DLA to run on GPU
                    "device": torch.device("cuda"), # Type of device to run engine on (for DLA use trtorch.DeviceType.DLA)
                    "capability": trtorch.EngineCapability.DEFAULT, # Restrict kernel selection to safe gpu kernels or safe dla kernels
                    "num_min_timing_iters": 2, # Number of minimization timing iterations used to select kernels
                    "num_avg_timing_iters": 1, # Number of averaging timing iterations used to select kernels
                    "workspace_size": 0, # Maximum size of workspace given to TensorRT
                    "max_batch_size": 0, # Maximum batch size (must be >= 1 to be set, 0 means not set)
                }

            Input Sizes can be specified as torch sizes, tuples or lists. Op precisions can be specified using
            torch datatypes or trtorch datatypes and you can use either torch devices or the trtorch device type enum
            to select device type.

    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    return trtorch._C._convert_graph_to_trt_engine(module._c, method_name, _parse_extra_info(extra_info))

def check_method_op_support(module: torch.jit.ScriptModule, method_name: str) -> bool:
    """Checks to see if a method is fully supported by TRTorch

    Checks if a method of a TorchScript module can be compiled by TRTorch, if not, a list of operators
    that are not supported are printed out and the function returns false, else true.

    Args:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        method_name (str): Name of method to check

    Returns:
        bool: True if supported Method
    """
    return trtorch._C._check_method_op_support(module._c, method_name)

def dump_build_info():
    """Prints build information about the TRTorch distribution to stdout
    """
    print(get_build_info())

def get_build_info() -> str:
    """Returns a string containing the build information of TRTorch distribution

    Returns:
        str: String containing the build information for TRTorch distribution
    """
    build_info = trtorch._C._get_build_info()
    build_info = "TRTorch Version: " + str(__version__) + '\n' + build_info
    return build_info

