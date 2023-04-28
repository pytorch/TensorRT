from enum import Enum
from typing import List, Optional, Callable
from packaging import version

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
import logging
from functorch import make_fx
from functorch.experimental import functionalize
from torch_tensorrt.fx.passes.lower_basic_pass import (
    replace_op_with_indices,
    run_const_fold,
)

from .types import Shape, TRTDataType


_LOGGER: logging.Logger = logging.getLogger(__name__)


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"

    @staticmethod
    def from_str(label: str) -> Optional["LowerPrecision"]:
        if label in ("fp32", "float32", "float", "torch.float32"):
            return LowerPrecision.FP32
        elif label in ("fp16", "float16", "half", "torch.half", "torch.float16"):
            return LowerPrecision.FP16
        elif label in ("int8"):
            return LowerPrecision.INT8
        elif label in ("bf16", "bfloat16", "torch.bfloat16"):
            return LowerPrecision.BF16
        else:
            return None


def torch_dtype_to_trt(
    dtype: torch.dtype, truncate_long_and_double: bool = False
) -> TRTDataType:
    """
    Convert PyTorch data types to TensorRT data types.

    Args:
        dtype (torch.dtype): A PyTorch data type.

    Returns:
        The equivalent TensorRT data type.
    """
    if trt.__version__ >= "7.0" and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.int64:
        if truncate_long_and_double:
            _LOGGER.warn(
                "Detected Int64 Input, Casting to Int32 for TRT Engine Compatibility"
            )
            return trt.int32
        else:
            raise TypeError(
                "Detected Int64 Input which is not supported by tensorrt, enable compilation"
                + "option truncate_long_and_double=True to cast input to Int32 for TRT Engine"
            )
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    elif dtype == torch.float64:
        if truncate_long_and_double:
            _LOGGER.warn(
                "Detected Float64 Input, Casting to Float32 for TRT Engine Compatibility"
            )
            return trt.float32
        else:
            raise TypeError(
                "Detected Float64 Input which is not supported by tensorrt, enable compilation"
                + "option truncate_long_and_double=True to cast input to Float32 for TRT Engine"
            )
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype: TRTDataType) -> torch.dtype:
    """
    Convert TensorRT data types to PyTorch data types.

    Args:
        dtype (TRTDataType): A TensorRT data type.

    Returns:
        The equivalent PyTorch data type.
    """
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= "7.0" and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def get_dynamic_dims(shape: Shape) -> List[int]:
    """
    This function finds the dynamic dimensions in the given
    shape. A dimension is dynamic if it's -1.

    Args:
        shape (Shape): A sequence of integer that represents
            the shape of a tensor.

    Returns:
        A list of integers contains all the dynamic dimensions
        in the given shape
    """
    dynamic_dims = []

    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)

    return dynamic_dims


def proxytensor_trace(mod, inputs):

    mod.eval()

    def f(*inp):
        return mod(*inp)

    mod = make_fx(functionalize(f))(*inputs)

    # Remove const operation. For ex, nn.Linear has transpose operation on weight
    mod.graph.eliminate_dead_code()
    mod = run_const_fold(mod)
    mod = replace_op_with_indices(mod)
    return mod


def req_torch_version(min_torch_version: str = "2.dev"):
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range

    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly

    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable):
        def function_wrapper(*args, **kwargs):
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(torch.__version__)

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    + f"when calling {f}. Detected version {torch.__version__}"
                )
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator
