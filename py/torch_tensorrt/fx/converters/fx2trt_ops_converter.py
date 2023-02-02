# flake8: noqa
import logging
import math
import operator
import warnings
from typing import cast, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch_tensorrt.fx.converters import acc_ops_converters

from ..converter_registry import tensorrt_converter

from ..types import *  # noqa: F403
from torch.fx.immutable_collections import immutable_list
from torch.fx.node import Argument, Target

from ..utils import get_dynamic_dims, torch_dtype_from_trt, torch_dtype_to_trt

from .converter_utils import *  # noqa: F403
import torch_tensorrt.fx.tracer.acc_tracer.acc_utils as acc_utils

_LOGGER: logging.Logger = logging.getLogger(__name__)

class contextGraphObject:
    #call_function op code signature
    def __init__(self, network, target, args, kwargs, name):
        self.network = network
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.name = name
    
    def unpack_argument(self, label):
        _LOGGER.info(f"Unpacking the arguments from the trace using label")
    

    
class contextGraphObjectAccTrace(contextGraphObject):
    
    def __init__(self, network, target, args, kwargs, name):
        super.__init__(network, target, args, kwargs, name)
    
    #In this case index is the label name
    def unpack_argument(self, label):
        super.__init__()
        _LOGGER.info(f"The label is", label)
        return self.kwargs[label]

class contextGraphObjectATenTrace(contextGraphObject):

    def __init__(self, network, target, args, kwargs, name):
        super.__init__(network, target, args, kwargs, name)
    
    #In this case label is the index no
    def unpack_argument(self, label):
        super.__init__()
        _LOGGER.info(f"The label index  is", label)
        return self.args[label]

@tensorrt_converter(torch.ops.aten.add.Tensor)
def convert_add(ctx):
    #input_a = ctx.method_args[0]
    #input_b = ctx.method_args[1]
    input_a = ctx.unpack_argument(0)
    input_b = ctx.unpack_argument(1)
    layer = ctx.network.add_elementwise(input_a, input_b, trt.ElementWiseOperation.SUM)
    output = layer.get_output(0) 
    return output
    
def add_missing_trt_tensors(network, tensors):
    """Creates missing TensorRT tensors as constants and attaches them to the Torch Tensors"""
    with use_shape_wrapping(False):
        trt_tensors = [None] * len(tensors)

        dtype = check_torch_dtype(*tensors)

        for i, t in enumerate(tensors):
            trt_tensor = None

            # GET TRT TENSOR (OR CREATE TRT CONSTANT)

            # get tensor w/ _trt
            # or... add constant for scalar primitive
            if isinstance(t, float) or isinstance(t, int):
                shape = (1,)
                scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
                trt_tensor = network.add_constant(shape, scalar).get_output(0)
            elif hasattr(t, "_trt"):
                trt_tensor = t._trt

            # or... add constant for leaf tensor w/o _trt
            else:

                # remove all preceding ones, these can be re-inserted later when broadcasting
                num_preceding_ones = 0
                for j in range(len(t.shape)):
                    if int(t.shape[j]) == 1:
                        num_preceding_ones += 1
                    else:
                        break
                shape = tuple(t.shape[num_preceding_ones:])

                weight = t.detach().cpu().numpy()
                t._trt = network.add_constant(shape, weight).get_output(0)
                trt_tensor = t._trt


            assert trt_tensor is not None

            trt_tensors[i] = trt_tensor

        return trt_tensors


def broadcast_trt_tensors(network, trt_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    with use_shape_wrapping(False):
        broadcasted_trt_tensors = [None] * len(trt_tensors)

        for i, t in enumerate(trt_tensors):

            if len(t.shape) < broadcast_ndim:
                # append 1 size dims to front
                diff = broadcast_ndim - len(t.shape)
                shape = tuple([1] * diff + list(t.shape))
                layer = network.add_shuffle(t)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)
            else:
                trt_tensor = t

            broadcasted_trt_tensors[i] = trt_tensor

        return broadcasted_trt_tensors

def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'Tensor data types must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype

class use_shape_wrapping:

    stack = [True] # default true

    def __init__(self, value: bool):
        self._value = value
    
    def __enter__(self, *args, **kwargs):
        self.stack.insert(0, self._value)

    def __exit__(self, *args, **kwargs):
        self.stack.pop(0)