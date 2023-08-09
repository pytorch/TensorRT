import copy
import sys
import torch
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from packaging import version
from torch._functorch.aot_autograd import _aot_export_function
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
import unittest.mock
from torch_tensorrt.dynamo.lowering import get_decompositions

def trace(model, inputs, **kwargs):
    # Place backend tracing within FakeTensor context allowing nonfake Tensors
    # fake_mode = detect_fake_mode(inputs)
    # if fake_mode is None:
    #     fake_mode = FakeTensorMode()
    
    # with unittest.mock.patch.object(
    #     fake_mode, "allow_non_fake_inputs", True
    # ), fake_mode:
    #     # Invoke AOTAutograd to translate operators to aten
    #     # torch._functorch.config.debug_assert = False
    #     graph_module = aot_export_joint_simple(
    #         model,
    #         inputs, 
    #         trace_joint=False,
    #         decompositions=get_decompositions(),
    #     )
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        graph_module, _, _, _ = _aot_export_function(model,
                inputs,
                decompositions=get_decompositions(),)
    # import pdb; pdb.set_trace()
    return graph_module