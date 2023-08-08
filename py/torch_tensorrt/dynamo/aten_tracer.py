import copy
import sys
import torch
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from packaging import version
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch._guards import detect_fake_mode
import unittest.mock
from torch_tensorrt.dynamo.lowering import get_decompositions

def trace(model, inputs, **kwargs):
    # Place backend tracing within FakeTensor context allowing nonfake Tensors
    fake_mode = detect_fake_mode(inputs)
    import pdb; pdb.set_trace()
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
    # Invoke AOTAutograd to translate operators to aten
        graph_module, _ = aot_export_joint_simple(
            model,
            inputs, 
            trace_joint=False,
            decompositions=get_decompositions(),
        )

    return graph_module