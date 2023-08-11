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
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        graph_module, _, _, _ = _aot_export_function(
            model,
            inputs,
            decompositions=get_decompositions(),
        )

    return graph_module
