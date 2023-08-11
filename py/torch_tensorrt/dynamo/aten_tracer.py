from __future__ import annotations

import unittest.mock
from typing import Any, Tuple

import torch
from torch._functorch.aot_autograd import _aot_export_function
from torch._subclasses import FakeTensorMode
from torch_tensorrt.dynamo.lowering import get_decompositions


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
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
