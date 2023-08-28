from __future__ import annotations

import logging
import unittest.mock
from typing import Any, Tuple

import torch
from torch._export import export
from torch_tensorrt.dynamo.lowering import get_decompositions

logger = logging.getLogger(__name__)


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if ("debug" in kwargs and kwargs["debug"]) and logger.parent:
        logger.parent.setLevel(logging.DEBUG)
    experimental_decompositions = kwargs.get(
        "enable_experimental_decompositions", False
    )
    with unittest.mock.patch(
        "torch._export.DECOMP_TABLE", get_decompositions(experimental_decompositions)
    ):
        exp_program = export(model, tuple(inputs))

    return exp_program
