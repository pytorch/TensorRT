from __future__ import annotations

import logging
import unittest.mock
from typing import Any, Tuple

import torch
from torch._export import export
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.utils import set_log_level

logger = logging.getLogger(__name__)


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if "debug" in kwargs and kwargs["debug"]:
        set_log_level(logger.parent, logging.DEBUG)

    experimental_decompositions = kwargs.get(
        "enable_experimental_decompositions", False
    )
    with unittest.mock.patch(
        "torch._export.DECOMP_TABLE", get_decompositions(experimental_decompositions)
    ):
        graph_module = export(model, tuple(inputs)).module()
        graph_module = apply_lowering_passes(graph_module, inputs)
    logger.debug("Post export graph: " + str(graph_module.graph))
    return graph_module
