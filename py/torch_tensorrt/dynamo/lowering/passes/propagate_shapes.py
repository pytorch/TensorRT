import logging
from typing import Sequence

import torch
from torch.fx.passes.shape_prop import ShapeProp

logger = logging.getLogger(__name__)


def propagate_shapes(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Attempts to propagate shapes through the graph"""

    # Propagate shapes through the graph
    try:
        ShapeProp(gm).propagate(*sample_inputs)
    except (RuntimeError, AssertionError):
        logger.warning(
            "Shape Propagation Failed on Graph, skipping propagate_shapes lowering pass",
            exc_info=True,
        )

    return gm
