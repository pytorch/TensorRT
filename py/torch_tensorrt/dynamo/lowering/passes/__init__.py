import logging
from typing import Callable, Optional

import torch

# Import and order lowering passes and pass manager
from .constant_folding import constant_fold
from .pass_manager import DynamoPassManager
from .repair_input_as_output import repair_input_as_output

ATEN_LOWERING_PASSES = DynamoPassManager.build_from_passlist(
    [
        constant_fold,
        repair_input_as_output,
    ]
)

logger = logging.getLogger(__name__)


def add_lowering_pass(
    lowering_pass: Callable[[torch.fx.GraphModule], torch.fx.GraphModule],
    index: Optional[int] = None,
) -> None:
    """Adds a lowering pass to the registry, at a specified index if desired

    If no index is specified, the lowering pass is inserted at the end of the list
    """
    ATEN_LOWERING_PASSES.add_pass_with_index(lowering_pass, index)
    logger.debug(
        f"Added lowering pass {lowering_pass} to list at index {index}, current passlist: {ATEN_LOWERING_PASSES}"
    )
    return


def remove_lowering_pass(index: int) -> None:
    """Removes a lowering pass at a specific index from the registry"""
    ATEN_LOWERING_PASSES.remove_pass_with_index(index)
    logger.debug(
        f"Removed lowering pass at index {index}, current passlist: {ATEN_LOWERING_PASSES}"
    )
    return


def apply_lowering_passes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module, returns the modified GraphModule"""
    logging.debug(
        f"Invoking DynamoPassManager and applying lowering passes: {ATEN_LOWERING_PASSES}"
    )
    return ATEN_LOWERING_PASSES(gm)


def dump_lowering_passes() -> str:
    """Returns a string containing the lowering passes"""
    return str(ATEN_LOWERING_PASSES)
