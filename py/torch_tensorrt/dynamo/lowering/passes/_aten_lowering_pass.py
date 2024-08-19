import logging
from typing import Callable, Optional, Sequence, Union

import torch

from .constant_folding import constant_fold
from .fuse_prims_broadcast import fuse_prims_broadcast
from .lower_linear import lower_linear
from .lower_scaled_dot_product_attention import lower_scaled_dot_product_attention
from .pass_manager import DynamoPassManager
from .remove_assert_scalar import remove_assert_scalar
from .remove_detach import remove_detach
from .remove_input_alias_fixing_clones import remove_input_alias_fixing_clones
from .repair_input_as_output import repair_input_as_output
from .replace_full_like_with_full import replace_full_like_with_full
from .replace_max_pool_with_indices import replace_max_pool_with_indices
from .view_to_reshape import view_to_reshape

ATEN_POST_LOWERING_PASSES = DynamoPassManager.build_from_passlist(
    [
        remove_input_alias_fixing_clones,
        constant_fold,
        repair_input_as_output,
        lower_scaled_dot_product_attention,
        lower_linear,
        fuse_prims_broadcast,
        replace_max_pool_with_indices,
        replace_full_like_with_full,
        view_to_reshape,
        remove_assert_scalar,
    ]
)

ATEN_PRE_LOWERING_PASSES = DynamoPassManager.build_from_passlist(
    [
        remove_detach,
    ]
)

logger = logging.getLogger(__name__)


LoweringPassSignature = Callable[
    [torch.fx.GraphModule, Sequence[torch.Tensor]], torch.fx.GraphModule
]


def _aten_lowering_pass(
    *args: LoweringPassSignature,
    index: Optional[int] = None,
) -> Union[
    LoweringPassSignature, Callable[[LoweringPassSignature], LoweringPassSignature]
]:
    """Adds a lowering pass to the registry, at a specified index if desired

    If no index is specified, the lowering pass is inserted at the end of the list
    """

    def add_lowering_pass(
        lowering_pass: LoweringPassSignature,
    ) -> LoweringPassSignature:
        ATEN_POST_LOWERING_PASSES.add_pass_with_index(lowering_pass, index)
        logger.debug(
            f"Added lowering pass {lowering_pass} to list at index {index}, current passlist: {ATEN_POST_LOWERING_PASSES}"
        )
        return lowering_pass

    # If there are arguments specified, the decorator may have been called as-is
    if args:
        # The decorator may only be called with the lowering pass
        # The index must be specified as a keyword argument
        if len(args) == 1 and callable(args[0]):
            return add_lowering_pass(args[0])
        else:
            raise AssertionError(
                f"aten_lowering_pass decorator called with invalid arguments {args} "
                "To specify an index to insert the pass, use the keyword 'index='"
            )
    # If no arguments are specified, the decorator was called with an index keyword
    else:
        return add_lowering_pass


def _remove_lowering_pass(*, index: int) -> None:
    """Removes a lowering pass at a specific index from the registry"""
    ATEN_POST_LOWERING_PASSES.remove_pass_with_index(index)
    logger.debug(
        f"Removed lowering pass at index {index}, current passlist: {ATEN_POST_LOWERING_PASSES}"
    )
    return


def post_lowering(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module after torch.export/ torch.compile and their decompositions, returns the modified GraphModule"""
    logging.debug(
        f"Invoking DynamoPassManager and applying lowering passes: {ATEN_POST_LOWERING_PASSES}"
    )
    return ATEN_POST_LOWERING_PASSES(gm)


def pre_export_lowering(ep: torch.export.ExportedProgram) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module after torch.export/ torch.compile and their decompositions, returns the modified GraphModule"""
    logging.debug(
        f"Invoking DynamoPassManager and applying lowering passes: {ATEN_PRE_LOWERING_PASSES}"
    )
    gm = ep.graph_module
    gm = ATEN_PRE_LOWERING_PASSES(gm)
    return ep


def dump_lowering_passes() -> str:
    """Returns a string containing the lowering passes"""
    return str(ATEN_POST_LOWERING_PASSES)
