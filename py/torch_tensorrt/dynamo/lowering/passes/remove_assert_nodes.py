from typing import Optional
import logging

from torch_tensorrt.dynamo._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo._supports_debugger import fn_supports_debugger
import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

@fn_supports_debugger
def remove_assert_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings, *, _debugger_settings: Optional[DebuggerConfig]=None
) -> torch.fx.GraphModule:
    """Remove assert_scalar ops in the graph"""
    if _debugger_settings is not None and _debugger_settings.break_in_remove_assert_nodes:
        breakpoint()
    count = 0
    for node in gm.graph.nodes:
        if (
            node.target == torch.ops.aten._assert_scalar.default
            or node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"Removed {count} assert_scalar nodes:\n{gm.graph}")

    return gm
