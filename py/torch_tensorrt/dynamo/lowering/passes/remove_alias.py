# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def _remove_alias_nodes(gm: torch.fx.GraphModule) -> int:
    count = 0
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.all_input_nodes[0])
            gm.graph.erase_node(node)
            count += 1
    return count


def remove_alias(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove alias ops in the graph

    The decomposition table has no entry for ``aten.alias``, because a
    decomposition that returns its own input breaks functionalization, so alias
    nodes reach the lowered graph. An alias shares storage with its input, so
    using the input instead changes no value.
    """
    count = _remove_alias_nodes(gm)
    if count:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Removed {count} alias nodes:\n{gm.graph}")

    return gm


def remove_alias_from_exported_program(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    """Remove alias ops from an exported program before it is unlifted

    ``ExportedProgram.module()`` returns a written input's ``copy_`` in place of
    an output only when the output and the written value are one node, and two
    aliases of one value would make them two. Buffer lifting also only
    recognizes a cache write that reads the buffer directly.
    """
    graph_module = exported_program.graph_module
    count = _remove_alias_nodes(graph_module)
    if not count:
        return exported_program
    graph_module.recompile()
    signature = exported_program.graph_signature
    output_args = graph_module.graph.output_node().args[0]
    # The graph was edited in place, so the output names have to follow it.
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs),
        output_specs=[
            (
                OutputSpec(spec.kind, TensorArgument(name=arg.name), spec.target)
                if isinstance(spec.arg, TensorArgument)
                else spec
            )
            for spec, arg in zip(signature.output_specs, output_args)
        ],
    )
    logger.debug(f"Removed {count} alias nodes from the exported program")
    return exported_program
