# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in CPU IO planning for the fixed-shape native benchmark."""

import torch
from torch.utils._pytree import tree_map


def refresh_output_specs(graph_module):
    """Refer to current output tensors after CUDA device-copy insertion.

    Some ExecuTorch revisions retain the pre-copy GPU specs on the output node.
    Refresh only that node: GPU intermediates must remain memory planned even
    when the application owns the final CPU output buffers.
    """
    for node in graph_module.graph.nodes:
        if node.op == "output":
            node.meta["spec"] = tree_map(
                lambda value: (
                    value.meta["spec"] if isinstance(value, torch.fx.Node) else None
                ),
                node.args[0],
            )


def backend_config():
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    class CallerIOMemoryPlanning(MemoryPlanningPass):
        def run(self, graph_module, graph_signature=None):
            refresh_output_specs(graph_module)
            return super().run(graph_module, graph_signature)

    return ExecutorchBackendConfig(
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=CallerIOMemoryPlanning(
            alloc_graph_input=False, alloc_graph_output=False
        ),
    )


def validate_program(program):
    """Fail export if the serialized public IO is still arena allocated."""
    for plan in program.execution_plan:
        for index in plan.inputs + plan.outputs:
            if plan.values[index].val.allocation_info is not None:
                raise ValueError("Export did not preserve caller-owned CPU IO")
