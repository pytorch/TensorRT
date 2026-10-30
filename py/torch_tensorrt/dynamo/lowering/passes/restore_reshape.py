# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def restore_reshape(gm: torch.fx.GraphModule) -> None:
    """Turn ``aten._reshape_copy`` into ``aten.clone(aten.reshape)`` wherever PyTorch runs it.

    ``view_decomposition`` rewrites every ``aten.view`` before partitioning, so a view
    TensorRT does not take stays in PyTorch as ``aten._reshape_copy``. That is not a
    core ATen op, and AOTInductor has no lowering for it, so it becomes a fallback
    kernel, which the ExecuTorch CUDA backend cannot run. ``aten.reshape`` decomposes
    into core ops and ``aten.clone`` is one. The clone keeps the contiguous copy that
    ``aten._reshape_copy`` made: lowering may already have dropped a user's clone of
    an input, and a reshape alone could then return the caller's input or an engine's
    reused output buffer.
    """
    for name, module in gm.named_modules():
        if "_run_on_acc" in name or not isinstance(module, torch.fx.GraphModule):
            continue
        nodes = [
            node
            for node in module.graph.nodes
            if node.target == torch.ops.aten._reshape_copy.default
        ]
        for node in nodes:
            node.target = torch.ops.aten.reshape.default
            users = list(node.users)
            with module.graph.inserting_after(node):
                copy = module.graph.call_function(
                    torch.ops.aten.clone.default,
                    (node,),
                    {"memory_format": torch.contiguous_format},
                )
            copy.meta = dict(node.meta)
            for user in users:
                user.replace_input_with(node, copy)
        if nodes:
            module.recompile()
