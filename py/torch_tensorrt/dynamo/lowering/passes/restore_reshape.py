# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def restore_reshape(gm: torch.fx.GraphModule) -> None:
    """Turn ``aten._reshape_copy`` into ``aten.view(aten.clone)`` wherever PyTorch runs it.

    ``view_decomposition`` rewrites every ``aten.view`` before partitioning, so a view
    TensorRT does not take stays in PyTorch as ``aten._reshape_copy``. That is not a
    core ATen op, and AOTInductor has no lowering for it, so it becomes a fallback
    kernel, which the ExecuTorch CUDA backend cannot run. A contiguous ``aten.clone``
    followed by ``aten.view`` is what ``aten._reshape_copy`` computes, with one copy,
    and both are core ATen ops. The copy matters: lowering may already have dropped a
    user's clone of an input, and a view alone could then return the caller's input or
    an engine's reused output buffer.
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
            source = node.args[0]
            with module.graph.inserting_before(node):
                copy = module.graph.call_function(
                    torch.ops.aten.clone.default,
                    (source,),
                    {"memory_format": torch.contiguous_format},
                )
            val = getattr(source, "meta", {}).get("val")
            if isinstance(val, torch.Tensor):
                copy.meta["val"] = val.clone(memory_format=torch.contiguous_format)
            node.target = torch.ops.aten.view.default
            node.replace_input_with(source, copy)
        if nodes:
            module.recompile()
