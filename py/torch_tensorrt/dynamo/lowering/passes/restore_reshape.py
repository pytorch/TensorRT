# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def restore_reshape(gm: torch.fx.GraphModule) -> None:
    """Turn ``aten._reshape_copy`` back into ``aten.reshape`` wherever PyTorch runs it.

    ``view_decomposition`` rewrites every ``aten.view`` before partitioning, so a view
    TensorRT does not take stays in PyTorch as ``aten._reshape_copy``. That is not a
    core ATen op, and AOTInductor has no lowering for it, so it becomes a fallback
    kernel, which the ExecuTorch CUDA backend cannot run. ``aten.reshape`` decomposes
    into core ops wherever it goes.
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
        if nodes:
            module.recompile()
