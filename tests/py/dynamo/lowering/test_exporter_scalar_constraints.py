# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import io

import pytest
import torch
from torch._guards import detect_fake_mode
from torch_tensorrt.dynamo._exporter import create_trt_exp_program


@pytest.mark.unit
def test_scalar_input_ranges_survive_save_load():
    """Cache indices added by lowering must retain their bounds in the saved program."""

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x + 1

    exported = torch.export.export(
        Identity(),
        (torch.ones(4),),
        dynamic_shapes=({0: torch.export.Dim("length", min=2, max=8)},),
    )
    fake_x = next(n for n in exported.graph.nodes if n.op == "placeholder").meta["val"]
    fake_mode = detect_fake_mode((fake_x,))
    shape_env = fake_mode.shape_env
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = fake_x
    scalars = []
    for name in ("start_idx", "end_idx"):
        node = graph.placeholder(name)
        symbol = shape_env.create_unbacked_symint()
        torch._check(symbol >= 0)
        torch._check(symbol <= 8)
        node.meta["val"] = symbol
        scalars.append(node)
    result = x
    with fake_mode:
        for scalar in scalars:
            node = graph.call_function(torch.ops.aten.add.Scalar, (result, scalar))
            node.meta["val"] = result.meta["val"] + scalar.meta["val"]
            result = node
    graph.output((result,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    program = create_trt_exp_program(gm)

    for node in scalars:
        bounds = program.range_constraints[node.meta["val"].node.expr]
        assert int(bounds.lower) == 0
        assert int(bounds.upper) == 8
    # Existing tensor dimension constraints must also be retained.
    assert (
        program.range_constraints[fake_x.shape[0].node.expr]
        == exported.range_constraints[fake_x.shape[0].node.expr]
    )

    saved = io.BytesIO()
    torch.export.save(program, saved)
    saved.seek(0)
    loaded = torch.export.load(saved)
    assert loaded.range_constraints == program.range_constraints
    for size, start, end in ((3, 0, 4), (7, 4, 8)):
        values = torch.arange(size, dtype=torch.float32)
        torch.testing.assert_close(
            loaded.module()(values, start, end), (values + start + end,)
        )
