# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions, pre_export_lowering
from torch_tensorrt.dynamo.lowering.passes.replace_interleave_stack_with_cat import (
    replace_interleave_stack_with_cat,
)

LOGGER = "torch_tensorrt.dynamo.lowering.passes.replace_interleave_stack_with_cat"
ON = CompilationSettings(replace_interleave_stack_with_cat=True)

STACK = torch.ops.aten.stack.default
UNSQUEEZE = torch.ops.aten.unsqueeze.default
CAT = torch.ops.aten.cat.default
RESHAPE = torch.ops.aten._reshape_copy.default


class Stack(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn(a, b)


def _interleave(a, b):
    return torch.stack([a, b], 2).flatten(1, 2)


def _interleave_without_grad(a, b):
    with torch.no_grad():
        return torch.stack([a, b], -2).flatten(-3, -2)


def _stack_without_grad(a, b):
    with torch.no_grad():
        return torch.stack([a, b], 1).flatten(0, 1)


def _two_users(a, b):
    stacked = torch.stack([a, b], 2)
    return stacked.flatten(1, 2), stacked + 1


def _targets(ep):
    return [n.target for n in ep.graph.nodes if n.op == "call_function"]


def _inputs(device="cpu"):
    return tuple(torch.randn(2, 3, 4, 5, device=device) for _ in range(2))


class TestReplaceInterleaveStackWithCat(unittest.TestCase):
    def test_off_by_default(self):
        gm = torch.export.export(Stack(_interleave), _inputs()).graph_module
        before = gm.code

        replace_interleave_stack_with_cat(gm, CompilationSettings())

        self.assertEqual(gm.code, before)

    def test_interleave_in_no_grad_region(self):
        gm = torch.export.export(
            Stack(_interleave_without_grad), _inputs()
        ).graph_module

        replace_interleave_stack_with_cat(gm, ON)

        targets = [
            n.target
            for m in gm.modules()
            if isinstance(m, torch.fx.GraphModule)
            for n in m.graph.nodes
        ]
        self.assertNotIn(STACK, targets)
        self.assertEqual(targets.count(UNSQUEEZE), 2)

    @parameterized.expand(
        [
            ("merge_into_previous_dim_2", _interleave),
            (
                "merge_into_previous_dim_3",
                lambda a, b: torch.stack([a, b], 3).flatten(2, 3),
            ),
            (
                "merge_into_next_dim_1",
                lambda a, b: torch.stack([a, b], 1).flatten(1, 2),
            ),
        ]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_interleave(self, _, fn):
        module = Stack(fn)
        inputs = _inputs("cuda")
        ep = torch.export.export(module, inputs)
        lowered = pre_export_lowering(ep, ON)
        self.assertEqual(
            _targets(lowered.run_decompositions(get_decompositions(False))),
            [UNSQUEEZE, UNSQUEEZE, CAT, RESHAPE],
        )
        # Lowering works on a copy, so the given program keeps its stack.
        self.assertIn(STACK, _targets(ep))

        with self.assertLogs(LOGGER, level="DEBUG"):
            trt_module = torch_tensorrt.dynamo.compile(
                torch.export.export(module, inputs),
                inputs,
                min_block_size=1,
                replace_interleave_stack_with_cat=True,
            )
        torch.testing.assert_close(trt_module(*inputs), module(*inputs), rtol=0, atol=0)

    @parameterized.expand(
        [
            (
                "reshape_without_merge",
                lambda a, b: torch.stack([a, b], 2).flatten(3, 4),
            ),
            (
                "reshape_adding_an_axis",
                lambda a, b: torch.stack([a, b], 2).reshape(2, 6, 4, 5, 1),
            ),
            ("one_input_reduced", lambda a, b: torch.stack([a], 2).sum(2)),
            ("two_users", _two_users),
            (
                "channels_last_inputs",
                lambda a, b: torch.stack(
                    [t.contiguous(memory_format=torch.channels_last) for t in (a, b)], 1
                ).flatten(1, 2),
            ),
            (
                "inside_vmap",
                lambda a, b: torch.vmap(
                    lambda x, y: torch.stack([x, y], 1).flatten(0, 1)
                )(a, b),
            ),
            (
                "in_no_grad_region_inside_vmap",
                lambda a, b: torch.vmap(_stack_without_grad)(a, b),
            ),
        ]
    )
    def test_leaves_other_stacks_alone(self, _, fn):
        ep = pre_export_lowering(torch.export.export(Stack(fn), _inputs()), ON)
        targets = [
            n.target
            for m in ep.graph_module.modules()
            if isinstance(m, torch.fx.GraphModule)
            for n in m.graph.nodes
        ]
        self.assertIn(STACK, targets)


if __name__ == "__main__":
    unittest.main()
