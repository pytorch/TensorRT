# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import ENABLED_FEATURES

from .harness import DispatchTestCase


class MaskedSelect(nn.Module):
    def forward(self, x, mask):
        return torch.ops.aten.masked_select.default(x, mask)


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "masked_select needs nonzero, which tensorrt_rtx does not support",
)
class TestMaskedSelectConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("same_shape", torch.randn(4, 16), torch.rand(4, 16) > 0.5),
            ("mask_broadcasts", torch.randn(4, 16), torch.rand(1, 16) > 0.5),
            ("input_broadcasts", torch.randn(16), torch.rand(4, 16) > 0.5),
            ("both_broadcast", torch.randn(4, 1), torch.rand(1, 5) > 0.5),
            ("row_mask", torch.randn(6, 8), torch.rand(6, 1) > 0.5),
            ("3d", torch.randn(2, 3, 4), torch.rand(2, 3, 4) > 0.5),
            (
                "int32_input",
                torch.randint(-10, 10, (3, 4), dtype=torch.int32),
                torch.rand(3, 4) > 0.5,
            ),
        ]
    )
    def test_masked_select(self, _, x, mask):
        self.run_test(
            MaskedSelect(),
            [x, mask],
            use_dynamo_tracer=True,
            enable_passes=True,
            require_full_compilation=True,
        )


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "masked_select needs nonzero, which tensorrt_rtx does not support",
)
class TestMaskedSelectEndToEnd(TestCase):
    """The output size of masked_select depends on the mask's values, so one engine must
    return a different number of elements for each mask it is given."""

    def _compile(self, inputs, dynamic_shapes=None):
        exported = torch.export.export(
            MaskedSelect().eval().cuda(), tuple(inputs), dynamic_shapes=dynamic_shapes
        )
        return torch_tensorrt.dynamo.compile(
            exported, arg_inputs=list(inputs), min_block_size=1
        )

    def _assert_runs_in_tensorrt(self, compiled):
        children = dict(compiled.named_children())
        self.assertTrue(
            any("_run_on_acc" in name for name in children),
            f"No TensorRT engine was built. Submodules: {list(children)}",
        )
        # TensorRT engines are TorchTensorRTModules. A GraphModule child, or the top-level
        # graph, is code that still runs in PyTorch.
        torch_graphs = [compiled] + [
            m for m in children.values() if isinstance(m, torch.fx.GraphModule)
        ]
        torch_ops = {
            node.target
            for graph_module in torch_graphs
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        }
        self.assertNotIn(torch.ops.aten.masked_select.default, torch_ops)

    def test_output_length_follows_the_mask(self):
        x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], device="cuda")
        mask = torch.tensor([[True, False, True], [False, True, False]], device="cuda")
        compiled = self._compile([x, mask])
        self._assert_runs_in_tensorrt(compiled)

        for rows, expected in [
            ([[True, False, True], [False, True, False]], [0.0, 2.0, 4.0]),
            ([[False, False, False], [False, False, True]], [5.0]),
            ([[True, True, True], [True, True, True]], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ]:
            out = compiled(x, torch.tensor(rows, device="cuda"))
            torch.testing.assert_close(out, torch.tensor(expected, device="cuda"))

    def test_all_false_mask_gives_an_empty_result(self):
        x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], device="cuda")
        compiled = self._compile([x, x > 2.5])
        self._assert_runs_in_tensorrt(compiled)

        out = compiled(x, torch.zeros(2, 3, dtype=torch.bool, device="cuda"))
        self.assertEqual(tuple(out.shape), (0,))

    def test_dynamic_batch(self):
        batch = torch.export.Dim("batch", min=1, max=8)
        x = torch.randn(4, 16, device="cuda")
        compiled = self._compile([x, x > 0], dynamic_shapes=({0: batch}, {0: batch}))
        self._assert_runs_in_tensorrt(compiled)

        for batch_size in (1, 6):
            xb = torch.randn(batch_size, 16, device="cuda")
            torch.testing.assert_close(
                compiled(xb, xb > 0), torch.masked_select(xb, xb > 0)
            )


if __name__ == "__main__":
    run_tests()
