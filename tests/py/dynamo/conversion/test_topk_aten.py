# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 1, 0, True, True),
            ((3, 3, 4), 2, -1, True, True),
            ((3, 3, 4), 2, -1, False, True),
            ((3850, 2), 3840, 0, False, True),
            ((3, 3), 2, 0, True, True),
            ((3, 3), 2, 1, True, False),
            ((5, 3), 2, 1, False, False),
            ((6, 4), 2, 1, False, False),
            # default dim:-1 largest:True, sorted:True
            ((3, 5, 12), 3),
            # ITopKLayer requires rank >= 2; converter broadcasts rank-1
            ((64,), 64, 0, True, True),
            ((64,), 8, 0, False, True),
            ((16,), 4, -1, True, True),
        ]
    )
    def test_topk(self, input_shape, k, dim=-1, largest=True, sorted=True):
        class Topk(nn.Module):
            def forward(self, x):
                return torch.ops.aten.topk.default(x, k, dim, largest, sorted)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Topk(),
            inputs,
            enable_passes=True,
        )

    def test_topk_of_0d_tensor(self):
        # x[0] * 2.0 is 0-d, and so are its top value and its index.
        class Topk(nn.Module):
            def forward(self, x):
                y = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 0), 2.0)
                return torch.ops.aten.topk.default(y, 1)

        self.run_test(
            Topk(),
            [torch.randn(4)],
        )

    def test_topk_0_of_0d_tensor_runs_in_pytorch(self):
        # PyTorch gives 0-d results for k=0 too, which a TopK layer cannot make.
        class Topk(nn.Module):
            def forward(self, x):
                return torch.ops.aten.topk.default(x[0] * 2.0, 0)

        x = torch.randn(4, device="cuda")
        compiled = torch_tensorrt.compile(
            Topk(), inputs=[x], ir="dynamo", min_block_size=1
        )
        # Only the submodules that run in PyTorch are graph modules.
        self.assertIn(
            torch.ops.aten.topk.default,
            [
                node.target
                for module in compiled.modules()
                if isinstance(module, torch.fx.GraphModule)
                for node in module.graph.nodes
            ],
        )
        # On CUDA, PyTorch leaves the k=0 results unwritten, so only their shapes and
        # dtypes can be compared.
        for result, expected in zip(compiled(x), Topk()(x)):
            self.assertEqual(
                (result.shape, result.dtype), (expected.shape, expected.dtype)
            )


if __name__ == "__main__":
    run_tests()
