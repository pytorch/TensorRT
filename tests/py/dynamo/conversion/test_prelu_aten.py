# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestPReLUConverter(DispatchTestCase):
    def test_prelu(self):
        class TestModule(nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._prelu_kernel.default(x, weight)

        inputs = [torch.randn(1, 10), torch.randn(1, 1)]
        self.run_test(TestModule(), inputs)

    def test_prelu_of_0d_tensor(self):
        # x[0] * 2.0 is 0-d, and the result takes the shape of the slope, as in PyTorch.
        class TestModule(nn.Module):
            def forward(self, x, weight):
                y = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 0), 2.0)
                return torch.ops.aten._prelu_kernel.default(y, weight)

        inputs = [torch.randn(4), torch.tensor([0.25])]
        self.run_test(TestModule(), inputs)

    def test_prelu_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._prelu_kernel.default(x, weight)

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(1, 2, 3),
                max_shape=(3, 3, 3),
                dtype=torch.float32,
                name="x",
            ),
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(1, 1, 1),
                max_shape=(1, 1, 1),
                dtype=torch.float32,
                name="weight",
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_prelu_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._prelu_kernel.default(x, weight)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 5),
                opt_shape=(1, 2, 3, 5),
                max_shape=(3, 3, 3, 5),
                dtype=torch.float32,
                name="x",
            ),
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(1, 2, 1, 1),
                max_shape=(1, 3, 1, 1),
                dtype=torch.float32,
                name="weight",
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
