# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAsStridedConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (5, 5),
                (2, 3),
                (1, 2),
                0,
            ),
            (
                (5, 5),
                (2, 3),
                (2, 2),
                1,
            ),
            (
                (20, 20),
                (2, 3, 2),
                (2, 2, 2),
                0,
            ),
            (
                (8, 8, 8),
                (2, 2, 3),
                (1, 2, 2),
                1,
            ),
            (
                (200, 200, 200),
                (9, 9, 3, 2),
                (2, 2, 2, 3),
                1,
            ),
            (
                (10, 25, 12),
                (3, 7, 3),
                (2, 1, 3),
                1,
            ),
            (
                (10, 25, 12),
                (3, 7, 3),
                (2, 0, 3),
                1,
            ),
            (
                (10, 25, 12, 100),
                (6, 5, 7, 10),
                (0, 0, 0, 0),
                0,
            ),
            (
                (10, 25, 12, 100),
                (6, 5, 7, 10),
                (0, 0, 0, 0),
                1,
            ),
        ]
    )
    def test_as_strided(
        self,
        input_shape,
        output_size,
        stride,
        storage_offset=0,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.as_strided.default(
                    x, output_size, stride, storage_offset
                )

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
        )

    def test_as_strided_of_0d_tensor_is_refused(self):
        # x[1] is a view of x, and as_strided reads x[0] from the storage of x.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.select.int(x, 0, 1)
                return torch.ops.aten.as_strided.default(y, (1,), (1,), 0)

        with self.assertRaisesRegex(RuntimeError, "as_strided of a 0-dim tensor"):
            self.run_test(TestModule(), [torch.randn(4)])

    def test_as_strided_of_computed_0d_tensor(self):
        # x[1] * 2.0 is 0-d and has a storage of its own, which as_strided reads.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 1), 2.0)
                return torch.ops.aten.as_strided.default(y, (2,), (0,), 0)

        self.run_test(TestModule(), [torch.randn(4)])

    def test_as_strided_of_a_view_of_a_computed_0d_tensor(self):
        # The view reads the storage of x[1] * 2.0, which holds the product only.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 1), 2.0)
                y = torch.ops.aten.squeeze.dim(
                    torch.ops.aten.unsqueeze.default(y, 0), 0
                )
                return torch.ops.aten.as_strided.default(y, (2,), (0,), 0)

        self.run_test(TestModule(), [torch.randn(4)])

    def test_as_strided_of_0d_tensor_from_a_python_operator(self):
        # torch.fx.symbolic_trace records x[1] * 2.0 as operator.mul, which makes a
        # new tensor too.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.select.int(x, 0, 1) * 2.0
                return torch.ops.aten.as_strided.default(y, (2,), (0,), 0)

        self.run_test(TestModule(), [torch.randn(4)], use_dynamo_tracer=False)


if __name__ == "__main__":
    run_tests()
