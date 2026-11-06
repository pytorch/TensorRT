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

    @parameterized.expand(
        [
            ("sliced", lambda x: x[:, 2:].unfold(1, 3, 2)),
            ("transposed", lambda x: x.transpose(1, 2).unfold(-1, 4, 2)),
        ]
    )
    def test_as_strided_view_input(self, _, view):
        # unfold decomposes to as_strided with strides into the view's base memory
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return view(x)

        self.run_test(
            TestModule(),
            [torch.randn(2, 20, 3)],
            use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
