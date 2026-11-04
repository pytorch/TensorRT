# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestSoftmaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (torch.float, False),
            (torch.half, False),
            (torch.half, True),
        ]
    )
    def test_softmax(self, dtype, half_to_float):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 1, half_to_float)

        inputs = [torch.randn(1, 3, 224, 224, dtype=dtype)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            (torch.float, False),
            (torch.half, False),
            (torch.half, True),
        ]
    )
    def test_softmax_with_dynamic_shape(self, dtype, half_to_float):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 2, half_to_float)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(2, 4, 6, 8),
                max_shape=(8, 8, 8, 8),
                dtype=dtype,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_softmax_over_size_one_dim_followed_by_matmul(self):
        skip_if_trt_rtx_turing(self, "aten.mm (an FP32 GEMM)")

        class TestModule(torch.nn.Module):
            def forward(self, x, other):
                softmax = torch.ops.aten._softmax.default(x, 1, False)
                return torch.ops.aten.mm.default(softmax, other)

        x = torch.randn(24, 1)
        x[0, 0] = float("inf")  # PyTorch gives NaN here, not 1
        self.run_test(TestModule(), [x, torch.randn(1, 2)])


if __name__ == "__main__":
    run_tests()
