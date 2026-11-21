# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSigmoidConverter(DispatchTestCase):
    def test_sigmoid(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sigmoid.default(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs)

    def test_sigmoid_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sigmoid.default(x)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_sigmoid_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sigmoid.default(x)

        input_specs = [
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_sigmoid_fp16(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sigmoid.default(x)

        inputs = [torch.randn(1, 10, dtype=torch.float16)]
        self.run_test(
            TestModule(),
            inputs,
            check_dtype=False,
        )

    @parameterized.expand([(torch.int8,), (torch.int32,), (torch.int64,)])
    def test_sigmoid_int(self, dtype):
        # The integer value is made inside the engine: TensorRT treats an int8 engine
        # input as quantized.
        class TestModule(nn.Module):
            def forward(self, x):
                x = torch.ops.aten._to_copy.default(x, dtype=dtype)
                return torch.ops.aten.sigmoid.default(x)

        inputs = [torch.randint(-5, 5, (2, 3)).float()]
        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
