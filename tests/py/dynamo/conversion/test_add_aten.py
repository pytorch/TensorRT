# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAddConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_add_tensor(self, _, shape):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_add_tensor_alpha(self, _, shape, alpha):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val, alpha=alpha)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_add_scalar(self, _, shape, scalar):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, 1.0),
            ("3d", (2, 1, 2), 2, 2),
        ]
    )
    def test_add_scalar_alpha(self, _, shape, scalar, alpha):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar, alpha=alpha)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    def test_add_int8_buffer(self):
        # TensorRT reads an INT8 constant as quantized data, so a plain int8 weight
        # must not become one.
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
                )

            def forward(self, x):
                return torch.ops.aten.add.Tensor(self.offsets, x)

        inputs = [torch.randn(16, 16)]
        self.run_test(
            add(),
            inputs,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    run_tests()
