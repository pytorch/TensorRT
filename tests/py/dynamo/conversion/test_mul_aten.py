# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_mul_tensor(self, _, shape):
        class mul(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.mul.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            mul(),
            inputs,
        )

    def test_mul_bool_tensor(self):
        """Bool multiplication must compile and preserve its Boolean output dtype."""

        class mul(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.mul.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 2, (2, 3), dtype=torch.bool),
            torch.randint(0, 2, (2, 3), dtype=torch.bool),
        ]
        self.run_test(
            mul(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_int", (2, 1), 1),
            ("3d_int", (2, 1, 2), 2),
            ("2d_float", (2, 1), 1.0),
            ("3d_float", (2, 1, 2), 2.0),
        ]
    )
    def test_mul_scalar(self, _, shape, scalar):
        class mul(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.mul.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            mul(),
            inputs,
        )

    def test_mul_0d_tensor_by_scalar_stays_0d(self):
        """A 0-dim tensor times a Python scalar must stay 0-dim, so the unsqueezed
        values are 1-D and can be concatenated with other vectors."""

        class mul(nn.Module):
            def forward(self, x):
                a = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 0), 2.0)
                b = torch.ops.aten.mul.Tensor(torch.ops.aten.select.int(x, 0, 1), 3)
                return torch.ops.aten.cat.default(
                    [
                        x,
                        torch.ops.aten.unsqueeze.default(a, 0),
                        torch.ops.aten.unsqueeze.default(b, 0),
                    ]
                )

        inputs = [torch.randn(4)]
        self.run_test(
            mul(),
            inputs,
        )

    def test_mul_by_stacked_0d_tensors_times_scalars(self):
        """Scaling boxes by [w, h, w, h], each a 0-dim tensor times a Python scalar,
        must use each value in its own place."""

        class mul(nn.Module):
            def forward(self, boxes, sizes):
                h, w = sizes[0, 0], sizes[0, 1]
                return boxes * torch.stack([w * 1.0, h * 1.0, w * 1.0, h * 1.0])

        inputs = [torch.rand(1, 5, 4), torch.tensor([[48.0, 64.0]])]
        self.run_test(
            mul(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("one_element", (1,)),
            ("three_elements", (3,)),
        ]
    )
    def test_mul_0d_tensor_by_constant_tensor(self, _, shape):
        """A 0-dim tensor times a constant tensor takes the constant's shape. This guards
        that only a Python scalar next to a 0-dim tensor becomes 0-dim."""

        class mul(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.randn(shape))

            def forward(self, x):
                return torch.ops.aten.mul.Tensor(
                    torch.ops.aten.select.int(x, 0, 0), self.scale
                )

        inputs = [torch.randn(4)]
        self.run_test(
            mul(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
