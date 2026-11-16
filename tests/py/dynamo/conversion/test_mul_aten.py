# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
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

    @parameterized.expand(
        [
            ("one_element", (1,)),
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

    @parameterized.expand(
        [
            ("int_size", lambda a: a[0, 0] * a.shape[0]),
        ]
    )
    def test_mul_0d_tensor_by_dynamic_size_stays_0d(self, _, scaled):
        """A 0-dim tensor times a size read from a dynamic shape is 0-dim, as with a
        Python number, so the two can be stacked."""

        class mul(nn.Module):
            def forward(self, a):
                return torch.stack([scaled(a), a[0, 1] * 2.0])

        a = torch.randn(4, 3, device="cuda")
        batch = torch.export.Dim("batch", min=2, max=16)
        program = torch.export.export(mul(), (a,), dynamic_shapes=({0: batch},))
        compiled = torch_tensorrt.dynamo.compile(
            program,
            arg_inputs=[
                torch_tensorrt.Input(
                    min_shape=(2, 3), opt_shape=(4, 3), max_shape=(16, 3)
                )
            ],
            min_block_size=1,
            pass_through_build_failures=True,
        )
        for n in (2, 4, 7):
            x = torch.randn(n, 3, device="cuda")
            torch.testing.assert_close(compiled(x), mul()(x))


if __name__ == "__main__":
    run_tests()
