# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestDivConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_div_tensor(self, _, shape):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.div.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), None),
            ("3d", (2, 1, 2), "trunc"),
            ("3d", (2, 3, 2), "floor"),
        ]
    )
    def test_div_tensor_rounding_mode(self, _, shape, rounding_mode):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.div.Tensor_mode(
                    lhs_val, rhs_val, rounding_mode=rounding_mode
                )

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), -1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_div_tensor(self, _, shape, scalar):
        class div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.div.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, None),
            ("3d", (2, 1, 2), 2.0, "trunc"),
            ("3d", (2, 3, 2), -3.0, "floor"),
        ]
    )
    def test_div_tensor_rounding_mode(self, _, shape, scalar, rounding_mode):
        class div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.div.Tensor_mode(
                    lhs_val, scalar, rounding_mode=rounding_mode
                )

        inputs = [torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_prims_div_tensor(self, _, shape):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.prims.div.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand([("int32", torch.int32), ("int64", torch.int64)])
    def test_prims_div_truncates_integers(self, _, dtype):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                # Multiplying back keeps the truncation visible; a cast of the output would hide it.
                return torch.ops.prims.div.default(lhs_val, rhs_val) * rhs_val

        inputs = [
            torch.tensor([-7, 7, -9, 9, 5, -5, 13, -13], dtype=dtype),
            torch.tensor([2, 2, 4, 4, 3, 3, 5, 5], dtype=dtype),
        ]
        self.run_test(div(), inputs)

    def test_div_odd_dynamic_size_gives_float(self):
        class div(nn.Module):
            def forward(self, x):
                return x * (x.shape[0] / 2)

        # The harness runs the max shape, so 15 / 2 has to give 7.5 and not 7.
        input_specs = [
            Input(min_shape=(2, 3), opt_shape=(8, 3), max_shape=(15, 3)),
        ]
        self.run_test_with_dynamic_shape(div(), input_specs, use_dynamo_tracer=True)

    def test_div_dynamic_size_keeps_half(self):
        class div(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 4).half()

            def forward(self, x):
                return self.lin(x * (x.shape[0] / 2))

        # A float32 scale would widen x, and the half Linear would then fail to build.
        input_specs = [
            Input(
                min_shape=(2, 8), opt_shape=(8, 8), max_shape=(15, 8), dtype=torch.half
            ),
        ]
        self.run_test_with_dynamic_shape(
            div(), input_specs, rtol=1e-2, atol=1e-2, use_dynamo_tracer=True
        )

    def test_div_dynamic_size_above_half_range(self):
        class div(nn.Module):
            def forward(self, x):
                return (x * 1e-4) * (x.shape[0] / 3)

        # At the max size the quotient is 100000, which float16 cannot hold.
        input_specs = [
            Input(
                min_shape=(2, 6),
                opt_shape=(1000, 6),
                max_shape=(300000, 6),
                dtype=torch.half,
            ),
        ]
        self.run_test_with_dynamic_shape(
            div(), input_specs, rtol=1e-2, atol=1e-2, use_dynamo_tracer=True
        )

    def test_div_dynamic_size_above_half_range_in_cond(self):
        class div(nn.Module):
            def forward(self, x):
                return torch.cond(
                    x.abs().amax() >= 0,
                    lambda v: (v * 1e-4) * (v.shape[0] / 3),
                    lambda v: v * 2,
                    (x,),
                )

        # A branch is converted by its own interpreter, which has to track the quotient too.
        # The run stays at the export size, which sets the engine's opt shape: TensorRT 11.4's
        # conditional can leave the rows past the opt shape unwritten.
        x = torch.rand(300000, 6, dtype=torch.half, device="cuda")
        n = torch.export.Dim("n", min=2, max=300000)
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(div(), (x,), dynamic_shapes=({0: n},)),
            arg_inputs=[
                Input(
                    min_shape=(2, 6),
                    opt_shape=(300000, 6),
                    max_shape=(300000, 6),
                    dtype=torch.half,
                )
            ],
            min_block_size=1,
            pass_through_build_failures=True,
        )
        torch.testing.assert_close(compiled(x), div()(x), rtol=1e-2, atol=1e-2)

    @parameterized.expand(
        [
            ("sum", lambda n, m: n / 3 + m / 3),
            ("product", lambda n, m: (n / 4) * (m / 6)),
            ("negation", lambda n, m: -(n / 3)),
        ]
    )
    def test_div_dynamic_size_derived_value_above_half_range(self, _, scale):
        class div(nn.Module):
            def forward(self, x, y):
                return (x * 1e-4) * scale(x.shape[0], y.shape[0])

        # At the max sizes each scale is above 65504, which float16 cannot hold.
        input_specs = [
            Input(
                min_shape=(2, 6),
                opt_shape=(1000, 6),
                max_shape=(300000, 6),
                dtype=torch.half,
            ),
            Input(min_shape=(2,), opt_shape=(3,), max_shape=(6,), dtype=torch.half),
        ]
        self.run_test_with_dynamic_shape(
            div(),
            input_specs,
            rtol=1e-2,
            atol=1e-2,
            use_dynamo_tracer=True,
            torch_export_dynamic_shapes={
                "x": {0: torch.export.Dim("n", min=2, max=300000)},
                "y": {0: torch.export.Dim("m", min=2, max=6)},
            },
        )

    def test_compare_half_with_dynamic_size_quotient(self):
        class div(nn.Module):
            def forward(self, x):
                return (x * 0 + 3000) == (x.shape[0] / 3)

        # At the max size the quotient is 2999.67, which float16 rounds to 3000.
        input_specs = [
            Input(
                min_shape=(2, 6),
                opt_shape=(1000, 6),
                max_shape=(8999, 6),
                dtype=torch.half,
            ),
        ]
        self.run_test_with_dynamic_shape(div(), input_specs, use_dynamo_tracer=True)

    def test_truediv_of_tensor_keeps_rank(self):
        class div(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cat.default([x / 2, x])

        # symbolic_trace records x / 2 as operator.truediv on a tensor, not on sizes.
        self.run_test(div(), [torch.randn(1)], use_dynamo_tracer=False)

    def test_prims_div_truncates_signed_integers(self):
        # Doubling keeps a float quotient from being truncated back at the output.
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.prims.div.default(lhs_val, rhs_val) * 2

        # Large values catch a sign taken from a product that overflows.
        lhs_val = torch.randint(-50, 50, (2, 8), dtype=torch.int32)
        lhs_val[0, :2] = torch.tensor([3000000, -3000000])
        rhs_val = torch.tensor([1, -2, 3, -4, 5, -6, 7, -8], dtype=torch.int32)
        rhs_val = rhs_val.repeat(2, 1)
        rhs_val[0, :2] = 1000
        self.run_test(div(), [lhs_val, rhs_val])

    def test_div_of_integers_is_float(self):
        # Reciprocal lowers to 1 / x, so it is a true division too. 2**32 has no int32 constant.
        class IntDiv(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return (
                    lhs_val / rhs_val,
                    lhs_val / 255,
                    lhs_val / 2**32,
                    torch.reciprocal(rhs_val),
                )

        inputs = [
            torch.randint(0, 256, (2, 3), dtype=torch.int32),
            torch.randint(1, 256, (2, 3), dtype=torch.int32),
        ]
        self.run_test(IntDiv(), inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_div_and_pow_of_integers_by_a_half_constant_stay_half(self):
        # The Linear needs both inputs in one type, so a float32 result does not build.
        class HalfScale(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(4.0, dtype=torch.half))
                self.register_buffer("root", torch.tensor(0.5, dtype=torch.half))
                self.lin = nn.Linear(8, 4).half()

            def forward(self, x):
                return self.lin(x / self.scale + x**self.root)

        inputs = [torch.randint(0, 256, (2, 8), dtype=torch.int32)]
        self.run_test(HalfScale(), inputs, use_dynamo_tracer=True, enable_passes=True)


if __name__ == "__main__":
    run_tests()
