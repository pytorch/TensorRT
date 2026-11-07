# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext

from .harness import DispatchTestCase


class TestPowConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_pow_tensor_tensor(self, _, shape):
        class pow(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.pow.Tensor_Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_pow_scalar(self, _, shape, scalar):
        class pow(nn.Module):
            def forward(self, rhs_val):
                return torch.ops.aten.pow.Scalar(scalar, rhs_val)

        inputs = [torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_pow_tensor_scalar(self, _, shape, scalar):
        class pow(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.pow.Tensor_Scalar(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )

    def test_pow_of_integers(self):
        # A float exponent makes the power float; an integer one keeps it integer.
        class IntPow(nn.Module):
            def forward(self, lhs_val):
                return lhs_val**0.5, lhs_val**2

        inputs = [torch.randint(0, 256, (2, 3), dtype=torch.int32)]
        self.run_test(IntPow(), inputs)

    @parameterized.expand(
        [
            ("float_square", trt.float32, 2.0, trt.ElementWiseOperation.PROD),
            ("half_square", trt.float16, 2, trt.ElementWiseOperation.PROD),
            ("cube", trt.float32, 3.0, trt.ElementWiseOperation.POW),
            ("integer_square", trt.int32, 2, trt.ElementWiseOperation.POW),
        ]
    )
    def test_square_is_a_product(self, _, dtype, exponent, operation):
        # TensorRT fuses x * x, but not x ** 2, into a reduction that reads it.
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        impl.elementwise.pow(
            ConversionContext(network),
            torch.ops.aten.pow.Tensor_Scalar,
            SourceIR.ATEN,
            "pow",
            network.add_input("x", dtype, (2, 3)),
            exponent,
        )
        operations = []
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.ELEMENTWISE:
                layer.__class__ = trt.IElementWiseLayer
                operations.append(layer.op)
        self.assertEqual(operations, [operation])

    @parameterized.expand(
        [
            (
                "2d_dim_dtype_half",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.half,
                torch.half,
            ),
            (
                "3d_dim_dtype_float",
                (1, 1, 1),
                (1, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.float,
            ),
        ]
    )
    def test_pow_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class pow(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.pow.Tensor_Tensor(lhs_val, rhs_val)

        class pow_scalar(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.pow.Tensor_Scalar(lhs_val, 2.0)

        class pow_operator(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return lhs_val**rhs_val

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            pow(), input_specs, output_dtypes=[output_type]
        )
        self.run_test_with_dynamic_shape(
            pow_scalar(), input_specs, output_dtypes=[output_type]
        )
        self.run_test_with_dynamic_shape(
            pow_operator(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
