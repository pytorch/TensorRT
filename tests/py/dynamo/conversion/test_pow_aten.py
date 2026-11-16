# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.lowering import get_decompositions

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
            ("half_root", trt.float16, 0.5, trt.LayerType.UNARY),
            ("bfloat16_root", trt.bfloat16, 0.5, trt.LayerType.UNARY),
            ("other_fraction", trt.float32, 0.25, trt.LayerType.ELEMENTWISE),
            ("integer_root", trt.int32, 0.5, trt.LayerType.ELEMENTWISE),
        ]
    )
    def test_square_root_is_a_sqrt(self, _, dtype, exponent, layer_type):
        # A vector norm ends in pow(sum, 0.5), which PyTorch computes with sqrt.
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
        types = [network.get_layer(i).type for i in range(network.num_layers)]
        self.assertIn(layer_type, types)
        if layer_type == trt.LayerType.UNARY:
            self.assertNotIn(trt.LayerType.ELEMENTWISE, types)
            layer = network.get_layer(types.index(trt.LayerType.UNARY))
            layer.__class__ = trt.IUnaryLayer
            self.assertEqual(layer.op, trt.UnaryOperation.SQRT)

    @parameterized.expand(
        [
            ("float32", torch.float32, nn.functional.normalize, trt.LayerType.UNARY),
            (
                "float16",
                torch.float16,
                nn.functional.normalize,
                trt.LayerType.ELEMENTWISE,
            ),
            (
                "bfloat16",
                torch.bfloat16,
                nn.functional.normalize,
                trt.LayerType.ELEMENTWISE,
            ),
            (
                "float16_root_of_cubes",
                torch.float16,
                lambda x: (x.float() ** 3).sum(1, keepdim=True) ** 0.5,
                trt.LayerType.UNARY,
            ),
        ]
    )
    def test_root_of_a_norm(self, _, dtype, function, layer_type):
        # PyTorch sums the norm of a float16 or bfloat16 tensor in float32, and TensorRT
        # 11.4 can plan more memory for that root as a SQRT, so it stays a POW. A sum of
        # cubes can be -inf, where only a SQRT gives NaN as PyTorch does.
        class Root(nn.Module):
            def forward(self, x):
                return function(x)

        program = torch.export.export(
            Root(), (torch.randn(2, 8, 4, dtype=dtype),)
        ).run_decompositions(get_decompositions())
        root = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.pow.Tensor_Scalar and node.args[1] == 0.5
        )
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network)
        ctx.current_node = root
        impl.elementwise.pow(
            ctx,
            torch.ops.aten.pow.Tensor_Scalar,
            SourceIR.ATEN,
            "pow",
            network.add_input("x", trt.float32, (2, 1, 4)),
            0.5,
        )
        types = [network.get_layer(i).type for i in range(network.num_layers)]
        self.assertIn(layer_type, types)
        other = {trt.LayerType.UNARY, trt.LayerType.ELEMENTWISE} - {layer_type}
        self.assertNotIn(other.pop(), types)

    def test_square_root_special_values(self):
        # PyTorch's CUDA pow computes an exponent of 0.5 with sqrt, so -inf gives NaN; POW gave +inf.
        class Root(nn.Module):
            def forward(self, x):
                return torch.ops.aten.pow.Tensor_Scalar(x, 0.5)

        inf = float("inf")
        inputs = [torch.tensor([[-inf, -4.0, -0.0, 0.0, 0.25, 4.0, inf]])]
        self.run_test(Root(), inputs)
        # The comparison above treats -0.0 and 0.0 as equal, so check the sign of zero.
        x = inputs[0].cuda()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Root(), (x,)), arg_inputs=(x,), min_block_size=1
        )
        self.assertTrue(torch.signbit(compiled(x)[0, 2]))

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
