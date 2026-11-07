# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
import tensorrt as trt
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext

from .harness import DispatchTestCase


class TestConstantPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2), (1, 1), 0),
            ((2, 1), (2, 1), 1),
            ((2, 2), (1, 1), 0),
            ((1, 2), (1, 1, 2, 0), 0),
            ((3, 4, 2), (1, 2), 2),
            ((3, 4, 2), (1, 2, 3, 1, 2, 3), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 2),
            ((3, 3, 4, 2, 1), (1, 2, 3, 4, 5, 1, 2, 3, 4, 5), 0),
            ((3, 3, 4, 2, 1, 2), (1, 2, 3, 4, 1, 2, 3, 4), 4),
        ]
    )
    def test_constant_pad(self, shape, pad, value):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.constant_pad_nd.default(input, pad, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "3d",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1, 1, 1, 1, 1),
                0,
            ),
        ]
    )
    def test_dynamic_shape_constant_pad(
        self, _, min_shape, opt_shape, max_shape, type, pad, value
    ):
        class constant_pad(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.constant_pad_nd.default(input, pad, value)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(constant_pad(), input_specs)

    def test_zero_constant_pad_read_by_a_mul(self):
        class ZeroPadThenMul(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.ops.aten.constant_pad_nd.default(x, [0, 0, 0, 0]) * 2.0 - 1.0
                )

        self.run_test(ZeroPadThenMul(), [torch.randn(1, 3, 8, 8)])

    @parameterized.expand(
        [
            ("scatter", True),
            ("cat_with_another_reader_of_the_input", True),
            ("cat_of_pads_and_mul", True),
            ("cat_of_pads_of_another_tensor", True),
            ("cat_of_pads", False),
        ]
    )
    def test_zero_constant_pad_keeps_its_slice_only_in_a_cat_of_pads(
        self, reader, left_out
    ):
        aten = torch.ops.aten
        graph = torch.fx.Graph()
        x, y = graph.placeholder("x"), graph.placeholder("y")
        pad = graph.call_function(aten.constant_pad_nd.default, (x, [0, 0, 0, 0]))
        if reader == "scatter":
            graph.call_function(aten.relu.default, (x,))
            graph.call_function(aten.scatter.src, (y, 3, y, pad))
        elif reader == "cat_with_another_reader_of_the_input":
            relu = graph.call_function(aten.relu.default, (x,))
            graph.call_function(aten.cat.default, ([pad, relu], 1))
        else:
            source = y if reader == "cat_of_pads_of_another_tensor" else x
            shifted = graph.call_function(
                aten.constant_pad_nd.default, (source, [1, -1, 0, 0])
            )
            graph.call_function(aten.cat.default, ([pad, shifted], 1))
            if reader == "cat_of_pads_and_mul":
                graph.call_function(aten.mul.Tensor, (pad, 2.0))
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        input = network.add_input("input", trt.float32, (1, 3, 8, 8))
        output = impl.pad.constant_padNd(
            ConversionContext(network, current_node=pad),
            torch.ops.aten.constant_pad_nd.default,
            SourceIR.ATEN,
            "pad",
            input,
            (0, 0, 0, 0),
        )
        self.assertEqual(output is input, left_out)
        self.assertEqual(network.num_layers == 0, left_out)


class TestReflectionPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_reflection_pad1d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad1d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "3d",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad1d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad1d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad1d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad1d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_reflection_pad2d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad2d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "4d",
                (2, 2, 2, 2),
                (4, 4, 4, 4),
                (6, 6, 6, 6),
                torch.float,
                (1, 1, 2, 2),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad2d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad2d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad2d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad2d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_reflection_pad3d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad3d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "5d",
                (2, 2, 2, 2, 2),
                (4, 4, 4, 4, 4),
                (6, 6, 6, 6, 6),
                torch.float,
                (1, 2, 2, 1, 1, 2),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad3d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad3d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad3d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad3d(), input_specs)


class TestReplicationPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_replication_pad1d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad1d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "3d",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
            ),
        ]
    )
    def test_dynamic_shape_replication_pad1d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class replication_pad1d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.replication_pad1d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(replication_pad1d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_replication_pad2d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad2d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "4d",
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                (1, 1, 2, 2),
            ),
        ]
    )
    def test_dynamic_shape_replication_pad2d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class replication_pad2d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.replication_pad2d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(replication_pad2d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_replication_pad3d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad3d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "5d",
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 2),
                (3, 3, 3, 3, 3),
                torch.float,
                (1, 1, 2, 2, 1, 2),
            ),
        ]
    )
    def test_dynamic_shape_replication_pad3d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class replication_pad3d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.replication_pad3d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(replication_pad3d(), input_specs)


class TestCircularPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_circular_pad1d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_circular_pad2d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_circular_pad3d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "circular_pad_1d",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
            ),
            (
                "circular_pad_2d",
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                (1, 1, 2, 2),
            ),
            (
                "circular_pad_3d",
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 2),
                (3, 3, 3, 3, 3),
                torch.float,
                (1, 1, 2, 2, 1, 2),
            ),
        ]
    )
    def test_dynamic_shape_circular_pad(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class circular_pad(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(circular_pad(), input_specs)


class TestPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 3), (2, 2), "constant"),
            ((2, 2, 4), (2, 3, 1, 0), "reflect"),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1), "replicate"),
            ((2, 3, 4, 5), (3, 2, 1, 0), "circular"),
        ]
    )
    def test_pad(self, shape, pad, mode, value=None):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.pad.default(input, pad, mode, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "constant_pad_nd",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
                "constant",
            ),
            (
                "reflection_padnd",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
                "reflect",
            ),
            (
                "replication_padnd",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
                "replicate",
            ),
            (
                "_pad_circular",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
                "circular",
            ),
        ]
    )
    def test_dynamic_shape_pad(
        self, _, min_shape, opt_shape, max_shape, type, padding, mode, value=None
    ):
        class pad(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.pad.default(input, padding, mode, value)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(pad(), input_specs)


if __name__ == "__main__":
    run_tests()
