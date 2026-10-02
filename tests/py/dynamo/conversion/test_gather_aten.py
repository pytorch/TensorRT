# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.aten_ops_converters import gather_validator

from .harness import DispatchTestCase


class TestGatherValueConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "gather_zero_dim_indexOne_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0]]),
            ),
            (
                "gather_zero_dim_indexTwo_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
            (
                "gather_one_dim_indexOne_constant_value",
                1,
                torch.tensor([[0, 1, 2, 0]]),
            ),
            (
                "gather_one_dim_indexTwo_costant_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
        ]
    )
    def test_gather_index_constant(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ("gather_zero_dim_indexOne_value", 0, torch.tensor([[0, 1, 2, 0]])),
            (
                "gather_zero_dim_indexTwo_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
            ("gather_one_dim_indexOne_value", 1, torch.tensor([[0, 1, 2, 0]])),
            (
                "gather_one_dim_indexTwo_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
        ]
    )
    def test_gather_index_input(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ("positive_dim", 1),
            ("negative_dim", -1),
        ]
    )
    def test_gather_dynamic_shape(self, _, dim):
        """The registry only consults supports_dynamic_shapes when a node carries symbolic
        shape metadata, which the legacy tracer does not produce, so this needs the dynamo
        tracer or it passes either way."""

        class TestModule(torch.nn.Module):
            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        # Generated integer examples are all zero, so pass distinct values and indices to
        # check that every row really reads the column it asks for.
        input_specs = [
            Input(
                min_shape=(1, 5),
                opt_shape=(3, 5),
                max_shape=(6, 5),
                dtype=torch.float32,
                torch_tensor=torch.arange(30, dtype=torch.float32).reshape(6, 5),
            ),
            Input(
                min_shape=(1, 4),
                opt_shape=(3, 4),
                max_shape=(6, 4),
                dtype=torch.int64,
                torch_tensor=torch.randint(0, 5, (6, 4)),
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )

    def test_gather_dynamic_gathered_axis(self):
        """The gathered axis itself is dynamic here, so index validity depends on the shape
        the engine is given at run time rather than on the shape it was built at."""

        class TestModule(torch.nn.Module):
            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, 1, index)

        input_specs = [
            Input(
                min_shape=(3, 1),
                opt_shape=(3, 4),
                max_shape=(3, 6),
                dtype=torch.float32,
                torch_tensor=torch.arange(18, dtype=torch.float32).reshape(3, 6),
            ),
            Input(
                min_shape=(3, 1),
                opt_shape=(3, 4),
                max_shape=(3, 6),
                dtype=torch.int64,
                torch_tensor=torch.tensor(
                    [[5, 0, 3, 1, 4, 2], [2, 2, 0, 5, 1, 3], [4, 1, 5, 0, 3, 2]]
                ),
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )


class TestGatherValidator(TestCase):
    @staticmethod
    def make_gather_node(data, index):
        graph = torch.fx.Graph()
        data_node = graph.placeholder("data")
        index_node = graph.placeholder("index")
        gather_node = graph.call_function(
            torch.ops.aten.gather.default, args=(data_node, 0, index_node)
        )
        graph.output(gather_node)
        data_node.meta["val"] = data
        index_node.meta["val"] = index
        return gather_node

    @staticmethod
    def export_gather_node(module):
        ep = torch.export.export(
            module, (torch.tensor([0, 1, 1]), torch.randn(4)), strict=False
        )
        return next(
            n for n in ep.graph.nodes if n.target is torch.ops.aten.gather.default
        )

    def test_data_dependent_index_known_non_empty_converts(self):
        """torch._check proves this length is positive, so the index still converts."""

        class TestModule(torch.nn.Module):
            def forward(self, x, data):
                index = torch.nonzero(x).flatten()
                torch._check(index.shape[0] > 0)
                return torch.ops.aten.gather.default(data, 0, index)

        node = self.export_gather_node(TestModule())
        self.assertIsInstance(node.args[2].meta["val"].shape[0], torch.SymInt)
        self.assertTrue(gather_validator(node))

    def test_data_dependent_index_that_may_be_empty_falls_back(self):
        """nonzero can select nothing at run time, and an empty output keeps the engine from
        running. A plain comparison on this length raises, and the registry only catches
        KeyError."""

        class TestModule(torch.nn.Module):
            def forward(self, x, data):
                index = torch.nonzero(x).flatten()
                return torch.ops.aten.gather.default(data, 0, index)

        node = self.export_gather_node(TestModule())
        self.assertIsInstance(node.args[2].meta["val"].shape[0], torch.SymInt)
        self.assertFalse(gather_validator(node))

    @parameterized.expand(
        [
            ("empty_index", torch.empty(4), torch.empty(0, dtype=torch.int64)),
            (
                "uint8_data",
                torch.empty(4, dtype=torch.uint8),
                torch.empty(2, dtype=torch.int64),
            ),
            (
                "float64_data",
                torch.empty(4, dtype=torch.float64),
                torch.empty(2, dtype=torch.int64),
            ),
            (
                "scalar_index",
                torch.empty(4),
                torch.empty((), dtype=torch.int64),
            ),
        ]
    )
    def test_refuses_with_val_metadata_only(self, _, data, index):
        """Nodes created by later passes can carry meta["val"] without tensor_meta."""
        self.assertFalse(gather_validator(self.make_gather_node(data, index)))


class TestGatherFallsBack(TestCase):
    """The validator only helps if the registry consults it, so compile each refused case
    and check that it still matches eager."""

    @parameterized.expand(
        [
            (
                "uint8_data",
                torch.randint(0, 200, (3, 5), dtype=torch.uint8),
                torch.tensor([[0, 4, 2, 1], [3, 3, 0, 1], [4, 0, 1, 2]]),
            ),
            (
                "float64_data",
                torch.randn(3, 5, dtype=torch.float64),
                torch.tensor([[0, 4, 2, 1], [3, 3, 0, 1], [4, 0, 1, 2]]),
            ),
            ("scalar_index", torch.randn(5), torch.tensor(2)),
        ]
    )
    def test_refused_gather_matches_eager(self, _, data, index):
        class TestModule(torch.nn.Module):
            def forward(self, data, index):
                return torch.ops.aten.gather.default(data, data.dim() - 1, index)

        data, index = data.cuda(), index.cuda()
        module = TestModule().eval().cuda()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (data, index)),
            arg_inputs=[data, index],
            min_block_size=1,
        )
        torch.testing.assert_close(compiled(data, index), module(data, index))


if __name__ == "__main__":
    run_tests()
