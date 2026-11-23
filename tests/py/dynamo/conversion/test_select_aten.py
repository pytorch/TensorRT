# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys
import unittest

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.aten_ops_converters import select_validator

from .harness import DispatchTestCase


class TestSelectConverterOne(DispatchTestCase):
    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_2d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(1, 2)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_4d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(4, 4, 4, 4)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            (
                "partial_dynamic_static_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                2,
                0,
            ),
            (
                "partial_dynamic_dynamic_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic_neg_dim",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                -1,
                1,
            ),
        ]
    )
    def test_dynamic_shape_select(
        self, _, min_shape, opt_shape, max_shape, type, dim, index
    ):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(select(), input_specs)

    @parameterized.expand(
        [
            ("rank0_index", [1, 2], [0], False),
            ("rank1_size1_index", [1, 2], [0], True),
            ("rank2_size1_index", [[1, 2]], [0, 1], True),
        ]
    )
    def test_select_runtime_index(self, _, positions, dims, keepdim):
        """Select row 3 of an (8, 4) input with an index computed in the engine.

        Eager gives shape (4,) for any index with one element. A rank 0 index is only
        cast. A (1,) index, the shape an integer scalar has after crossing a partition
        boundary, and a (1, 1) index from a keepdim reduction are reshaped to rank 0.
        """

        class SelectRuntimeIndex(nn.Module):
            def forward(self, values, positions):
                index = torch.ops.aten.sum.dim_IntList(positions, dims, keepdim)
                return torch.ops.aten.select.int(values, 0, index)

        inputs = [
            torch.randn(8, 4),
            torch.tensor(positions, dtype=torch.int64),
        ]
        # torch.export rejects a tensor index for select, so trace with symbolic_trace.
        self.run_test(SelectRuntimeIndex(), inputs, use_dynamo_tracer=False)

    @parameterized.expand(
        [
            ("two_elements", Input(shape=(2,), dtype=torch.int64)),
            (
                "dynamic_size",
                Input(
                    min_shape=(1,), opt_shape=(1,), max_shape=(2,), dtype=torch.int64
                ),
            ),
        ]
    )
    def test_select_runtime_index_not_single_element(self, _, index_spec):
        class SelectRuntimeIndex(nn.Module):
            def forward(self, values, index):
                return torch.ops.aten.select.int(values, 0, index)

        input_specs = [Input(shape=(8, 4), dtype=torch.float32), index_spec]
        # Eager rejects a two-element index as well, so skip the dtype check that runs it.
        with self.assertRaisesRegex(RuntimeError, "exactly one element"):
            self.run_test_with_dynamic_shape(
                SelectRuntimeIndex(),
                input_specs,
                use_dynamo_tracer=False,
                check_dtype=False,
            )

    def test_select_uint8_input_compiles(self):
        # The uint8 select would fail the engine build, so it must stay in PyTorch.
        class SelectThenCast(nn.Module):
            def forward(self, frames):
                return frames[1].float() / 255

        frames = torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8, device="cuda")
        model = SelectThenCast()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (frames,)),
            arg_inputs=[frames],
            min_block_size=1,
            pass_through_build_failures=True,
        )
        torch.testing.assert_close(compiled(frames), model(frames))

    def test_select_validator_reads_tensor_meta(self):
        class Select(nn.Module):
            def forward(self, frames):
                return torch.ops.aten.select.int(frames, 0, 1)

        graph_module = torch.fx.symbolic_trace(Select())
        ShapeProp(graph_module).propagate(torch.zeros(2, 4, dtype=torch.uint8))
        node = next(n for n in graph_module.graph.nodes if n.op == "call_function")
        self.assertNotIn("val", node.args[0].meta)
        self.assertFalse(select_validator(node))


def _index_form_graph(case):
    """The select of plane 2 of x in a graph that uses it as the case says."""
    aten = torch.ops.aten
    graph = torch.fx.Graph()
    x = graph.call_function(aten.exp.default, (graph.placeholder("x"),))
    data = x
    if case.startswith("of_a_permute"):
        data = graph.call_function(aten.permute.default, (x, [0, 1, 2]))
    elif case.startswith("of_a_split_piece"):
        pieces = graph.call_function(aten.split_with_sizes.default, (x, [2, 3], 1))
        data = graph.call_function(operator.getitem, (pieces, 1))
    elif case.startswith("of_a_full_slice"):
        # A dynamic dim makes export keep x[:, ...] as a slice of the whole dim.
        data = graph.call_function(aten.slice.Tensor, (x, 0, 0, sys.maxsize))
    elif case.startswith("of_a_view_by_a_size"):
        # As export writes x.view(x.shape[0], ...) for a dynamic batch dim.
        size = graph.call_function(aten.sym_size.int, (x, 0))
        size.meta["val"] = 2
        data = graph.call_function(aten.view.default, (x, [size, 5, 4]))
    elif case.startswith("of_an_index"):
        data = graph.call_function(aten.index.Tensor, (x, [graph.placeholder("i")]))
    elif case.startswith("of_a_"):
        raise ValueError(case)
    select = graph.call_function(aten.select.int, (data, 1, 2))
    plane = graph.call_function(aten.unsqueeze.default, (select, 1))
    if case.endswith("next_to_a_slice"):
        graph.call_function(aten.slice.Tensor, (x, 1, 0, 2))
    elif case.endswith("concatenated_with_its_slices"):
        head = graph.call_function(aten.slice.Tensor, (x, 1, 0, 2))
        tail = graph.call_function(aten.slice.Tensor, (x, 1, 3, 5))
        graph.call_function(aten.cat.default, ([head, plane, tail], 1))
    elif case.endswith("concatenated_with_its_planes"):
        planes = [
            graph.call_function(
                aten.unsqueeze.default,
                (graph.call_function(aten.select.int, (x, 1, i)), 1),
            )
            for i in (0, 1, 3, 4)
        ]
        graph.call_function(aten.cat.default, ([*planes[:2], plane, *planes[2:]], 1))
    elif case.endswith("concatenated_with_another_tensor"):
        other = graph.call_function(
            aten.slice.Tensor, (graph.placeholder("y"), 1, 0, 2)
        )
        graph.call_function(aten.cat.default, ([other, plane], 1))
    elif case.endswith("concatenated_with_the_other_piece"):
        other = graph.call_function(operator.getitem, (data.args[0], 0))
        graph.call_function(aten.cat.default, ([other, plane], 1))
    elif case.endswith("added_to_another_tensor_scattered_back"):
        total = graph.call_function(aten.add.Tensor, (select, graph.placeholder("y")))
        plane = graph.call_function(aten.unsqueeze.default, (total, 1))
        graph.call_function(aten.slice_scatter.default, (x, plane, 1, 2, 3))
    elif case.endswith("written_into_then_scattered_back"):
        # As in v[:, 2, 1:] = y: the select is the base of the inner write.
        inner = graph.call_function(
            aten.slice_scatter.default, (select, graph.placeholder("y"), 1, 1)
        )
        plane = graph.call_function(aten.unsqueeze.default, (inner, 1))
        graph.call_function(aten.slice_scatter.default, (x, plane, 1, 2, 3))
    elif case.endswith("scattered_back"):
        graph.call_function(aten.slice_scatter.default, (x, plane, 1, 2, 3))
    elif case.endswith("scattered_into_another_tensor"):
        y = graph.placeholder("y")
        graph.call_function(aten.slice_scatter.default, (y, plane, 1, 2, 3))
    elif case != "alone":
        raise ValueError(case)
    return select


class TestSelectIndexForm(TestCase):
    # A scalar index gives wrong values when the gather ends up in a concatenation that
    # rebuilds the tensor, and TensorRT fails to build some engines with a one element
    # index. So a select gets the one element index when its result goes back to its data.
    @parameterized.expand(
        [
            ("alone", False),
            ("next_to_a_slice", False),
            ("concatenated_with_its_slices", True),
            ("concatenated_with_its_planes", True),
            ("concatenated_with_another_tensor", False),
            ("scattered_back", True),
            ("scattered_into_another_tensor", False),
            ("added_to_another_tensor_scattered_back", True),
            ("written_into_then_scattered_back", True),
            ("of_a_permute_concatenated_with_its_slices", True),
            ("of_a_split_piece_concatenated_with_the_other_piece", True),
            ("of_a_split_piece_next_to_a_slice", False),
            ("of_a_full_slice_concatenated_with_its_slices", True),
            ("of_a_view_by_a_size_concatenated_with_its_slices", True),
            ("of_an_index_concatenated_with_its_slices", False),
        ]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA device")
    def test_a_select_put_back_with_its_data_gathers_with_a_one_element_index(
        self, case, one_element
    ):
        node = _index_form_graph(case)
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        impl.select.select(
            ConversionContext(network, current_node=node),
            torch.ops.aten.select.int,
            SourceIR.ATEN,
            "select",
            network.add_input("x", trt.float32, (2, 5, 4)),
            1,
            2,
        )
        gathers = [
            network.get_layer(i)
            for i in range(network.num_layers)
            if network.get_layer(i).type == trt.LayerType.GATHER
        ]
        self.assertEqual(len(gathers), 1)
        self.assertEqual(
            tuple(gathers[0].get_input(1).shape), (1,) if one_element else ()
        )


class WriteBack(nn.Module):
    """Puts a selected plane of a tensor back with the rest of the tensor."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, x):
        if self.kind == "concat_of_parts":
            return torch.cat([x[:, :4], x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_parts_times_one":
            return torch.cat([x[:, :4] * 1, x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_parts_times_a_buffer_of_one":
            return torch.cat([x[:, :4] * self.one, x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_parts_times_a_buffer_of_one_on_the_left":
            return torch.cat([self.one * x[:, :4], x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_parts_sliced_from_the_end":
            # x has 2 rows, so [-2:] keeps them all.
            return torch.cat([x[:, :4][-2:], x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_last_dim_parts":
            return torch.cat([x[..., :3], x[..., 3].unsqueeze(-1)], -1)
        if self.kind == "concat_of_a_concat":
            inner = torch.cat([x[:, :2], x[:, 2:4]], 1)
            return torch.cat([inner, x[:, 4].unsqueeze(1)], 1)
        if self.kind == "concat_of_planes":
            return torch.cat([x[:, i].unsqueeze(1) for i in range(5)], 1)
        if self.kind == "concat_of_chunks":
            return torch.cat([*x.chunk(5, 1)[:4], x[:, 4].unsqueeze(1)], 1)
        if self.kind == "select_of_a_view":
            return torch.cat([x[:, :4], x.view(2, 5, 3, 4)[:, 4].unsqueeze(1)], 1)
        if self.kind == "select_of_a_permute":
            return torch.cat([x[:, :4], x.permute(0, 1, 2, 3)[:, 4].unsqueeze(1)], 1)
        if self.kind == "select_of_a_slice":
            return torch.cat([x[:, :4], x[:, 4:][:, 0].unsqueeze(1)], 1)
        if self.kind == "select_of_a_split_piece":
            first, last = x.split([4, 1], 1)
            return torch.cat([first, last[:, 0].unsqueeze(1)], 1)
        if self.kind == "slice_of_a_copy":
            copy = torch.zeros_like(x).copy_(x)
            return torch.cat([copy[:, :4], x[:, 4].unsqueeze(1)], 1)
        v = x.clone()
        if self.kind == "unchanged":
            v[:, 2] = v[:, 2]
        elif self.kind == "times_one":
            v[:, 0] = v[:, 0] * 1
        elif self.kind == "where_of_itself":
            v[:, 2] = torch.where(x[:, 0] > 0, v[:, 2], v[:, 2] * 1)
        elif self.kind == "slice_of_plane":
            v[:, 1, :, 1:] = v[:, 1, :, 1:]
        elif self.kind == "plane_of_a_slice":
            v[1:, 2] = v[1:, 2]
        elif self.kind == "times_one_of_a_vector":
            v[2] = v[2] * 1
        return v


@unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA device")
class TestSelectWrittenBack(TestCase):
    def _assert_ran_in_tensorrt(self, trt_module):
        # A select, cat or scatter left to PyTorch would not test the TensorRT network.
        # The graph modules are the top one and the PyTorch segments, not the engines.
        aten = torch.ops.aten
        left_to_pytorch = [
            node.target
            for submodule in trt_module.modules()
            if isinstance(submodule, torch.fx.GraphModule)
            for node in submodule.graph.nodes
            if node.target
            in (aten.select.int, aten.cat.default, aten.slice_scatter.default)
        ]
        self.assertFalse(left_to_pytorch)

    def _check(self, module, inputs):
        module = module.eval().cuda()
        inputs = [i.cuda() for i in inputs]
        ep = torch.export.export(module, tuple(i.clone() for i in inputs))
        expected = module(*[i.clone() for i in inputs])
        trt_module = torch_tensorrt.dynamo.compile(
            ep, inputs=[i.clone() for i in inputs], min_block_size=1
        )
        self._assert_ran_in_tensorrt(trt_module)
        with torch.no_grad():
            torch.testing.assert_close(
                trt_module(*[i.clone() for i in inputs]), expected
            )

    @parameterized.expand(
        [
            ("unchanged",),
            ("times_one",),
            ("where_of_itself",),
            ("slice_of_plane",),
            ("plane_of_a_slice",),
            ("concat_of_parts",),
            ("concat_of_parts_times_one",),
            ("concat_of_parts_times_a_buffer_of_one",),
            ("concat_of_parts_times_a_buffer_of_one_on_the_left",),
            ("concat_of_parts_sliced_from_the_end",),
            ("concat_of_last_dim_parts",),
            ("times_one_of_a_vector", (5,)),
            ("concat_of_a_concat",),
            ("concat_of_planes",),
            ("concat_of_chunks",),
            ("select_of_a_view",),
            ("select_of_a_permute",),
            ("slice_of_a_copy",),
            ("select_of_a_split_piece",),
            ("select_of_a_slice",),
        ]
    )
    def test_select_written_back_matches_pytorch(self, kind, shape=(2, 5, 3, 4)):
        self._check(WriteBack(kind), [torch.randn(shape)])

    def test_concat_of_parts_with_a_dynamic_batch_matches_pytorch(self):
        module = WriteBack("concat_of_parts").eval().cuda()
        x = torch.randn(2, 5, 3, 4).cuda()
        # Dim.AUTO makes export keep x[:, ...] as a slice of the whole batch dim.
        ep = torch.export.export(
            module, (x,), dynamic_shapes=({0: torch.export.Dim.AUTO},)
        )
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[
                Input(
                    min_shape=(1, 5, 3, 4),
                    opt_shape=(2, 5, 3, 4),
                    max_shape=(8, 5, 3, 4),
                )
            ],
            min_block_size=1,
        )
        self._assert_ran_in_tensorrt(trt_module)
        with torch.no_grad():
            for size in (1, 3, 8):
                x = torch.randn(size, 5, 3, 4).cuda()
                torch.testing.assert_close(trt_module(x), module(x))

    def test_empty_write_into_input_plane_matches_pytorch(self):
        # On PyTorch nightlies, writing into a graph input also needs #5059.
        class WriteInput(nn.Module):
            def forward(self, x, y):
                x[:, 0, :, 4:] = y[:, 0, :, 4:]
                return x

        self._check(WriteInput(), [torch.randn(1, 3, 3, 4), torch.randn(1, 8, 3, 4)])


if __name__ == "__main__":
    run_tests()
