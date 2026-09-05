import operator
import unittest
from unittest.mock import Mock

import torch
import torch_tensorrt
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo.conversion._symbolic_shape_capture import (
    extract_symbolic_shape_expressions,
)
from torch_tensorrt.dynamo.conversion.truncate_double import repair_double_inputs


class TestTruncateDoubleMetadata(TestCase):
    def _make_graphs(
        self, *, with_scalar_output: bool
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule, torch.fx.Node]:
        tensor64 = torch.empty((2, 3), dtype=torch.float64)

        subgraph = torch.fx.Graph()
        subgraph_input = subgraph.placeholder("x")
        subgraph_input.meta["val"] = tensor64
        tensor_output = subgraph.call_function(
            torch.ops.aten.add.Tensor, args=(subgraph_input, 1.0)
        )
        tensor_output.meta["val"] = tensor64
        if with_scalar_output:
            scalar_output = subgraph.call_function(
                torch.ops.aten.sym_size.int, args=(tensor_output, 0)
            )
            scalar_output.meta["val"] = 2
            subgraph.output((tensor_output, scalar_output))
        else:
            subgraph.output(tensor_output)
        submodule = torch.fx.GraphModule({}, subgraph)

        root = nn.Module()
        root.add_module("run_on_acc_0", submodule)
        parent_graph = torch.fx.Graph()
        parent_input = parent_graph.placeholder("x")
        parent_input.meta["val"] = tensor64
        engine_node = parent_graph.call_module("run_on_acc_0", args=(parent_input,))
        if with_scalar_output:
            engine_node.meta["val"] = [tensor64, 2]
            tensor_getitem = parent_graph.call_function(
                operator.getitem, args=(engine_node, 0)
            )
            tensor_getitem.meta["val"] = tensor64
            scalar_getitem = parent_graph.call_function(
                operator.getitem, args=(engine_node, 1)
            )
            scalar_getitem.meta["val"] = 2
            parent_graph.output((tensor_getitem, scalar_getitem))
        else:
            engine_node.meta["val"] = tensor64
            parent_graph.output(engine_node)

        parent = torch.fx.GraphModule(root, parent_graph)
        return parent, submodule, engine_node

    def _repair(
        self, parent: torch.fx.GraphModule, submodule: torch.fx.GraphModule
    ) -> Input:
        # Any attempt to rediscover output dtypes by executing the partition is a
        # regression: parameters may be offloaded and outputs may include scalars.
        submodule.forward = Mock(
            side_effect=AssertionError("truncate_double executed the partition")
        )
        input_spec = Input((2, 3), dtype=torch.float64)
        repaired = repair_double_inputs(
            parent,
            submodule,
            [input_spec],
            torch.device("cpu"),
            "run_on_acc_0",
        )
        return repaired[0]

    def test_single_output_metadata_matches_engine_boundary(self):
        parent, submodule, engine_node = self._make_graphs(with_scalar_output=False)

        repaired_input = self._repair(parent, submodule)

        casts = [
            node
            for node in parent.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
        ]
        self.assertEqual(len(casts), 2)
        input_cast = next(
            node for node in casts if node.kwargs["dtype"] == torch.float32
        )
        output_cast = next(
            node for node in casts if node.kwargs["dtype"] == torch.float64
        )
        self.assertIs(engine_node.args[0], input_cast)
        self.assertEqual(input_cast.meta["val"].dtype, torch.float32)
        self.assertEqual(engine_node.meta["val"].dtype, torch.float32)
        self.assertEqual(output_cast.meta["val"].dtype, torch.float64)
        self.assertEqual(repaired_input.dtype, dtype.float32)

    def test_scalar_tuple_output_is_not_executed_or_retyped(self):
        parent, submodule, engine_node = self._make_graphs(with_scalar_output=True)

        self._repair(parent, submodule)

        engine_values = engine_node.meta["val"]
        self.assertEqual(engine_values[0].dtype, torch.float32)
        self.assertEqual(engine_values[1], 2)

        getitems = [
            node for node in parent.graph.nodes if node.target == operator.getitem
        ]
        tensor_getitem = next(node for node in getitems if node.args[1] == 0)
        scalar_getitem = next(node for node in getitems if node.args[1] == 1)
        self.assertEqual(tensor_getitem.meta["val"].dtype, torch.float32)
        self.assertEqual(scalar_getitem.meta["val"], 2)

        restoring_casts = [
            node
            for node in parent.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
            and node.kwargs["dtype"] == torch.float64
        ]
        self.assertEqual(len(restoring_casts), 1)
        self.assertIs(restoring_casts[0].args[0], tensor_getitem)
        self.assertEqual(restoring_casts[0].meta["val"].dtype, torch.float64)

    def test_repairs_float64_output_with_only_float32_inputs(self):
        # A float64 weight/constant (not a runtime input) can still produce a
        # float64 output -- e.g. float32_input + float64_weight promotes to
        # float64 in PyTorch. Output repair must not be gated on finding a
        # float64 input.
        tensor32 = torch.empty((2, 3), dtype=torch.float32)
        tensor64_out = torch.empty((2, 3), dtype=torch.float64)

        subgraph = torch.fx.Graph()
        subgraph_input = subgraph.placeholder("x")
        subgraph_input.meta["val"] = tensor32
        tensor_output = subgraph.call_function(
            torch.ops.aten.add.Tensor, args=(subgraph_input, 1.0)
        )
        tensor_output.meta["val"] = tensor64_out
        subgraph.output(tensor_output)
        submodule = torch.fx.GraphModule({}, subgraph)

        root = nn.Module()
        root.add_module("run_on_acc_0", submodule)
        parent_graph = torch.fx.Graph()
        parent_input = parent_graph.placeholder("x")
        parent_input.meta["val"] = tensor32
        engine_node = parent_graph.call_module("run_on_acc_0", args=(parent_input,))
        engine_node.meta["val"] = tensor64_out
        parent_graph.output(engine_node)
        parent = torch.fx.GraphModule(root, parent_graph)

        submodule.forward = Mock(
            side_effect=AssertionError("truncate_double executed the partition")
        )
        input_spec = Input((2, 3), dtype=torch.float32)
        repaired = repair_double_inputs(
            parent, submodule, [input_spec], torch.device("cpu"), "run_on_acc_0"
        )

        self.assertEqual(repaired[0].dtype, dtype.float32)

        casts = [
            node
            for node in parent.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
        ]
        self.assertEqual(len(casts), 1)
        self.assertEqual(casts[0].kwargs["dtype"], torch.float64)
        self.assertIs(casts[0].args[0], engine_node)
        self.assertEqual(engine_node.meta["val"].dtype, torch.float32)
        self.assertEqual(casts[0].meta["val"].dtype, torch.float64)

    def test_symbolic_shape_metadata_uses_truncated_binding_dtype(self):
        _, submodule, _ = self._make_graphs(with_scalar_output=True)

        metadata = extract_symbolic_shape_expressions(submodule, truncate_double=True)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["inputs"][0]["dtype"], torch.float32)
        self.assertEqual(metadata["outputs"][0]["dtype"], torch.float32)
        self.assertEqual(metadata["outputs"][1]["dtype"], torch.int64)
        self.assertTrue(metadata["outputs"][1]["is_scalar"])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_float64_weight_restores_output_without_float64_input(self):
        class Float64WeightModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = nn.Parameter(
                    torch.randn((1, 8), dtype=torch.float64),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.bias

        model = Float64WeightModule().eval().cuda()
        input_tensor = torch.randn((4, 8), dtype=torch.float32, device="cuda")
        expected = model(input_tensor)
        exported = torch.export.export(model, (input_tensor,))

        compiled = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=[input_tensor],
            min_block_size=1,
            truncate_double=True,
            pass_through_build_failures=True,
        )
        actual = compiled(input_tensor)

        self.assertEqual(expected.dtype, torch.float64)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

        engine_node = next(
            node for node in compiled.graph.nodes if node.op == "call_module"
        )
        engine_values = engine_node.meta["val"]
        if not isinstance(engine_values, (tuple, list)):
            engine_values = [engine_values]
        self.assertEqual(engine_values[0].dtype, torch.float32)

        restoring_casts = [
            node
            for node in compiled.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
            and node.kwargs.get("dtype") == torch.float64
        ]
        self.assertEqual(len(restoring_casts), 1)


if __name__ == "__main__":
    run_tests()
