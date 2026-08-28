import operator
from unittest.mock import Mock

import torch
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

    def test_symbolic_shape_metadata_uses_truncated_binding_dtype(self):
        _, submodule, _ = self._make_graphs(with_scalar_output=True)

        metadata = extract_symbolic_shape_expressions(submodule, truncate_double=True)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["inputs"][0]["dtype"], torch.float32)
        self.assertEqual(metadata["outputs"][0]["dtype"], torch.float32)
        self.assertEqual(metadata["outputs"][1]["dtype"], torch.int64)
        self.assertTrue(metadata["outputs"][1]["is_scalar"])


if __name__ == "__main__":
    run_tests()
