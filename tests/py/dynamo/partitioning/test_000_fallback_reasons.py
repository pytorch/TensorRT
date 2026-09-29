from unittest.mock import patch

import tensorrt as trt
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    ConverterRegistry,
    ConverterSupport,
)
from torch_tensorrt.dynamo.partitioning._adjacency_partitioner import OpSupportTester
from torch_tensorrt.dynamo.partitioning._global_partitioner import (
    TorchTensorRTOperatorSupport,
)

SUPPORT_CLASSES = [
    ("fast", OpSupportTester),
    ("global", TorchTensorRTOperatorSupport),
]


class TestFallbackReasons(TestCase):
    def setUp(self):
        super().setUp()
        for name, value in (
            ("compilation_settings", CompilationSettings()),
            ("disallowed_targets", set()),
        ):
            patcher = patch.object(DYNAMO_CONVERTERS, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _node(target, input_value, output_value):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = input_value
        node = graph.call_function(target, (x,))
        node.meta["val"] = output_value
        graph.output(node)
        return node

    @parameterized.expand(SUPPORT_CLASSES)
    def test_refusal_reasons(self, _, support_class):
        x = torch.empty(2, device="cuda")
        high_rank = torch.empty((1,) * (trt.Dims.MAX_DIMS + 1), device="cuda")
        complex_value = torch.empty(2, dtype=torch.complex64, device="cuda")
        cases = [
            (
                torch.ops.aten._to_copy.default,
                x,
                x.cpu(),
                "explicit non-target device region",
            ),
            (
                torch.ops.aten.clone.default,
                high_rank,
                high_rank,
                "tensor rank exceeds TensorRT limit",
            ),
            (
                torch.ops.aten.clone.default,
                complex_value,
                complex_value,
                "complex tensor dtype",
            ),
            (
                torch.ops.aten.nonzero.default,
                x,
                torch.empty(1, 1, dtype=torch.int64, device="cuda"),
                "data-dependent output shape (fallback_data_dependent_ops=True)",
            ),
            (torch.ops.aten.rand_like.default, x, x, "no validated TensorRT converter"),
        ]
        DYNAMO_CONVERTERS.compilation_settings.fallback_data_dependent_ops = True
        for target, input_value, output_value, reason in cases:
            with self.subTest(reason=reason):
                support = support_class()
                node = self._node(target, input_value, output_value)
                name = ConverterRegistry.qualified_name_or_str(target)
                if target == torch.ops.aten.nonzero.default:
                    self.assertTrue(
                        DYNAMO_CONVERTERS[node][2]["requires_output_allocator"]
                    )
                self.assertFalse(support.is_node_supported({}, node))
                self.assertEqual(support.fallback_operators, {name: 1})
                self.assertEqual(support.fallback_reasons, {name: {reason}})
                if target == torch.ops.aten.rand_like.default:
                    self.assertTrue(node.is_impure())
                    self.assertEqual(support.unsupported_operators, {})

    @parameterized.expand(SUPPORT_CLASSES)
    def test_requested_fallback(self, _, support_class):
        x = torch.empty(2, device="cuda")
        target = torch.ops.aten.relu.default
        node = self._node(target, x, x)
        name = ConverterRegistry.qualified_name_or_str(target)
        self.assertIn(node, DYNAMO_CONVERTERS)
        for excluded in (name, target):
            with self.subTest(excluded=excluded):
                support = support_class(torch_executed_ops={excluded})
                self.assertFalse(support.is_node_supported({}, node))
                self.assertEqual(support.fallback_operators, {name: 1})
                self.assertEqual(
                    support.fallback_reasons,
                    {name: {"excluded by torch_executed_ops"}},
                )

    @parameterized.expand(SUPPORT_CLASSES)
    def test_rejected_converter_is_not_reported_as_missing(self, _, support_class):
        x = torch.empty(2, device="cuda")
        target = torch.ops.aten.clone.default
        node = self._node(target, x, x)
        converters = {
            target: [
                ConverterSupport(
                    converter_implementation=lambda *args: None,
                    capability_validator=lambda node, settings: False,
                )
            ]
        }
        with patch.object(DYNAMO_CONVERTERS, "registries", [converters]):
            self.assertIsNotNone(DYNAMO_CONVERTERS.get_unvalidated(target))
            support = support_class()
            self.assertFalse(support.is_node_supported({}, node))
            name = ConverterRegistry.qualified_name_or_str(target)
            self.assertEqual(
                support.fallback_reasons,
                {name: {"no validated TensorRT converter"}},
            )

    @parameterized.expand(SUPPORT_CLASSES)
    def test_same_operator_keeps_multiple_reasons(self, _, support_class):
        support = support_class()
        target = torch.ops.aten.clone.default
        for value in (
            torch.empty((1,) * (trt.Dims.MAX_DIMS + 1), device="cuda"),
            torch.empty(2, dtype=torch.complex64, device="cuda"),
        ):
            self.assertFalse(
                support.is_node_supported({}, self._node(target, value, value))
            )
        name = ConverterRegistry.qualified_name_or_str(target)
        self.assertEqual(support.fallback_operators, {name: 2})
        self.assertEqual(
            support.fallback_reasons,
            {name: {"tensor rank exceeds TensorRT limit", "complex tensor dtype"}},
        )

    @parameterized.expand(SUPPORT_CLASSES)
    def test_supported_and_structural_nodes_have_no_fallback(self, _, support_class):
        x = torch.empty(2, device="cuda")
        node = self._node(torch.ops.aten.relu.default, x, x)
        support = support_class()
        self.assertTrue(support.is_node_supported({}, node))
        for structural in node.graph.nodes:
            if structural.op in ("placeholder", "output"):
                support.is_node_supported({}, structural)
        self.assertEqual(support.fallback_operators, {})
        self.assertEqual(support.fallback_reasons, {})


if __name__ == "__main__":
    run_tests()
