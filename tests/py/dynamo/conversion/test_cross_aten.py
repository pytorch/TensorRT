import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.aten_ops_converters import cross_validator

from .harness import DispatchTestCase


class TestCrossConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), -1),
            ((4, 3), (4, 3), -1),
            ((4, 3), (4, 3), 1),
            ((3, 4, 5), (3, 4, 5), 0),
            ((2, 3, 4), (2, 3, 4), 1),
        ]
    )
    def test_linalg_cross(self, a_shape, b_shape, dim):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=dim)

        inputs = [torch.randn(a_shape), torch.randn(b_shape)]
        self.run_test(
            Cross(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_linalg_cross_broadcast(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=-1)

        inputs = [torch.randn(5, 3), torch.randn(1, 3)]
        self.run_test(
            Cross(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_linalg_cross_broadcast_non_last_dim(self):
        # PyTorch broadcasts over the non-cross (batch) dims; the cross dim
        # itself (dim=1 here, size 3) must match exactly on both operands.
        # https://docs.pytorch.org/docs/2.8/generated/torch.linalg.cross.html
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=1)

        inputs = [torch.randn(2, 3, 1), torch.randn(1, 3, 4)]
        self.run_test(
            Cross(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_linalg_cross_fp16(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=-1)

        inputs = [
            torch.randn(4, 3, dtype=torch.half),
            torch.randn(4, 3, dtype=torch.half),
        ]
        self.run_test(
            Cross(),
            inputs,
            precision=torch.half,
            use_dynamo_tracer=True,
        )


class TestCrossValidator(TestCase):
    """Directly exercises cross_validator against constructed FX nodes with
    controlled meta, rather than depending on which tracer a harness test
    happens to use to populate meta["val"]."""

    def _cross_node(self, dim=-1):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=dim)

        gm = torch.fx.symbolic_trace(Cross())
        node = next(
            n for n in gm.graph.nodes if n.target == torch.ops.aten.linalg_cross.default
        )
        return node

    def test_missing_metadata_rejected(self):
        # symbolic_trace leaves node.args[i].meta empty -- no "val", no
        # "tensor_meta". With no shape info at all we can't prove the cross
        # dim is size 3, so the validator must reject (not fall open).
        node = self._cross_node()
        self.assertFalse(cross_validator(node))

    def test_symbolic_dim_rejected(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        d = torch.export.Dim("d", min=1, max=8)
        a = torch.randn(4, 3)
        b = torch.randn(4, 3)
        ep = torch.export.export(M(), (a, b), dynamic_shapes={"a": {1: d}, "b": {1: d}})
        gm = ep.module()
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        with gm.graph.inserting_before(out_node):
            node = gm.graph.call_function(
                torch.ops.aten.linalg_cross.default,
                args=(placeholders[0], placeholders[1]),
                kwargs={"dim": -1},
            )
        self.assertFalse(cross_validator(node))

    def test_non_3_sized_dim_rejected(self):
        node = self._cross_node()
        a_node, b_node = node.args[0], node.args[1]
        a_node.meta["val"] = torch.randn(4, 4)
        b_node.meta["val"] = torch.randn(4, 4)
        self.assertFalse(cross_validator(node))

    def test_val_metadata_accepted(self):
        node = self._cross_node()
        a_node, b_node = node.args[0], node.args[1]
        a_node.meta["val"] = torch.randn(4, 3)
        b_node.meta["val"] = torch.randn(4, 3)
        self.assertTrue(cross_validator(node))

    def test_tensor_meta_fallback(self):
        class _TM:
            def __init__(self, shape):
                self.shape = shape

        node = self._cross_node()
        a_node, b_node = node.args[0], node.args[1]

        a_node.meta["tensor_meta"] = _TM((4, 3))
        b_node.meta["tensor_meta"] = _TM((4, 3))
        self.assertTrue(cross_validator(node))

        a_node.meta["tensor_meta"] = _TM((4, 4))
        b_node.meta["tensor_meta"] = _TM((4, 4))
        self.assertFalse(cross_validator(node))


class TestCrossConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 3), (4, 3), (6, 3), -1),
            ((2, 3, 4), (4, 3, 4), (6, 3, 4), 1),
        ]
    )
    def test_linalg_cross_dynamic(self, min_shape, opt_shape, max_shape, dim):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=dim)

        input_specs = [
            Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape),
            Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape),
        ]
        self.run_test_with_dynamic_shape(
            Cross(),
            input_specs,
            use_dynamo_tracer=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestCrossEndToEnd(TestCase):
    """torch.cross and torch.linalg.cross both funnel into
    aten.linalg_cross.default after decomposition (torch.cross resolves its
    optional dim and delegates to linalg_cross), so one converter handles
    both public APIs. These tests go through the full compile() pipeline to
    confirm that end to end.
    """

    def test_torch_linalg_cross_compiles(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.linalg.cross(a, b, dim=-1)

        mod = Cross().eval().cuda()
        inputs = [torch.randn(4, 5, 3).cuda(), torch.randn(4, 5, 3).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_torch_cross_compiles(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.cross(a, b)

        mod = Cross().eval().cuda()
        inputs = [torch.randn(5, 3, 2).cuda(), torch.randn(5, 3, 2).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


if __name__ == "__main__":
    run_tests()
