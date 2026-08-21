import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException
from torch_tensorrt.dynamo.conversion.aten_ops_converters import glu_validator

from .harness import DispatchTestCase


class TestGluConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("last_dim_fp32", (2, 8), -1, torch.float32),
            ("first_dim_fp32", (6, 4), 0, torch.float32),
            ("middle_dim_fp16", (2, 4, 6), 1, torch.float16),
        ]
    )
    def test_glu(self, _, input_shape, dim, dtype):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, dim)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(Glu(), inputs, use_dynamo_tracer=True)

    def test_glu_keyword_dim(self):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, dim=0)

        inputs = [torch.randn(6, 4)]
        self.run_test(Glu(), inputs, use_dynamo_tracer=False, propagate_shapes=True)

    def test_glu_default_dim(self):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input)

        inputs = [torch.randn(2, 8)]
        self.run_test(Glu(), inputs, use_dynamo_tracer=False, propagate_shapes=True)

    def test_glu_zero_sized_split_dim_rejected(self):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, dim=0)

        inputs = [torch.randn(0, 4)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(Glu(), inputs, use_dynamo_tracer=False, propagate_shapes=True)

    def test_glu_with_dynamic_batch(self):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, -1)

        input_specs = [
            Input(
                min_shape=(2, 4, 8),
                opt_shape=(3, 4, 8),
                max_shape=(5, 4, 8),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(Glu(), input_specs, use_dynamo_tracer=True)


class TestGluValidator(TestCase):
    @staticmethod
    def make_glu_node(input_shape=None, dim=None, *, keyword=False, include_meta=True):
        graph = torch.fx.Graph()
        input_node = graph.placeholder("input")
        args = (input_node,)
        kwargs = {}
        if dim is not None:
            if keyword:
                kwargs["dim"] = dim
            else:
                args += (dim,)
        glu_node = graph.call_function(
            torch.ops.aten.glu.default, args=args, kwargs=kwargs
        )
        graph.output(glu_node)
        if include_meta:
            input_node.meta["val"] = torch.empty(input_shape)
        return glu_node

    def test_keyword_dim(self):
        node = self.make_glu_node((6, 3), dim=0, keyword=True)
        self.assertTrue(glu_validator(node))

    def test_default_dim(self):
        node = self.make_glu_node((3, 8))
        self.assertTrue(glu_validator(node))

    @parameterized.expand(
        [
            ("zero_sized_split_dim", (0, 4), 0),
            ("odd_split_dim", (2, 7), -1),
        ]
    )
    def test_rejects_invalid_split_dim(self, _, input_shape, dim):
        node = self.make_glu_node(input_shape, dim=dim)
        self.assertFalse(glu_validator(node))

    def test_rejects_missing_metadata(self):
        node = self.make_glu_node(include_meta=False)
        self.assertFalse(glu_validator(node))


if __name__ == "__main__":
    run_tests()
