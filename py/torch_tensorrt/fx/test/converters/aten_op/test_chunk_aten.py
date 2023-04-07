import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSelectConverterImplicitBatch(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_chunk_dim", 6, 0),
        ]
    )
    def test_chunk(self, _, chunk, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.chunk(input, chunk, dim)
                return out

        input = [torch.randn(11)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.chunk},
        )


class TestSelectConverterExplicitBatch(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_chunk_dim", 6, 0),
        ]
    )
    def test_chunk(self, _, chunk, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.chunk(input, chunk, dim)
                return out

        input = [torch.randn(12)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.chunk},
            test_explicit_precision=True,
        )


if __name__ == "__main__":
    run_tests()
