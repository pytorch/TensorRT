from typing import Callable

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

unary_ops = [
    (torch.sqrt, torch.ops.aten.sqrt.default, False),
]


class TestRSqrtConverter(DispatchTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1], op[2]) for op in unary_ops])
    def test_unary_ops(
        self, name, orig_op: Callable, expected_op: Callable, range_req: bool
    ):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x)

        m = TestModule(orig_op)
        if orig_op.__name__ == "sqrt":
            inputs = [torch.randn((2, 1)) + 1]
        else:
            inputs = (
                [torch.distributions.uniform.Uniform(-1, 1).sample([2, 2, 3])]
                if range_req
                else [torch.randn(2, 2, 3)]
            )
        self.run_test(m, inputs, expected_ops={expected_op})


if __name__ == "__main__":
    run_tests()
