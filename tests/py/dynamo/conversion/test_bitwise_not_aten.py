import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestBitwiseNotConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (5, 3)),
            ("3d", (5, 3, 2)),
        ]
    )
    def test_bitwise_not_tensor(self, _, shape):
        class bitwise_not(nn.Module):
            def forward(self, val):
                return torch.ops.aten.bitwise_not.default(val)

        inputs = [
            torch.randint(0, 2, shape, dtype=torch.bool),
        ]
        self.run_test(
            bitwise_not(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d", (2, 3), (5, 3), (6, 4)),
            ("3d", (2, 3, 2), (3, 4, 2), (5, 4, 2)),
        ]
    )
    def test_bitwise_not_tensor_dynamic_shape(self, _, min_shape, opt_shape, max_shape):
        class bitwise_not(nn.Module):
            def forward(self, val):
                return torch.ops.aten.bitwise_not.default(val)

        inputs = [
            Input(
                dtype=torch.bool,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                torch_tensor=torch.randint(0, 2, opt_shape, dtype=bool),
            )
        ]
        self.run_test_with_dynamic_shape(
            bitwise_not(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )


if __name__ == "__main__":
    run_tests()
