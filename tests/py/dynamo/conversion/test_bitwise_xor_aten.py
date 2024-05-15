import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestBitwiseXorConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (5, 3)),
            ("3d", (5, 3, 2)),
        ]
    )
    def test_bitwise_xor_tensor(self, _, shape):
        class bitwise_xor(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_xor.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_xor(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), True),
            ("3d", (5, 3, 2), False),
        ]
    )
    def test_bitwise_xor_scalar(self, _, shape, scalar):
        class bitwise_xor(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_xor.Scalar(tensor, scalar)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_xor(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), True),
            ("3d", (5, 3, 2), False),
        ]
    )
    def test_bitwise_xor_scalar_tensor(self, _, shape, scalar):
        class bitwise_xor(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_xor.Scalar_Tensor(scalar, tensor)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_xor(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
