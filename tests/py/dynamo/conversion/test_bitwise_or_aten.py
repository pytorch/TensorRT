import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestBitwiseOrConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 3), (2, 3)),
            ("3d", (5, 3, 2), (5, 3, 2)),
            ("3d_broadcast", (2, 3), (2, 1, 3)),
            ("4d_broadcast_1", (2, 3), (1, 2, 1, 3)),
            ("4d_broadcast_2", (2, 3), (2, 2, 2, 3)),
        ]
    )
    def test_bitwise_or_tensor(self, _, lhs_shape, rhs_shape):
        class bitwise_or(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_or.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 2, lhs_shape, dtype=bool),
            torch.randint(0, 2, rhs_shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d-2d", (2, 3), (3, 3), (5, 3), (2, 3), (3, 3), (5, 3)),
            ("3d-3d", (2, 2, 2), (2, 3, 2), (2, 4, 2), (1, 2, 2), (1, 3, 2), (1, 4, 2)),
        ]
    )
    def test_bitwise_or_tensor_dynamic_shape(
        self,
        _,
        lhs_min_shape,
        lhs_opt_shape,
        lhs_max_shape,
        rhs_min_shape,
        rhs_opt_shape,
        rhs_max_shape,
    ):
        class bitwise_or(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_or.Tensor(lhs_val, rhs_val)

        inputs = [
            Input(
                dtype=torch.bool,
                min_shape=lhs_min_shape,
                opt_shape=lhs_opt_shape,
                max_shape=lhs_max_shape,
                torch_tensor=torch.randint(0, 2, lhs_opt_shape, dtype=bool),
            ),
            Input(
                dtype=torch.bool,
                min_shape=rhs_min_shape,
                opt_shape=rhs_opt_shape,
                max_shape=rhs_max_shape,
                torch_tensor=torch.randint(0, 2, rhs_opt_shape, dtype=bool),
            ),
        ]
        self.run_test_with_dynamic_shape(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), True),
            ("3d", (5, 3, 2), False),
        ]
    )
    def test_bitwise_or_scalar(self, _, shape, scalar):
        class bitwise_or(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_or.Scalar(tensor, scalar)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
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
    def test_bitwise_or_scalar_tensor(self, _, shape, scalar):
        class bitwise_or(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_or.Scalar_Tensor(scalar, tensor)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
