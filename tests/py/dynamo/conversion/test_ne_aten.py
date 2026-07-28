import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestNotEqualConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (5, 3)),
            ("3d", (5, 3, 2)),
        ]
    )
    def test_ne_tensor(self, _, shape):
        class ne(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.ne.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 3, shape, dtype=torch.int32),
            torch.randint(0, 3, shape, dtype=torch.int32),
        ]
        self.run_test(
            ne(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), 1),
            ("3d", (5, 3, 2), 2.0),
        ]
    )
    def test_ne_tensor_scalar(self, _, shape, scalar):
        class ne(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.ne.Tensor(lhs_val, torch.tensor(scalar))

        inputs = [torch.randint(0, 3, shape, dtype=torch.int32)]
        self.run_test(
            ne(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), 1),
            ("3d", (5, 3, 2), 2.0),
        ]
    )
    def test_ne_scalar(self, _, shape, scalar):
        class ne(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.ne.Scalar(lhs_val, scalar)

        inputs = [torch.randint(0, 3, shape, dtype=torch.int32)]
        self.run_test(
            ne(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (5,)),
            ((1, 20), (2, 20), (3, 20)),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 5, 6), (4, 5, 6, 7)),
        ]
    )
    def test_ne_tensor_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class ne(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.ne.Tensor(lhs_val, rhs_val)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ne(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (5,)),
            ((1, 20), (2, 20), (3, 20)),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 5, 6), (4, 5, 6, 7)),
        ]
    )
    def test_ne_tensor_scalar_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class ne(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.ne.Tensor(lhs_val, torch.tensor(1))

        input_specs = [
            Input(
                dtype=torch.int32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ne(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (5,)),
            ((1, 20), (2, 20), (3, 20)),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 5, 6), (4, 5, 6, 7)),
        ]
    )
    def test_ne_scalar_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class ne(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.ne.Scalar(lhs_val, 1.0)

        input_specs = [
            Input(
                dtype=torch.int32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ne(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
