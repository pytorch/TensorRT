import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogicalXorConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_logical_xor(self, _, shape):
        class logical_xor(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.logical_xor.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            logical_xor(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d-1d", (2, 1), (3, 5), (4, 7), (1,), (5,), (7,)),
            ("2d-3d", (3, 2), (4, 5), (6, 7), (2, 3, 2), (3, 4, 5), (5, 6, 7)),
            ("2d-3d", (3, 2), (4, 5), (6, 7), (2, 3, 1), (3, 4, 1), (5, 6, 1)),
        ]
    )
    def test_logical_xor_dynamic_shape(
        self,
        _,
        l_min_shape,
        l_opt_shape,
        l_max_shape,
        r_min_shape,
        r_opt_shape,
        r_max_shape,
    ):
        class logical_xor(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.logical_xor.default(lhs_val, rhs_val)

        inputs = [
            Input(
                dtype=torch.bool,
                min_shape=l_min_shape,
                opt_shape=l_opt_shape,
                max_shape=l_max_shape,
                torch_tensor=torch.randint(0, 2, l_opt_shape, dtype=bool),
            ),
            Input(
                dtype=torch.bool,
                min_shape=r_min_shape,
                opt_shape=r_opt_shape,
                max_shape=r_max_shape,
                torch_tensor=torch.randint(0, 2, r_opt_shape, dtype=bool),
            ),
        ]
        self.run_test_with_dynamic_shape(
            logical_xor(),
            inputs,
            use_example_tensors=False,
        )


if __name__ == "__main__":
    run_tests()
