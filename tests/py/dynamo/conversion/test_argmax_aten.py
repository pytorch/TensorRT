import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestArgmaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # input dimension == 1
            ("dim_1_keep_dim_true", (3,), 0, True),
            ("dim_1_keep_dim_false", (3,), 0, False),
            # dim == None
            ("dim_1_none_true", (3,), None, True),
            ("dim_2_none_true", (3, 3), None, True),
            ("dim_3_none_false", (3, 3, 3), None, False),
            # # common cases
            ("dim_1_keep_dim_true", (3, 3), 1, True),
            ("dim_1_keep_dim_false", (3, 3), 1, False),
            ("dim_0_keep_dim_true", (4, 4, 4), 0, True),
            ("dim_0_keep_dim_false", (4, 4, 4), 0, False),
            ("dim_negative_keep_dim_true", (1, 2, 3), -1, True),
        ]
    )
    def test_argmax(self, _, input_shape, dim, keep_dim):
        class ArgMax(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.argmax.default(input, dim, keep_dim)

        input = [torch.randn(*input_shape)]

        self.run_test(ArgMax(), input)

    @parameterized.expand(
        [
            # input dimension == 1
            ("dim_1_keep_dim_true", (1,), (3,), (3,), 0, True),
            ("dim_1_keep_dim_false", (1,), (3,), (3,), 0, False),
            # dim == None
            ("dim_1_none_true", (1,), (3,), (3,), None, True),
            ("dim_2_none_true", (1, 3), (3, 3), (3, 3), None, True),
            ("dim_3_none_false", (1, 3, 3), (3, 3, 3), (3, 3, 3), None, False),
            # common cases
            ("dim_1_keep_dim_true", (3, 1), (3, 3), (3, 3), 1, True),
            ("dim_1_keep_dim_false", (3, 1), (3, 3), (3, 3), 1, False),
            ("dim_0_keep_dim_true", (1, 4, 4), (4, 4, 4), (4, 4, 4), 0, True),
            ("dim_0_keep_dim_false", (1, 4, 4), (4, 4, 4), (4, 4, 4), 0, False),
            ("dim_negative_keep_dim_true", (1, 4, 4), (4, 4, 4), (4, 4, 4), -3, True),
        ]
    )
    def test_argmax_dynamic(self, _, min_shape, opt_shape, max_shape, dim, keep_dim):
        class ArgMax(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.argmax.default(input, dim, keep_dim)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ArgMax(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
