import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestExpandConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (2, 3), (2, 1)),
            ("3d_dim", (2, 3, 4), (2, 1, 1)),
            ("4d_dim", (2, 3, 4, 5), (2, 1, 1, 1)),
            ("keep_dim", (2, 3, -1, -1), (2, 1, 5, 5)),
            ("different_ranks", (2, 3, -1, -1), (1, 5, 7)),
        ]
    )
    def test_expand(self, _, sizes, init_size):
        class Expand(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, sizes)

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Expand(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_dim", (2, 1), (4, 1), (6, 1), (-1, 3)),
            ("3d_dim", (2, 1, 1), (4, 1, 1), (6, 1, 1), (-1, 3, 4)),
            ("4d_dim", (1, 1, 1, 1), (3, 1, 1, 1), (5, 1, 1, 1), (-1, 2, 3, 6)),
            ("keep_dim", (2, 1, 5, 5), (4, 1, 5, 5), (6, 1, 5, 5), (-1, 3, -1, -1)),
            ("different_ranks", (1, 2, 1), (1, 2, 1), (2, 2, 1), (2, -1, -1, -1)),
        ]
    )
    def test_expand_dynamic(self, _, min_shape, opt_shape, max_shape, expanded_shape):
        class ExpandDynamic(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, expanded_shape)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ExpandDynamic(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
