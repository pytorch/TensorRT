import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (0), (2, 1)),
            ("3d_one_dim", (0), (2, 2, 1)),
        ]
    )
    def test_squeeze_single_dim(self, _, dim, init_size):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dim(x, dim)

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Squeeze(),
            inputs,
        )

    @parameterized.expand(
        [
            ("3d_two_dim", (0, 1), (2, 1, 1)),
            ("4d_dim", (0, 1, 2), (2, 2, 1, 1)),
        ]
    )
    def test_squeeze_multi_dims(self, _, dim, init_size):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dims(x, dim)

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Squeeze(),
            inputs,
        )


class TestSqueezeConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "5d_two_dynamic_shape_-1",
                (0,),
                (1, 1, 1, 1, 1),
                (1, 2, 1, 2, 1),
                (1, 4, 1, 3, 1),
            ),
            (
                "5d_two_dynamic_shape_-2",
                (0, 2),
                (1, 1, 1, 1, 1),
                (1, 2, 1, 2, 1),
                (1, 4, 1, 3, 1),
            ),
            (
                "5d_three_dynamic_shape_-2",
                (0, 4),
                (1, 1, 1, 1, 1),
                (1, 2, 4, 2, 1),
                (1, 4, 4, 3, 1),
            ),
            (
                "4d_two_dynamic_shape_-2",
                (0, 2),
                (1, 1, 2, 1),
                (1, 2, 2, 2),
                (1, 4, 2, 3),
            ),
        ]
    )
    def test_squeeze(self, _, dim, min_shape, opt_shape, max_shape):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dims(x, dim)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Squeeze(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
