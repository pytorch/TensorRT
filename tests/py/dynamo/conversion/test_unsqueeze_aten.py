import torch
import torch.fx
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


class TestUnsqueeze(DispatchTestCase):
    @parameterized.expand(
        [
            ("negative_dim", -2),
            ("positive_dim", 2),
        ]
    )
    def test_unsqueeze(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.ops.aten.unsqueeze.default(x, self.dim)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Unsqueeze(dim), inputs)

    @parameterized.expand(
        [
            ("1_dynamic_shape_2d_-3", -3, (2, 5), (3, 5), (4, 5)),
            ("1_dynamic_shape_2d_-2", -2, (2, 3), (2, 4), (2, 5)),
            ("1_dynamic_shape_2d_-1", -1, (2, 3), (2, 4), (2, 5)),
            ("1_dynamic_shape_2d_0", 0, (2, 3), (2, 4), (2, 5)),
            ("1_dynamic_shape_2d_1", 1, (2, 3), (2, 4), (2, 5)),
            ("1_dynamic_shape_2d_2", 2, (2, 3), (2, 4), (2, 5)),
            ("2_dynamic_shape_3d_-1", -1, (2, 2, 3), (4, 3, 3), (5, 5, 3)),
            ("2_dynamic_shape_3d_0", 2, (2, 2, 3), (4, 3, 3), (5, 5, 3)),
            ("2_dynamic_shape_3d_1", 1, (2, 2, 3), (4, 3, 3), (5, 6, 3)),
            ("2_dynamic_shape_3d_2", 2, (2, 2, 3), (4, 3, 3), (6, 5, 3)),
            ("4_dynamic_shape_4d_-4", -4, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
            ("4_dynamic_shape_4d_-3", -3, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
            ("4_dynamic_shape_4d_-2", -2, (1, 2, 3, 4), (2, 2, 3, 5), (4, 3, 5, 6)),
            ("4_dynamic_shape_4d_-1", -1, (1, 2, 3, 4), (2, 2, 3, 5), (4, 3, 5, 6)),
            ("4_dynamic_shape_4d_0", 0, (1, 2, 3, 4), (2, 2, 5, 7), (2, 3, 6, 8)),
            ("4_dynamic_shape_4d_1", 1, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
            ("4_dynamic_shape_4d_2", 2, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
            ("4_dynamic_shape_4d_3", 3, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
            ("4_dynamic_shape_4d_4", 4, (1, 2, 3, 4), (2, 2, 3, 5), (3, 3, 5, 5)),
        ]
    )
    def test_unsqueeze_with_dynamic_shape(
        self, _, dim, min_shape, opt_shape, max_shape
    ):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.ops.aten.unsqueeze.default(x, self.dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(Unsqueeze(dim), input_specs)


class TestBroadcastInDim(DispatchTestCase):
    def test_broadcast_in_dim_supported(
        self,
    ):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.ops.prims.broadcast_in_dim.default(
                    x, [4, 5, 6, 1, 1], [0, 1, 2]
                )

        inputs = [torch.randn(4, 5, 6)]
        self.run_test(
            Unsqueeze(),
            inputs,
        )

    def test_broadcast_in_dim_supported_singleton(
        self,
    ):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.ops.prims.broadcast_in_dim.default(x, [1, 1, 1], [0, 1])

        inputs = [torch.randn(1, 1)]
        self.run_test(
            Unsqueeze(),
            inputs,
        )

    # TODO: Remove this test when support is updated
    def test_broadcast_in_dim_unsupported(
        self,
    ):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.ops.prims.broadcast_in_dim.default(
                    x, [4, 5, 6, 7, 1], [0, 1, 2]
                )

        inputs = [torch.randn(4, 5, 6)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(
                Unsqueeze(),
                inputs,
            )


if __name__ == "__main__":
    run_tests()
