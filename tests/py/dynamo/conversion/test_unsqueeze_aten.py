import torch
import torch.fx
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from torch_tensorrt import Input

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

    # Testing with more than one dynamic dims results in following error:
    # AssertionError: Currently we don't support unsqueeze with more than one dynamic dims.

    @parameterized.expand(
        [
            ("negative_dim_dynamic", -4),
            ("positive_dim_dynamic", 1),
        ]
    )
    def test_unsqueeze_with_dynamic_shape(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.ops.aten.unsqueeze.default(x, self.dim)

        input_specs = [
            Input(
                shape=(-1, 2, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3), (2, 2, 3), (3, 2, 3))],
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
