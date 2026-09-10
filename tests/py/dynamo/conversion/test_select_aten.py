import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSelectConverterOne(DispatchTestCase):
    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_2d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(1, 2)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_4d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(4, 4, 4, 4)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            (
                "partial_dynamic_static_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                2,
                0,
            ),
            (
                "partial_dynamic_dynamic_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic_neg_dim",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                -1,
                1,
            ),
        ]
    )
    def test_dynamic_shape_select(
        self, _, min_shape, opt_shape, max_shape, type, dim, index
    ):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(select(), input_specs)

    def test_select_runtime_index(self):
        """The index is a tensor rather than a constant here, which is what a preceding Torch
        subgraph produces. Selecting row 3 of an (8, 4) input must give shape (4,), the same
        as eager: add_gather keeps the axis it gathers along unless the index is rank 0.
        """

        class SelectRuntimeIndex(nn.Module):
            def forward(self, values, positions):
                index = torch.ops.aten.sum.default(positions)
                return torch.ops.aten.select.int(values, 0, index)

        inputs = [
            torch.randn(8, 4),
            torch.tensor([1, 2], dtype=torch.int64),
        ]
        self.run_test(
            SelectRuntimeIndex(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
