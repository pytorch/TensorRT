import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 0, True),
            ((2, 3, 4, 5), 1, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
            ((1, 2, 5, 3), -2, False),
            ((6, 2, 1, 3), -4, True),
        ]
    )
    def test_sort(self, input_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sort(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


class TestSortConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_descending",
                (2, 1, 4),
                (3, 2, 4),
                (3, 3, 4),
                2,
                True,
            ),
            (
                "4d_dynamic_ascending",
                (2, 2, 1, 4),
                (2, 2, 2, 4),
                (3, 3, 2, 4),
                3,
                False,
            ),
            (
                "4d_dynamic_descending_neg_dim",
                (1, 3, 1, 1),
                (2, 3, 2, 2),
                (3, 3, 2, 4),
                -3,
                True,
            ),
        ]
    )
    def test_sort_dynamic(self, _, min_shape, opt_shape, max_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sort(),
            input_specs,
            output_dtypes=[torch.float, torch.int64],
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
