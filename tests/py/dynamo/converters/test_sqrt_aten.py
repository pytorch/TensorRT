import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.test_utils import DispatchTestCase
from torch_tensorrt import Input
import numpy as np


class TestRSqrtConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # int is giving pytorch error. But int cases are covered below
            # ("2d_dtype_int", (2, 1), torch.int32),
            # ("3d_dtype_int", (2, 1, 2),torch.int32),
            ("2d_dim_dtype_float", (2, 1, 2), torch.float),
            ("3d_dim_dtype_float", (2, 1, 2), torch.float),
        ]
    )
    def test_sqrt(self, _, x, type):
        class sqrt(nn.Module):
            def forward(self, input):
                return torch.sqrt(input)

        inputs = [torch.randn(x, dtype=type)]
        self.run_test(
            sqrt(),
            inputs,
            expected_ops={torch.ops.aten.sqrt.default},
        )

    @parameterized.expand(
        [
            ("2d_dtype_pos_int_min_max", (2, 1), torch.int32, 0, 5),
            ("3d_dtype_pos_int_min_max", (2, 1, 2), torch.int32, 0, 5),
            ("2d_dtype_pos_float_min_max", (2, 1), torch.float, 0, 5),
            ("3d_dtype_pos_float_min_max", (2, 1, 2), torch.float, 0, 5),
            ("2d_dtype_neg_int_min_max", (2, 1), torch.int32, -10, -5),
            ("3d_dtype_neg_int_min_max", (2, 1, 2), torch.int32, -10, -5),
            ("2d_dtype_neg_float_min_max", (2, 1), torch.float, -10, -5),
            ("3d_dtype_neg_float_min_max", (2, 1, 2), torch.float, -10, -5),
        ]
    )
    def test_sqrt(self, _, x, type, min, max):
        class sqrt(nn.Module):
            def forward(self, input):
                return torch.sqrt(input)

        inputs = [torch.randint(min, max, (x), dtype=type)]
        self.run_test(
            sqrt(),
            inputs,
            expected_ops={torch.ops.aten.sqrt.default},
        )


if __name__ == "__main__":
    run_tests()
