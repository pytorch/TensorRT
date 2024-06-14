import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

full_like_ops = [
    (
        "full_like_one_dimension",
        torch.empty(1),
        1,
        None,
    ),
    (
        "full_like_two_dimension",
        torch.empty(2, 3),
        1,
        None,
    ),
    (
        "full_like_three_dimension",
        torch.empty(2, 3, 4),
        1,
        None,
    ),
    (
        "full_like_one_dimension_dtype",
        torch.empty(1),
        1,
        torch.float32,
    ),
    (
        "full_like_two_dimension_dtype",
        torch.empty(2, 3),
        1,
        torch.float32,
    ),
    (
        "full_like_four_dimension_dtype",
        torch.empty(1, 2, 2, 1),
        1,
        torch.float32,
    ),
    (
        "full_like_five_dimension_dtype",
        torch.empty(1, 2, 3, 2, 1),
        1,
        torch.float32,
    ),
]


class Testfull_likeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (full_like_op[0], full_like_op[1], full_like_op[2], full_like_op[3])
            for full_like_op in full_like_ops
        ]
    )
    def test_full_like(self, name, tensor, value, data_type):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, tensor):
                return torch.ops.aten.full_like.default(
                    tensor,
                    value,
                    dtype=data_type,
                )

        inputs = [tensor]
        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
