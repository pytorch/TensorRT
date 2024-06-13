import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestLocalScalarDenseConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (torch.randn((5, 10, 5), dtype=torch.float32),),
            (torch.randint(-10, 10, (5, 1, 15), dtype=torch.int32),),
            (torch.randn((1), dtype=torch.float32),),
            ((torch.tensor([-2.4])),),
            ((torch.tensor([5.5, 3.5, 3.6])),),
            ((torch.tensor([True])),),
            (
                torch.tensor(
                    [
                        float("nan"),
                        1.23,
                        float("inf"),
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        float("-inf"),
                        1.23,
                        float("nan"),
                    ]
                ),
            ),
            ((torch.tensor([float("inf")])),),
        ]
    )
    def test_local_scalar_dense(self, data):
        class local_scalar_dense(nn.Module):
            def forward(self, input):
                return torch.ops.aten._local_scalar_dense.default(input)

        inputs = [data]
        self.run_test(
            local_scalar_dense(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
