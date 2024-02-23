import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestEmptyLikeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1), torch.int),
            ((1, 2), torch.int),
            ((2, 1), torch.int),
            ((2, 2), torch.int),
            ((1, 2, 3), torch.int),
            ((1, 3, 2), torch.int),
            ((1), torch.int32),
            ((1, 2), torch.int32),
            ((2, 1), torch.int32),
            ((2, 2), torch.int32),
            ((1, 2, 3), torch.int32),
            ((1, 3, 2), torch.int32),
        ]
    )
    def test_empty_like(self, input_shape, dtype):
        class Empty_Like(nn.Module):
            def forward(self, x):
                return torch.ops.aten.empty_like(x, dtype)

        inputs = [torch.empty(input_shape)]
        self.run_test(
            Empty_Like(),
            inputs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
