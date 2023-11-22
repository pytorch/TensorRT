import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestConstantPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2), (1, 1), 0),
            ((2, 1), (2, 1), 1),
            ((3, 4, 2), (1, 2), 2),
            ((3, 4, 2), (1, 2, 3, 1, 2, 3), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 2),
            ((3, 3, 4, 2, 1), (1, 2, 3, 4, 5, 1, 2, 3, 4, 5), 0),
            ((3, 3, 4, 2, 1, 2), (1, 2, 3, 4, 1, 2, 3, 4), 4),
        ]
    )
    def test_constant_pad(self, shape, pad, value):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.constant_pad_nd.default(input, pad, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


if __name__ == "__main__":
    run_tests()
