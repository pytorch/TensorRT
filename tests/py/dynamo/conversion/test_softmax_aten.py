import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSoftmaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (torch.float, False),
            (torch.half, False),
            (torch.half, True),
        ]
    )
    def test_softmax(self, dtype, half_to_float):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 1, half_to_float)

        inputs = [torch.randn(1, 3, 224, 224, dtype=dtype)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            (torch.float, False),
            (torch.half, False),
            (torch.half, True),
        ]
    )
    def test_softmax_with_dynamic_shape(self, dtype, half_to_float):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 2, half_to_float)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(2, 4, 6, 8),
                max_shape=(8, 8, 8, 8),
                dtype=dtype,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
