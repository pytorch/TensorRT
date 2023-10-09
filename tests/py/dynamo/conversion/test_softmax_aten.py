import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSoftMaxConverter(DispatchTestCase):
    def test_softmax(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 1, False)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs)

    def test_softmax_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._softmax.default(x, 2, False)

        input_specs = [
            Input(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 5, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
