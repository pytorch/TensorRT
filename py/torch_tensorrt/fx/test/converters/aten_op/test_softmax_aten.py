import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSoftMaxConverter(DispatchTestCase):
    def test_softmax(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax(1)

            def forward(self, x):
                return self.softmax(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten._softmax.default}
        )

    def test_softmax_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax(2)

            def forward(self, x):
                return self.softmax(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 5, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten._softmax.default}
        )


if __name__ == "__main__":
    run_tests()
