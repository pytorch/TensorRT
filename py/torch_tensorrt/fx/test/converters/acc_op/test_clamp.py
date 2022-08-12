import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestClampConverter(AccTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
        ]
    )
    def test_clamp(
        self,
        test_name,
        min=None,
        max=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min, max)

        inputs = [torch.randn(3, 4)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.clamp})

    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
        ]
    )
    def test_clamp_with_dynamic_shape_four_dimensions(
        self,
        test_name,
        min=None,
        max=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min, max)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (5, 5, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.clamp}
        )


if __name__ == "__main__":
    run_tests()
