import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestClampConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
            param("float32Boundary", min=-3.4028234663852886e38),
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
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.clamp.default})

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

        class TestScalarModule(torch.nn.Module):
            def forward(self, x):
                y = torch.sum(x)
                return torch.clamp(y, min, max)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (5, 5, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.clamp.default}
        )
        self.run_test_with_dynamic_shape(
            TestScalarModule(), input_specs, expected_ops={torch.ops.aten.clamp.default}
        )


if __name__ == "__main__":
    run_tests()
