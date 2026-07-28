import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


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
                return torch.ops.aten.clamp.default(x, min, max)

        inputs = [torch.randn(3, 4)]
        self.run_test(TestModule(), inputs)

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
                return torch.ops.aten.clamp.default(x, min, max)

        class TestScalarModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.mean.dim(x, None, True)
                return torch.ops.aten.clamp.default(y, min, max)

        input_specs = [
            Input(
                min_shape=(1, 1, 3, 3),
                opt_shape=(3, 3, 3, 3),
                max_shape=(5, 5, 3, 3),
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)
        self.run_test_with_dynamic_shape(TestScalarModule(), input_specs)

    @parameterized.expand(
        [
            param("default", min=-1 * torch.randn(3, 4), max=0 * torch.randn(3, 4)),
            param("min", min=0.5 * torch.randn(3, 4)),
            param("max", max=0.5 * torch.randn(3, 4)),
            param(
                "minBiggerThanMax", min=1 * torch.randn(3, 4), max=0 * torch.randn(3, 4)
            ),
            param("float32Boundary", min=-3.4028234663852886e38 * torch.randn(3, 4)),
        ]
    )
    def test_clamp_tensor(
        self,
        test_name,
        min=None,
        max=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clamp.Tensor(x, min, max)

        inputs = [torch.randn(3, 4)]
        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
