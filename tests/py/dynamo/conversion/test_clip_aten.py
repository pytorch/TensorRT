import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestClipConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
            param("float32Boundary", min=-3.4028234663852886e38),
        ]
    )
    def test_clip(self, test_name, min=None, max=None):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clip.default(x, min, max)

        inputs = [torch.randn(3, 4)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            param(
                "defaultInt32",
                min=torch.tensor(-1, dtype=torch.int32),
                max=torch.tensor(0, dtype=torch.int32),
            ),
            param(
                "defaultFloat32",
                min=torch.tensor(0.5, dtype=torch.float32),
                max=torch.tensor(1.0, dtype=torch.float32),
            ),
            param(
                "minBiggerThanMax",
                min=torch.tensor(1.0, dtype=torch.float32),
                max=torch.tensor(0, dtype=torch.int32),
            ),
        ]
    )
    def test_clip(self, test_name, min=None, max=None):
        class TestModule(torch.nn.Module):
            def forward(self, x, min, max):
                return torch.ops.aten.clip.Tensor(x, min, max)

        inputs = [torch.randn(3, 4), min, max]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
        ]
    )
    def test_clip_with_dynamic_shape_four_dimensions(
        self, test_name, min=None, max=None
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clip.default(x, min, max)

        class TestScalarModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.mean.dim(x, None, True)
                return torch.ops.aten.clip.default(y, min, max)

        input_specs = [
            Input(
                shape=(-1, -1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (5, 5, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs)
        self.run_test_with_dynamic_shape(TestScalarModule(), input_specs)


if __name__ == "__main__":
    run_tests()
