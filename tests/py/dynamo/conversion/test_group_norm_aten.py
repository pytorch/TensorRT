import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGroupNormConverter(DispatchTestCase):
    def test_groupnorm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = torch.nn.GroupNorm(2, 6)

            def forward(self, x):
                return self.gn(x)

        inputs = [torch.randn(1, 6, 224, 224)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={torch.ops.aten.native_group_norm.default},
            disable_passes=True,
        )

    def test_groupnorm_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = torch.nn.GroupNorm(2, 6)

            def forward(self, x):
                return self.gn(x)

        input_specs = [
            Input(
                shape=(-1, 6, 5),
                dtype=torch.float32,
                shape_ranges=[((2, 6, 5), (6, 6, 5), (10, 6, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={torch.ops.aten.native_group_norm.default},
            disable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
