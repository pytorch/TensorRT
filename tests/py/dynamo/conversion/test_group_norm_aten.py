import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestNativeGroupNormConverter(DispatchTestCase):
    def test_groupnorm_1d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_group_norm.default(
                    x, None, None, 3, 6, 224, 2, 1e-05
                )[0]

        inputs = [torch.randn(3, 6, 224)]
        self.run_test(GroupNorm(), inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_groupnorm_2d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_group_norm.default(
                    x, weight, bias, 3, 6, 224 * 224, 2, 1e-05
                )[0]

        inputs = [torch.randn(3, 6, 224, 224), torch.ones(6), torch.zeros(6)]
        self.run_test(GroupNorm(), inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_groupnorm_sd(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_group_norm.default(
                    x, weight, bias, 2, 320, 64 * 64, 32, 1e-05
                )[0]

        inputs = [
            torch.randn(2, 320, 64, 64, dtype=torch.half),
            torch.randn(320, dtype=torch.half),
            torch.randn(320, dtype=torch.half),
        ]
        self.run_test(
            GroupNorm(),
            inputs,
            precision=torch.half,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (5, 4, 4, 1, (2, 4, 2), (3, 4, 2), (5, 4, 4)),
            (5, 4, 2 * 2, 4, (2, 4, 2, 2), (3, 4, 2, 2), (5, 4, 2, 2)),
            (5, 9, 6 * 3, 3, (3, 9, 3, 3), (4, 9, 3, 3), (5, 9, 6, 3)),
            (8, 9, 6 * 3 * 2, 3, (3, 9, 2, 3, 2), (5, 9, 3, 3, 2), (8, 9, 6, 3, 2)),
        ]
    )
    def test_groupnorm_with_dynamic_shape(
        self, N, C, HxW, group, min_shape, opt_shape, max_shape
    ):
        class GroupNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_group_norm.default(
                    x, weight, bias, N, C, HxW, group, 1e-05
                )[0]

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
            Input(dtype=torch.float32, shape=(C,)),
            Input(dtype=torch.float32, shape=(C,)),
        ]
        self.run_test_with_dynamic_shape(GroupNorm(), input_specs, check_dtype=False)


if __name__ == "__main__":
    run_tests()
