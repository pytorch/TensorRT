import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGroupNormConverter(DispatchTestCase):
    def test_groupnorm1d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.group_norm.default(
                    x,
                    2,
                    torch.ones((6,)),
                    torch.zeros((6,)),
                    1e-05,
                    True,
                )

        inputs = [torch.randn(3, 6, 224)]
        self.run_test(
            GroupNorm(),
            inputs,
        )

    def test_groupnorm2d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.group_norm.default(
                    x,
                    2,
                    torch.randn((6,)),
                    torch.randn((6,)),
                    1e-05,
                    True,
                )

        inputs = [torch.randn(3, 6, 224, 224)]
        with torch.no_grad():
            self.run_test(
                GroupNorm(),
                inputs,
            )

    def test_groupnorm_with_dynamic_shape(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.group_norm.default(
                    x,
                    2,
                    torch.randn((6,)),
                    torch.randn((6,)),
                    1e-05,
                    True,
                )

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(3, 6, 24, 24),
                opt_shape=(5, 6, 24, 24),
                max_shape=(8, 6, 48, 24),
            ),
        ]
        self.run_test_with_dynamic_shape(
            GroupNorm(),
            input_specs,
        )


class TestNativeGroupNormConverter(DispatchTestCase):
    def test_groupnorm1d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_group_norm.default(
                    x,
                    torch.ones((6,)),
                    torch.zeros((6,)),
                    3,
                    6,
                    224,
                    2,
                    1e-05,
                )[0]

        inputs = [torch.randn(3, 6, 224)]
        self.run_test(
            GroupNorm(),
            inputs,
        )

    def test_groupnorm2d(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_group_norm.default(
                    x,
                    torch.ones((6,)),
                    torch.zeros((6,)),
                    3,
                    6,
                    224 * 224,
                    2,
                    1e-05,
                )[0]

        inputs = [torch.randn(3, 6, 224, 224)]
        with torch.no_grad():
            self.run_test(
                GroupNorm(),
                inputs,
            )

    def test_groupnorm_sd(self):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_group_norm.default(
                    x,
                    torch.randn((320,)).half(),
                    torch.randn((320,)).half(),
                    2,
                    320,
                    4096,
                    32,
                    1e-05,
                )[0]

        inputs = [torch.randn(2, 320, 64, 64).half()]
        with torch.no_grad():
            self.run_test(
                GroupNorm(),
                inputs,
            )

    @parameterized.expand(
        [
            (5, 4, 4, 2, (2, 4, 2), (5, 4, 4), (5, 4, 4)),
            (5, 4, 2 * 2, 2, (2, 4, 2, 2), (5, 4, 2, 2), (5, 4, 2, 2)),
            (5, 9, 6 * 3, 3, (3, 9, 3, 3), (5, 9, 6, 3), (5, 9, 6, 3)),
            (8, 9, 6 * 6, 3, (3, 9, 2, 3, 2), (8, 9, 6, 3, 2), (8, 9, 6, 3, 2)),
        ]
    )
    def test_groupnorm_with_dynamic_shape(
        self, N, C, HxW, groups, min_shape, opt_shape, max_shape
    ):
        class GroupNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_group_norm.default(
                    x,
                    torch.ones((C,)),
                    torch.zeros((C,)),
                    N,
                    C,
                    HxW,
                    groups,
                    1e-5,
                )[0]

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            GroupNorm(),
            input_specs,
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
