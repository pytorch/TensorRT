import torch
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
                    torch.ones((6,)),
                    torch.zeros((6,)),
                    1e-05,
                    True,
                )

        inputs = [torch.randn(3, 6, 224, 224)]
        with torch.no_grad():
            self.run_test(
                GroupNorm(),
                inputs,
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


if __name__ == "__main__":
    run_tests()
