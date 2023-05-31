import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestPermuteConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("positive", [0, 2, 1]),
            ("negative", [0, -1, -2]),
        ]
    )
    def test_permute_list(self, _, permutation):
        class Permute(nn.Module):
            def forward(self, x):
                return x.permute(permutation)

        inputs = [torch.randn(1, 3, 2)]
        self.run_test(Permute(), inputs, expected_ops={torch.ops.aten.permute.default})

    @parameterized.expand(
        [
            ("positive", [0, 2, 1]),
            ("negative", [0, -1, -2]),
        ]
    )
    def test_permute(self, _, permutation):
        class Permute(nn.Module):
            def forward(self, x):
                return x.permute(*permutation)

        inputs = [torch.randn(1, 3, 2)]
        self.run_test(Permute(), inputs, expected_ops={torch.ops.aten.permute.default})

    @parameterized.expand(
        [
            ("positive", (1, 2)),
            ("negative", (-1, -2)),
        ]
    )
    def test_transpose(self, _, dims):
        class Transpose(nn.Module):
            def forward(self, x):
                return x.transpose(*dims)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Transpose(), inputs, expected_ops={torch.ops.aten.transpose.int})

    def test_permute_with_dynamic_shape(self):
        class Permute(nn.Module):
            def forward(self, x):
                return x.permute(1, 2, 0)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Permute(), input_specs, expected_ops={torch.ops.aten.permute.default}
        )

    def test_permute_with_dynamic_shape_four_dimensions(self):
        class Permute(nn.Module):
            def forward(self, x):
                return x.permute(1, 2, 3, 0)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Permute(), input_specs, expected_ops={torch.ops.aten.permute.default}
        )


if __name__ == "__main__":
    run_tests()
