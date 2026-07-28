import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestPermuteConverter(AccTestCase):
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
        self.run_test(Permute(), inputs, expected_ops={acc_ops.permute})

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
        self.run_test(Permute(), inputs, expected_ops={acc_ops.permute})

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
        self.run_test(Transpose(), inputs, expected_ops={acc_ops.permute})

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
            Permute(), input_specs, expected_ops={acc_ops.permute}
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
            Permute(), input_specs, expected_ops={acc_ops.permute}
        )


if __name__ == "__main__":
    run_tests()
