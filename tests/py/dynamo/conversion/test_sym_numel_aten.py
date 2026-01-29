import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSymNumelConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("1d", (6,)),
            ("2d", (3, 4)),
            ("3d", (2, 3, 4)),
            ("4d", (2, 3, 4, 5)),
        ]
    )
    def test_sym_numel(self, _, input_shape):
        class NumelModel(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sym_numel.default(x)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            NumelModel(),
            inputs,
        )

    @parameterized.expand(
        [
            ("1d_dynamic", (2,), (4,), (8,)),
            ("2d_dynamic_batch", (1, 4), (3, 4), (6, 4)),
            ("2d_dynamic_all", (2, 2), (4, 4), (8, 8)),
            ("3d_dynamic", (1, 2, 4), (2, 3, 4), (4, 4, 4)),
        ]
    )
    def test_sym_numel_dynamic_shape(self, _, min_shape, opt_shape, max_shape):
        class NumelModel(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sym_numel.default(x)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            NumelModel(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
