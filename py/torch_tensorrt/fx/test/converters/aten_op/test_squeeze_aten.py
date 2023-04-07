import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (0), (2, 1)),
            ("3d_one_dim", (0), (2, 2, 1)),
            # ("3d_two_dim", (0, 1), (2, 2, 1)),
            # ("4d_dim", (0, 1, 2), (2, 2, 2, 1)),
        ]
    )
    def test_squeeze(self, _, dim, init_size):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim)

        inputs = [torch.randn(*init_size)]
        expected_op = {}
        if isinstance(dim, int) == 1:
            expected_op = {torch.ops.aten.squeeze.dim}
        else:
            expected_op = {torch.ops.aten.squeeze.dims}
        self.run_test(
            Squeeze(),
            inputs,
            expected_ops=expected_op,
        )


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (1), (-1, 1), [((1, 1), (1, 1), (3, 1))]),
            ("3d_one_dim", (1), (-1, 2, 1), [((1, 2, 1), (1, 2, 1), (3, 2, 1))]),
            # ("3d_two_dim", (0, 1), (-1, -1, 1), [((1, 3, 1, 1), (1, 3, 1, 1))]),
        ]
    )
    def test_squeeze(self, _, dim, init_size, shape_range):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim)

        if isinstance(dim, int) == 1:
            expected_op = {torch.ops.aten.squeeze.dim}
        else:
            expected_op = {torch.ops.aten.squeeze.dims}
        input_specs = [
            InputTensorSpec(
                shape=init_size,
                dtype=torch.float32,
                shape_ranges=shape_range,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Squeeze(),
            input_specs,
            expected_ops=expected_op,
        )


if __name__ == "__main__":
    run_tests()
