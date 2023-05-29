import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSqueeze(DispatchTestCase):
    def test_squeeze(self):
        class Squeeze(nn.Module):
            def forward(self, x):
                return x.squeeze(2)

        inputs = [torch.randn(1, 2, 1)]
        self.run_test(Squeeze(), inputs, expected_ops={torch.ops.aten.squeeze.dim})

    # Testing with shape=(-1, -1, -1, -1) results in error:
    # AssertionError: We don't support squeeze dynamic dim.

    # Testing with more than one dynamic dim results in error:
    # AssertionError: Currently more than one dynamic dim for input to squeeze is not supported.

    def test_squeeze_with_dynamic_shape(self):
        class Squeeze(nn.Module):
            def forward(self, x):
                return x.squeeze(0)

        input_specs = [
            InputTensorSpec(
                shape=(1, -1, 2),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 2), (1, 2, 2), (1, 3, 2))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Squeeze(), input_specs, expected_ops={torch.ops.aten.squeeze.dim}
        )


if __name__ == "__main__":
    run_tests()
