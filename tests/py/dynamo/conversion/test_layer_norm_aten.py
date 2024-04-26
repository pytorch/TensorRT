import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLayerNormConverter(DispatchTestCase):
    def test_layer_norm(self):
        class LayerNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.layer_norm.default(
                    x,
                    torch.tensor([3, 224, 224]),
                    torch.ones((3, 224, 224)),
                    torch.zeros((3, 224, 224)),
                    1e-05,
                    True,
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            LayerNorm(),
            inputs,
        )


class TestNativeLayerNormConverter(DispatchTestCase):
    def test_layer_norm(self):
        class LayerNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_layer_norm.default(
                    x,
                    torch.tensor([3, 224, 224]),
                    torch.ones((3, 224, 224)),
                    torch.zeros((3, 224, 224)),
                    1e-05,
                )[0]

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            LayerNorm(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
