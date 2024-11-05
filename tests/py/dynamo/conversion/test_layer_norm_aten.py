import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestNativeLayerNormConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 4, 6), [6]),
            ((2, 4, 6), [4, 6]),
            ((2, 4, 6), [2, 4, 6]),
        ]
    )
    def test_layer_norm_1d(self, input_shape, normalized_shape):
        class LayerNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_layer_norm.default(
                    x, normalized_shape, None, None, 1e-05
                )[0]

        inputs = [torch.randn(input_shape)]
        self.run_test(LayerNorm(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            ((5, 3, 2, 4), [4]),
            ((5, 3, 2, 4), [2, 4]),
            ((5, 3, 2, 4), [3, 2, 4]),
            ((5, 3, 2, 4), [5, 3, 2, 4]),
        ]
    )
    def test_layer_norm_2d(self, input_shape, normalized_shape):
        class LayerNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_layer_norm.default(
                    x, normalized_shape, weight, bias, 1e-05
                )[0]

        inputs = [
            torch.randn(input_shape),
            torch.randn(normalized_shape),
            torch.randn(normalized_shape),
        ]
        self.run_test(LayerNorm(), inputs, use_dynamo_tracer=True)

    def test_layernorm_with_dynamic_shape(self):
        class LayerNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_layer_norm.default(
                    x, [3, 224, 224], weight, bias, 1e-05
                )[0]

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 3, 224, 224),
                opt_shape=(5, 3, 224, 224),
                max_shape=(10, 3, 224, 224),
            ),
            Input(dtype=torch.float32, shape=(3, 224, 224)),
            Input(dtype=torch.float32, shape=(3, 224, 224)),
        ]

        self.run_test_with_dynamic_shape(
            LayerNorm(), input_specs, use_dynamo_tracer=True
        )

    def test_layernorm_with_dynamic_shape_1(self):
        class LayerNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.native_layer_norm.default(
                    x, [3], weight, bias, 1e-05
                )[0]

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 2, 3),
                opt_shape=(3, 3, 3),
                max_shape=(4, 5, 3),
            ),
            Input(dtype=torch.float32, shape=(3,)),
            Input(dtype=torch.float32, shape=(3,)),
        ]

        self.run_test_with_dynamic_shape(
            LayerNorm(), input_specs, use_dynamo_tracer=True
        )


if __name__ == "__main__":
    run_tests()
