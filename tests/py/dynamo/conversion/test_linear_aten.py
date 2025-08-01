import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLinearConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (10, 10),
            (10, 100),
            (100, 10),
            (100, 100),
        ]
    )
    def test_linear_converter(self, in_features, out_features):
        class LinearModel(nn.Module):
            def __init__(self, in_features, out_features):
                super(LinearModel, self).__init__()
                self.linear = nn.Linear(in_features, out_features)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel(in_features, out_features).eval().cuda()
        inputs = [torch.randn(int(torch.randint(1, 20, (1,))), in_features).cuda()]
        self.run_test(model, inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_linear_with_dynamic_shape(self):
        class LinearModel(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.linear.default(x, weight, bias)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 10),
                opt_shape=(10, 10),
                max_shape=(100, 10),
            ),
            Input(dtype=torch.float32, shape=(20, 10)),
            Input(dtype=torch.float32, shape=(20,)),
        ]

        self.run_test_with_dynamic_shape(
            LinearModel(), input_specs, use_dynamo_tracer=True, enable_passes=True
        )


if __name__ == "__main__":
    run_tests()
