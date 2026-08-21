import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGluConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("last_dim_fp32", (2, 8), -1, torch.float32),
            ("first_dim_fp32", (6, 4), 0, torch.float32),
            ("middle_dim_fp16", (2, 4, 6), 1, torch.float16),
        ]
    )
    def test_glu(self, _, input_shape, dim, dtype):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, dim)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(Glu(), inputs, use_dynamo_tracer=True)

    def test_glu_with_dynamic_batch(self):
        class Glu(nn.Module):
            def forward(self, input):
                return torch.ops.aten.glu.default(input, -1)

        input_specs = [
            Input(
                min_shape=(2, 4, 8),
                opt_shape=(3, 4, 8),
                max_shape=(5, 4, 8),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(Glu(), input_specs, use_dynamo_tracer=True)


if __name__ == "__main__":
    run_tests()
