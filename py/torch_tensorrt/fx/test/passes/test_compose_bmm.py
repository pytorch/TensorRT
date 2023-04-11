import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer import trace
from torch_tensorrt.fx.passes.lower_basic_pass_aten import compose_bmm


class TestComposeBMM(TestCase):
    @parameterized.expand(
        [
            ("3_dim", (2, 3, 4), (2, 4, 3)),
            ("3_dim_same_shape", (4, 4, 4), (4, 4, 4)),
        ]
    )
    def test_compose_bmm(self, test_name, x_shape, y_shape):
        class BMM(nn.Module):
            def forward(self, x, y):
                return torch.bmm(x, y)

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        fx_model, _ = trace(BMM(), inputs)
        composed_module = compose_bmm(fx_model)
        out = composed_module.graph_module(*inputs)


if __name__ == "__main__":
    run_tests()
