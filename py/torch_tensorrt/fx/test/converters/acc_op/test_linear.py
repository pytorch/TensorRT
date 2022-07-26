import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestLinearConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default", [1, 512]),
            ("matrix", [32, 512]),
            ("no_bias", [1, 512], False),
        ]
    )
    def test_linear(
        self,
        test_name,
        shape,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256, bias)

            def forward(self, x):
                return self.linear(x)

        inputs = [torch.randn(shape)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.linear})

    def test_linear_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256)

            def forward(self, x):
                return self.linear(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, 512),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 512), (3, 3, 512), (4, 3, 512))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={acc_ops.linear},
        )

    # Testing with (-1, -1, 512) results into following error:
    # AssertionError: Currently we only support one dynmaic dim for linear and it can't be the last dim.


if __name__ == "__main__":
    run_tests()
