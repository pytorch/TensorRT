import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestAdaptiveAvgPoolConverter(AccTestCase):
    @parameterized.expand(
        [
            ((64, 64),),
            ((128, 64),),
            (64,),
        ]
    )
    def test_adaptive_avgpool(
        self,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 256, 256)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.adaptive_avg_pool2d})

    def test_adaptive_avgpool_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool2d((64, 64))

            def forward(self, x):
                return self.pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, 256, 256),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 256, 256), (3, 3, 256, 256), (5, 5, 256, 256))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.adaptive_avg_pool2d}
        )

    @parameterized.expand(
        [
            ((16, 16, 16),),
            ((32, 16, 4),),
            (32,),
        ]
    )
    def test_adaptive_avgpool3d(
        self,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool3d(output_size)

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32, 64, 64)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.adaptive_avg_pool3d})

    def test_adaptive_avgpool3d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AdaptiveAvgPool3d((16, 16, 16))

            def forward(self, x):
                return self.pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, 32, 64, 64),
                dtype=torch.float32,
                shape_ranges=[
                    ((1, 1, 32, 64, 64), (3, 3, 32, 64, 64), (5, 5, 32, 64, 64))
                ],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.adaptive_avg_pool3d}
        )

    #  Testing with shape(-1, -1, -1, -1) results into error: "AdaptiveAvgPool2d and AdaptiveAvgPool3d currently doesn't support dynamic shapes for last two dims."


if __name__ == "__main__":
    run_tests()
