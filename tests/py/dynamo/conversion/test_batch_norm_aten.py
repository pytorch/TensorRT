import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase

FEATURE_NUM = 3


class TestBatchNormConverter(DispatchTestCase):
    def test_batchnorm_static_weights(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.batch_norm.default(
                    x,
                    torch.full((FEATURE_NUM,), 3, dtype=torch.float32),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.full((FEATURE_NUM,), 3, dtype=torch.float32),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_batchnorm_ITensor_weights_bias(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.batch_norm.default(
                    x,
                    weight,
                    bias,
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        inputs = [
            torch.randn(1, 3, 224, 224),
            torch.ones((FEATURE_NUM,)),
            torch.zeros((FEATURE_NUM,)),
        ]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_batchnorm_ITensor_weights(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten.batch_norm.default(
                    x,
                    weight,
                    None,
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        inputs = [
            torch.randn(1, 3, 224, 224),
            torch.ones((FEATURE_NUM,)),
        ]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_batchnorm_static_bias_only(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.batch_norm.default(
                    x,
                    None,
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_batchnorm1d_with_dynamic_shape(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.batch_norm.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        input_specs = [
            Input(
                shape=(-1, 3, 5),
                dtype=torch.float32,
                shape_ranges=[((2, 3, 5), (6, 3, 5), (10, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            BatchNorm(),
            input_specs,
        )

    def test_batchnorm2d_with_dynamic_shape(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.batch_norm.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                    True,
                )

        input_specs = [
            Input(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 5, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            BatchNorm(),
            input_specs,
        )


class TestNativeBatchNormConverter(DispatchTestCase):
    def test_native_batchnorm_static_weights(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_batch_norm.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_native_batchnorm_legit_no_training_with_trt_tensor(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x, running_mean, running_var):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    running_mean,
                    running_var,
                    0.1,
                    1e-05,
                )[0]

        inputs = [
            torch.randn(1, 3, 224, 224),
            torch.zeros((FEATURE_NUM,)),
            torch.ones((FEATURE_NUM,)),
        ]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_native_batchnorm_legit_no_training_with_static_means(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_native_batchnorm1d_with_dynamic_shape(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_batch_norm.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                )[0]

        input_specs = [
            Input(
                shape=(-1, 3, 5),
                dtype=torch.float32,
                shape_ranges=[((2, 3, 5), (6, 3, 5), (10, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            BatchNorm(),
            input_specs,
        )

    def test_native_batchnorm2d_with_dynamic_shape(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_batch_norm.default(
                    x,
                    torch.ones((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.zeros((FEATURE_NUM,)),
                    torch.ones((FEATURE_NUM,)),
                    False,
                    0.1,
                    1e-05,
                )[0]

        input_specs = [
            Input(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 5, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            BatchNorm(),
            input_specs,
        )


class TestBatchNormConvFusion(DispatchTestCase):
    """Regression test for https://github.com/pytorch/TensorRT/issues/3699.

    When Conv output is negative and feeds into BatchNorm, the fused
    Conv+Scale kernel must not produce NaN. This test uses the full
    torch_tensorrt.dynamo.compile() pipeline to exercise the TRT optimizer's
    layer fusion.
    """

    def test_conv_batchnorm_conv_no_nan(self):
        import torch_tensorrt

        class ConvBNConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(1)
                self.conv2 = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)

            def forward(self, x):
                return self.conv2(self.bn1(self.conv1(x)))

        model = ConvBNConv().eval().to("cuda")
        with torch.no_grad():
            model.conv1.weight.fill_(1.0)
            model.conv2.weight.fill_(1.0)

        inp = torch.full((1, 1, 1, 1), -1.0, device="cuda")
        exp_program = torch.export.export(model, (inp,), strict=False)
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_program,
            inputs=[torch_tensorrt.Input(shape=(1, 1, 1, 1), dtype=torch.float32)],
            device=torch_tensorrt.Device("cuda:0"),
            enabled_precisions={torch.float32},
            min_block_size=1,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        pyt_out = model(inp)
        trt_out = trt_mod(inp)

        nan_count = torch.isnan(trt_out).sum().item()
        self.assertEqual(nan_count, 0, "TRT output contains NaN")
        torch.testing.assert_close(trt_out, pyt_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
