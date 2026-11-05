# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import TRTInterpreter

from .harness import DispatchTestCase, skip_if_trt_rtx_turing

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

    @parameterized.expand(
        [
            ("fp16_1d", torch.float16, (2, FEATURE_NUM)),
            ("fp16_3d", torch.float16, (2, FEATURE_NUM, 4, 8, 8)),
            ("bf16_2d", torch.bfloat16, (2, FEATURE_NUM, 8, 8)),
        ]
    )
    def test_native_batchnorm_legit_no_training_low_precision_input_fp32_weights(
        self, _, dtype, shape
    ):
        if dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        # torch.autocast runs batch norm on a float16 input but keeps its weights
        # and running stats in float32.
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    torch.tensor([0.5, 1.0, 2.0]),
                    torch.tensor([-0.5, 0.0, 0.5]),
                    torch.tensor([0.1, -0.2, 0.3]),
                    torch.tensor([0.5, 1.0, 1.5]),
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(shape, dtype=dtype)]
        # The scale and shift run in bfloat16, which keeps about 2 digits.
        tol = {"rtol": 1.6e-2, "atol": 1.6e-2} if dtype == torch.bfloat16 else {}
        self.run_test(
            BatchNorm(),
            inputs,
            **tol,
        )

    def _network(self, module, inputs, **settings):
        mod = self.generate_graph(
            module, inputs, use_dynamo_tracer=False, enable_passes=False
        )
        interpreter = TRTInterpreter(
            mod,
            [Input.from_tensor(i) for i in inputs],
            compilation_settings=CompilationSettings(**settings),
        )
        interpreter.run()
        return interpreter.ctx.net

    @parameterized.expand(
        [
            ("immutable", {"immutable_weights": True}, torch.float32, True),
            (
                "refit_identical",
                {"immutable_weights": False, "refit_identical_engine_weights": True},
                torch.float32,
                True,
            ),
            (
                "refittable_no_affine",
                {"immutable_weights": False},
                torch.float16,
                False,
            ),
        ]
    )
    def test_native_batchnorm_legit_no_training_fp16_input_adds_no_casts(
        self, _, settings, params_dtype, affine
    ):
        # The scale layer stays in float16, so a preceding convolution can fuse with it.
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    (
                        torch.tensor([0.5, 1.0, 2.0], dtype=params_dtype)
                        if affine
                        else None
                    ),
                    (
                        torch.tensor([-0.5, 0.0, 0.5], dtype=params_dtype)
                        if affine
                        else None
                    ),
                    torch.tensor([0.1, -0.2, 0.3], dtype=params_dtype),
                    torch.tensor([0.5, 1.0, 1.5], dtype=params_dtype),
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(2, FEATURE_NUM, 8, 8, dtype=torch.float16)]
        net = self._network(BatchNorm(), inputs, **settings)
        layer_types = [net.get_layer(i).type for i in range(net.num_layers)]
        self.assertIn(trt.LayerType.SCALE, layer_types)
        self.assertNotIn(trt.LayerType.CAST, layer_types)

    @parameterized.expand(
        [
            ("no_training_refittable", False, True, False, trt.float32),
            ("no_stats_refittable", True, True, False, trt.float32),
            ("no_stats_immutable", True, True, True, trt.float16),
            ("no_stats_refittable_no_affine", True, False, False, trt.float16),
        ]
    )
    def test_native_batchnorm_fp16_input_compute_dtype(
        self, _, no_stats, affine, immutable_weights, compute_dtype
    ):
        # Refit may bring weights that do not fit float16, so a refittable engine with
        # float32 weights normalizes in float32 and casts the output back.
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                weight = torch.tensor([0.5, 1.0, 2.0]) if affine else None
                bias = torch.tensor([-0.5, 0.0, 0.5]) if affine else None
                if no_stats:
                    return torch.ops.aten._native_batch_norm_legit.no_stats(
                        x, weight, bias, True, 0.1, 1e-05
                    )[0]
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    weight,
                    bias,
                    torch.tensor([0.1, -0.2, 0.3]),
                    torch.tensor([0.5, 1.0, 1.5]),
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(2, FEATURE_NUM, 8, 8, dtype=torch.float16)]
        net = self._network(BatchNorm(), inputs, immutable_weights=immutable_weights)
        layer_type = trt.LayerType.NORMALIZATION if no_stats else trt.LayerType.SCALE
        layers = [net.get_layer(i) for i in range(net.num_layers)]
        norm_layer = next(layer for layer in layers if layer.type == layer_type)
        self.assertEqual(norm_layer.get_input(0).dtype, compute_dtype)
        self.assertEqual(net.get_output(0).dtype, trt.float16)

    def test_native_batchnorm_legit_no_stats_fp16_input_weight_out_of_range(self):
        # A float32 weight above the float16 range on a constant channel, whose
        # normalized values are 0, so the output fits float16.
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit.no_stats(
                    x,
                    torch.tensor([1e5, 1.0, 2.0]),
                    torch.tensor([0.5, 0.0, 0.5]),
                    True,
                    0.1,
                    1e-05,
                )[0]

        x = torch.randn(2, FEATURE_NUM, 8, 8)
        x[:, 0] = 3.0
        inputs = [x.half()]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    @parameterized.expand(
        [
            # A large weight over a small variance folds to a scale above the float16
            # range, while the output itself fits float16.
            ("scale", [1000.0, 1.0, 2.0], [0.0, -0.2, 0.3], (2, FEATURE_NUM)),
            # The scale fits float16, but scale * running_mean does not.
            ("shift", [10.0, 1.0, 2.0], [100.0, -0.2, 0.3], (2, FEATURE_NUM, 8)),
        ]
    )
    def test_native_batchnorm_legit_no_training_fp16_input_folded_out_of_range(
        self, _, weight, mean, shape
    ):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    torch.tensor(weight),
                    torch.tensor([0.0, 0.0, 0.5]),
                    torch.tensor(mean),
                    torch.tensor([1e-4, 1.0, 1.5]),
                    0.1,
                    1e-05,
                )[0]

        running_mean = torch.tensor(mean).view(1, FEATURE_NUM, *[1] * (len(shape) - 2))
        inputs = [(running_mean + 0.1 * torch.randn(shape)).half()]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_native_batchnorm_legit_no_training_fp16_weight_folded_out_of_range(self):
        # All parameters are float16, and a float16 fold would overflow.
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit_no_training.default(
                    x,
                    torch.tensor([1000.0, 1.0, 2.0], dtype=torch.float16),
                    torch.zeros(FEATURE_NUM, dtype=torch.float16),
                    torch.zeros(FEATURE_NUM, dtype=torch.float16),
                    torch.tensor([1e-4, 1.0, 1.5], dtype=torch.float16),
                    0.1,
                    1e-05,
                )[0]

        inputs = [(1e-3 * torch.randn(2, FEATURE_NUM, 8, 8)).half()]
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

    @parameterized.expand(
        [
            ("1d", (4, FEATURE_NUM), torch.float, True),
            ("2d", (2, FEATURE_NUM, 8, 8), torch.float, True),
            ("3d", (2, FEATURE_NUM, 4, 8, 8), torch.float, True),
            ("2d_no_affine", (2, FEATURE_NUM, 8, 8), torch.float, False),
            ("2d_fp16", (2, FEATURE_NUM, 8, 8), torch.half, True),
        ]
    )
    def test_native_batchnorm_legit_no_stats(self, _, shape, dtype, affine):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit.no_stats(
                    x,
                    torch.tensor([2.0, 1.0, 0.5]) if affine else None,
                    torch.tensor([0.5, -1.0, 0.0]) if affine else None,
                    True,
                    0.1,
                    1e-05,
                )[0]

        inputs = [(torch.randn(shape) * 3 + 2).to(dtype)]
        self.run_test(
            BatchNorm(),
            inputs,
        )

    def test_native_batchnorm_legit_no_stats_rank3_with_dynamic_shape(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit.no_stats(
                    x,
                    torch.tensor([2.0, 1.0, 0.5]),
                    torch.tensor([0.5, -1.0, 0.0]),
                    True,
                    0.1,
                    1e-03,
                )[0]

        input_specs = [
            Input(
                shape=(-1, 3, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 2), (2, 3, 8), (4, 3, 16))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            BatchNorm(),
            input_specs,
        )

    def test_native_batchnorm_training_uses_batch_statistics(self):
        class BatchNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.native_batch_norm.default(
                    x,
                    torch.tensor([2.0, 1.0, 0.5]),
                    torch.tensor([0.5, -1.0, 0.0]),
                    torch.zeros(FEATURE_NUM),
                    torch.ones(FEATURE_NUM),
                    True,
                    0.1,
                    1e-05,
                )[0]

        inputs = [torch.randn(2, 3, 8, 8) * 3 + 2]
        self.run_test(
            BatchNorm(),
            inputs,
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
