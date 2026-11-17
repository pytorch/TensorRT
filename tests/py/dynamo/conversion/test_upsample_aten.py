# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    antialiased_upsample_validator,
)

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestUpsampleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ([7], [3], None),
            ([7], None, [1.5]),
        ]
    )
    def test_nearest1d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], None),
            (None, [1.5]),
        ]
    )
    def test_nearest1d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(5, 5, 5),
                max_shape=(9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], None),
            ([7, 7], None, [0.5, 1.5]),
        ]
    )
    def test_nearest2d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], None),
            (None, [0.5, 1.5]),
        ]
    )
    def test_nearest2d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7, 7], [3, 3, 3], None),
            ([7, 7, 7], None, [0.5, 1.0, 1.5]),
        ]
    )
    def test_nearest3d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], None),
            (None, [0.5, 1.0, 1.5]),
        ]
    )
    def test_nearest3d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5, 5),
                max_shape=(9, 9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7], [3], True, None),
            ([7], [3], False, None),
            ([7], None, True, [1.5]),
            ([7], None, False, [1.5]),
        ]
    )
    def test_linear1d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], True, None),
            ([3], False, None),
            (None, True, [1.5]),
            (None, False, [1.5]),
        ]
    )
    def test_linear1d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(5, 5, 5),
                max_shape=(9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], True, None),
            ([7, 7], [3, 3], False, None),
            ([7, 7], None, True, [0.5, 1.5]),
            ([7, 7], None, False, [0.5, 1.5]),
        ]
    )
    def test_bilinear2d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None),
            ([3, 3], False, None),
            (None, True, [0.5, 1.5]),
            (None, False, [0.5, 1.5]),
        ]
    )
    def test_bilinear2d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7, 7], [3, 3, 3], True, None),
            ([7, 7, 7], [3, 3, 3], False, None),
            ([7, 7, 7], None, True, [0.5, 1.0, 1.5]),
            ([7, 7, 7], None, False, [0.5, 1.0, 1.5]),
        ]
    )
    def test_trilinear3d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], True, None),
            ([3, 3, 3], False, None),
            (None, True, [0.5, 1.0, 1.5]),
            (None, False, [0.5, 1.0, 1.5]),
        ]
    )
    def test_trilinear3d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5, 5),
                max_shape=(9, 9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], True, None),
            ([7, 7], [3, 3], False, None),
            ([7, 7], None, True, [0.5, 1.5]),
            ([7, 7], None, False, [0.5, 1.5]),
        ]
    )
    def test_bicubic2d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None),
            ([3, 3], False, None),
            (None, True, [0.5, 1.5]),
            (None, False, [0.5, 1.5]),
        ]
    )
    def test_bicubic2d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([torch.tensor(3), 3], None),
            (None, [torch.tensor(0.5), 1.5]),
        ]
    )
    def test_nearest2d_mixed_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                out_size = output_size
                scale = scale_factors

                return torch.ops.aten.upsample_nearest2d.vec(x, out_size, scale)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            # Mix of Tensor and int in output_size
            ([torch.tensor(3), 3], None),
            # Mix of Tensor and float in scale_factors
            (None, [torch.tensor(0.5), 1.5]),
        ]
    )
    def test_nearest2d_mixed_static_input(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                out_size = output_size
                scale = scale_factors
                return torch.ops.aten.upsample_nearest2d.vec(x, out_size, scale)

        input_size = [7, 7]  # H, W
        inputs = [torch.randn([1, 1] + input_size)]  # shape [1, 1, 7, 7]

        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            (f"{mode}_{name}", op, *case)
            for mode, op in (
                ("bilinear", torch.ops.aten._upsample_bilinear2d_aa.default),
                ("bicubic", torch.ops.aten._upsample_bicubic2d_aa.default),
            )
            for name, *case in (
                # name, input shape, output size, align_corners, scales, dtype
                ("2x_3x", (2, 3, 32, 48), [16, 16], False, (None, None), torch.float32),
                # align_corners ignores the scales.
                (
                    "align_corners",
                    (2, 3, 32, 48),
                    [16, 1],
                    True,
                    (0.5, 0.5),
                    torch.float32,
                ),
                (
                    "scales",
                    (2, 3, 30, 40),
                    [13, 18],
                    False,
                    (0.45, 0.47),
                    torch.float32,
                ),
                ("upscale", (1, 2, 7, 9), [12, 9], False, (None, None), torch.float32),
                (
                    "same_size_scale",
                    (1, 2, 10, 12),
                    [10, 6],
                    False,
                    (1.04, 0.5),
                    torch.float32,
                ),
                (
                    "float16",
                    (1, 3, 480, 640),
                    [448, 448],
                    False,
                    (None, None),
                    torch.float16,
                ),
                (
                    "bfloat16",
                    (2, 3, 64, 64),
                    [24, 40],
                    False,
                    (None, None),
                    torch.bfloat16,
                ),
            )
        ]
    )
    def test_antialias2d(
        self, _, op, input_shape, output_size, align_corners, scales, dtype
    ):
        if dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return op(x, output_size, align_corners, *scales)

        tolerance = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}
        self.run_test(
            TestModule(),
            [torch.randn(input_shape, dtype=dtype)],
            rtol=tolerance[dtype],
            atol=tolerance[dtype],
            use_dynamo_tracer=True,
            convert_antialiased_resize=True,
        )

    def test_antialias2d_non_finite_pixels(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bicubic2d_aa.default(
                    x, [16, 16], False, None, None
                )

        x = torch.randn(1, 1, 64, 64)
        x[0, 0, 20, 30] = float("inf")
        x[0, 0, 63, 63] = float("inf")
        x[0, 0, 40, 10] = float("nan")
        self.run_test(
            TestModule(),
            [x],
            rtol=1e-4,
            atol=1e-4,
            use_dynamo_tracer=True,
            convert_antialiased_resize=True,
        )

    @parameterized.expand(
        [
            ("uint8", torch.uint8, True, False),
            ("float64", torch.float64, False, False),
            ("float64_truncated", torch.float64, True, True),
        ]
    )
    def test_antialias2d_validator_dtypes(self, _, dtype, truncate_double, expected):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bilinear2d_aa.default(
                    x, [4, 4], False, None, None
                )

        exported = torch.export.export(
            TestModule(), (torch.zeros(1, 3, 8, 8, dtype=dtype),)
        )
        (node,) = (
            n
            for n in exported.graph.nodes
            if n.target == torch.ops.aten._upsample_bilinear2d_aa.default
        )
        settings = CompilationSettings(
            convert_antialiased_resize=True, truncate_double=truncate_double
        )
        self.assertEqual(antialiased_upsample_validator(node, settings), expected)

    def test_antialias2d_dynamic_batch_and_channels(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bilinear2d_aa.default(
                    x, [16, 16], False, None, None
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 32, 48),
                opt_shape=(2, 3, 32, 48),
                max_shape=(4, 6, 32, 48),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            convert_antialiased_resize=True,
        )

    @parameterized.expand(
        [
            ("static", None, True, 1, 0),
            ("off_by_default", None, False, 2, 1),
            (
                "dynamic_height",
                ({2: torch.export.Dim("h", min=32, max=96)}, None),
                True,
                2,
                1,
            ),
            (
                "dynamic_size",
                (None, {0: torch.export.Dim("size", min=8, max=32)}),
                True,
                2,
                1,
            ),
        ]
    )
    def test_antialias2d_engines(
        self, _, dynamic_shapes, convert, acc_count, gpu_count
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x, size):
                x = F.interpolate(
                    x.relu(), size=size.shape, mode="bicubic", antialias=True
                )
                return x + 1

        module = TestModule().eval().cuda()
        inputs = (
            torch.randn(2, 3, 64, 64, device="cuda"),
            torch.empty(24, 40, device="cuda"),
        )
        exported = torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes)
        options = {"convert_antialiased_resize": True} if convert else {}
        compiled = torch_tensorrt.dynamo.compile(
            exported, inputs=inputs, min_block_size=1, **options
        )
        names = [name for name, _ in compiled.named_children()]
        self.assertEqual(sum("_run_on_acc" in name for name in names), acc_count)
        self.assertEqual(sum("_run_on_gpu" in name for name in names), gpu_count)
        torch.testing.assert_close(
            compiled(*inputs), module(*inputs), rtol=5e-3, atol=5e-3
        )


if __name__ == "__main__":
    run_tests()
