# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
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
from torch_tensorrt.dynamo.conversion.impl.upsample import _antialias_taps

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

    @parameterized.expand(
        [
            ("bilinear", torch.ops.aten._upsample_bilinear2d_aa.default),
            ("bicubic", torch.ops.aten._upsample_bicubic2d_aa.default),
        ]
    )
    def test_antialias2d_matches_pytorch(self, _, op):
        # Images are often rounded to uint8 after the resize, and at this 16/13 downscale
        # many exact results end in .5, so this must stay far below one level. 1e-4,
        # far below main's 4e-3, covers both modes when TensorRT sums in another order
        # or without fused multiply-adds: the measured worst case was 6.1e-5 (bicubic).
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return op(x, [234, 416], False, None, None)

        self.run_test(
            TestModule(),
            [torch.randint(0, 256, (2, 3, 288, 512)).float()],
            rtol=0,
            atol=1e-4,
            use_dynamo_tracer=True,
            convert_antialiased_resize=True,
        )

    def test_antialias2d_float64_compile(self):
        # Partitioning hands the engine a float32 copy of a float64 input, but the
        # converter must still weigh it in float64, like ATen.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bilinear2d_aa.default(
                    x, [234, 416], False, None, None
                )

        module = TestModule()
        x = torch.randint(0, 256, (2, 3, 288, 512), device="cuda").double()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (x,)),
            inputs=(x,),
            min_block_size=1,
            truncate_double=True,
            convert_antialiased_resize=True,
        )
        self.assertEqual(
            [name for name, _ in compiled.named_children()], ["_run_on_acc_0"]
        )
        output = compiled(x)
        torch.testing.assert_close(output, module(x), rtol=0, atol=1e-4)

    @parameterized.expand(
        [
            (f"{mode}_{name}", mode, op, *case)
            for mode, op in (
                ("bilinear", torch.ops.aten._upsample_bilinear2d_aa.default),
                ("bicubic", torch.ops.aten._upsample_bicubic2d_aa.default),
            )
            for name, *case in (
                # name, input size, output size, align_corners, scale
                ("downscale_9x", 1920, 224, False, None),
                ("upscale", 7, 12, False, None),
                ("align_corners", 32, 16, True, None),
                ("scale", 40, 18, False, 0.47),
            )
        ]
    )
    def test_antialias_taps_match_pytorch(
        self, _, mode, op, input_size, output_size, align_corners, scale
    ):
        indices, weights, in_window = _antialias_taps(
            input_size, output_size, align_corners, scale, mode
        )
        dense = np.zeros((input_size, output_size), dtype=np.float32)
        outputs, taps = np.nonzero(in_window)
        dense[indices[outputs, taps], outputs] = weights[outputs, taps]
        # Resizing the rows of an identity to one row reads out PyTorch's weights
        # exactly: each output sums a single nonzero product.
        eye = torch.eye(input_size, device="cuda").view(input_size, 1, 1, input_size)
        expected = op(eye, [1, output_size], align_corners, None, scale)
        torch.testing.assert_close(
            torch.from_numpy(dense),
            expected.view(input_size, output_size).cpu(),
            rtol=0,
            atol=0,
        )

    def test_antialias2d_empty_windows(self):
        # Scales that disagree with the output size put the windows of outputs 6 to 19
        # past the input. PyTorch leaves those outputs undefined, so they must be 0 here,
        # and the others come from a resize to 6x6 with the same scales.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bilinear2d_aa.default(
                    x, [20, 20], False, 0.5, 0.5
                )

        module = TestModule()
        x = torch.randn(1, 2, 10, 10, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (x,)),
            inputs=(x,),
            min_block_size=1,
            convert_antialiased_resize=True,
        )
        self.assertEqual(
            [name for name, _ in compiled.named_children()], ["_run_on_acc_0"]
        )
        output = compiled(x)
        expected = torch.zeros_like(output)
        expected[..., :6, :6] = torch.ops.aten._upsample_bilinear2d_aa.default(
            x, [6, 6], False, 0.5, 0.5
        )
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_antialias2d_tensor_meta_only(self):
        # Without the dynamo tracer the input node carries tensor_meta but no val, and its
        # dtype must still select float64 weights.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._upsample_bilinear2d_aa.default(
                    x, [234, 416], False, None, None
                )

        self.run_test(
            TestModule(),
            [torch.randint(0, 256, (2, 3, 288, 512)).to(torch.float64)],
            rtol=0,
            atol=1e-4,
            check_dtype=False,
            use_dynamo_tracer=False,
            propagate_shapes=True,
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
