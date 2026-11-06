# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    adaptive_max_pool_validator,
)

from .harness import DispatchTestCase


class TestAdaptiveMaxPoolConverter(DispatchTestCase):
    def test_adaptive_max_pool1d(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool1d.default(x, (5,))[0]

        inputs = [torch.randn(2, 3, 9)]
        self.run_test(TestModule(), inputs, enable_passes=True)

    @parameterized.expand(
        [
            ("not_divisible_3d_input", (3, 9, 7), (7, 3), torch.float),
            ("divisible", (2, 3, 14, 12), (7, 4), torch.float),
            ("not_divisible", (2, 3, 9, 10), (7, 4), torch.float),
            ("one_axis_unchanged", (2, 3, 10, 6), (4, 6), torch.float),
            ("not_divisible_fp16", (2, 3, 9, 10), (7, 4), torch.half),
            ("one_more_than_divisible", (2, 3, 15, 8), (7, 7), torch.float),
            ("long_not_divisible", (1, 1, 1, 300002), (1, 3), torch.float),
            ("window_over_pooling_limit", (1, 1, 1, 100000), (1, 1), torch.float),
            ("grow_one_axis_shrink_other", (1, 1, 1, 65536), (65536, 1), torch.float),
        ]
    )
    def test_adaptive_max_pool2d(self, _, input_shape, output_size, dtype):
        if torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx and any(
            out > size for out, size in zip(output_size, input_shape[-2:])
        ):
            self.skipTest(
                "An output axis larger than its input falls back to PyTorch on TensorRT-RTX"
            )

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, output_size)[0]

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(TestModule(), inputs, enable_passes=True)

    @parameterized.expand(
        [
            ("divisible_4d_input", (2, 4, 4, 4), (2, 2, 2), torch.float),
            ("not_divisible", (1, 2, 5, 6, 7), (3, 4, 3), torch.float),
            ("stride_over_pooling_limit", (1, 1, 200002, 1, 1), (2, 1, 1), torch.float),
        ]
    )
    def test_adaptive_max_pool3d(self, _, input_shape, output_size, dtype):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool3d.default(x, output_size)[0]

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(TestModule(), inputs, enable_passes=True)

    def test_dynamic_shape_adaptive_max_pool2d(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (4, 5))[0]

        input_specs = [
            Input(
                min_shape=(1, 1, 9, 9),
                opt_shape=(2, 3, 9, 9),
                max_shape=(3, 4, 9, 9),
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_dynamic_spatial_shape_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (3, 3))[0]

        input_specs = [
            Input(
                min_shape=(1, 2, 7, 7),
                opt_shape=(1, 2, 8, 8),
                max_shape=(1, 2, 9, 9),
                dtype=torch.float,
            ),
        ]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_empty_output_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (0, 3))[0]

        inputs = [torch.randn(1, 2, 7, 7)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, enable_passes=True)

    def test_float64_input_needs_truncate_double(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (3, 3))[0]

        exported = torch.export.export(
            TestModule(), (torch.randn(1, 2, 7, 7).double(),)
        )
        # Export keeps the unused indices getitem, which lowering removes.
        exported.graph.eliminate_dead_code()
        (node,) = [
            n
            for n in exported.graph.nodes
            if n.target == torch.ops.aten.adaptive_max_pool2d.default
        ]
        for truncate_double in (False, True):
            settings = CompilationSettings(truncate_double=truncate_double)
            self.assertEqual(
                adaptive_max_pool_validator(node, settings), truncate_double
            )

    def test_used_indices_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (3, 3))

        inputs = [torch.randn(1, 2, 7, 7)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, enable_passes=True)


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestAdaptiveMaxPoolEndToEnd(TestCase):
    def test_one_engine(self):
        model = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((7, 3)),
            torch.nn.Sigmoid(),
        )
        inputs = [torch.randn(2, 3, 16, 16).cuda()]
        trt_mod = torch_tensorrt.compile(
            model, ir="dynamo", inputs=inputs, min_block_size=1
        )
        engine_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(engine_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), model(*inputs))


if __name__ == "__main__":
    run_tests()
