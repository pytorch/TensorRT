# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


class TestAdaptiveMaxPoolConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("divisible", (2, 8), (4,)),
            ("not_divisible", (2, 3, 9), (5,)),
        ]
    )
    def test_adaptive_max_pool1d(self, _, input_shape, output_size):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool1d.default(x, output_size)[0]

        inputs = [torch.randn(input_shape)]
        self.run_test(TestModule(), inputs, enable_passes=True)

    @parameterized.expand(
        [
            ("divisible_3d_input", (3, 8, 8), (4, 4), torch.float),
            ("not_divisible_3d_input", (3, 9, 7), (7, 3), torch.float),
            ("divisible", (2, 3, 14, 12), (7, 4), torch.float),
            ("global", (2, 3, 16, 16), (1, 1), torch.float),
            ("not_divisible", (2, 3, 9, 10), (7, 4), torch.float),
            ("mixed", (2, 3, 16, 9), (4, 7), torch.float),
            ("one_axis_unchanged", (2, 3, 10, 6), (4, 6), torch.float),
            ("unchanged", (2, 3, 7, 7), (7, 7), torch.float),
            ("larger_than_input", (1, 2, 3, 4), (5, 7), torch.float),
            ("not_divisible_fp16", (2, 3, 9, 10), (7, 4), torch.half),
            ("one_more_than_divisible", (2, 3, 15, 8), (7, 7), torch.float),
            ("long_not_divisible", (1, 1, 1, 300002), (1, 3), torch.float),
            ("window_over_pooling_limit", (1, 1, 1, 100000), (1, 1), torch.float),
        ]
    )
    def test_adaptive_max_pool2d(self, _, input_shape, output_size, dtype):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, output_size)[0]

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(TestModule(), inputs, enable_passes=True)

    @parameterized.expand(
        [
            ("divisible_4d_input", (2, 4, 4, 4), (2, 2, 2), torch.float),
            ("not_divisible", (1, 2, 5, 6, 7), (3, 4, 3), torch.float),
            ("not_divisible_fp16", (1, 2, 5, 6, 7), (3, 4, 3), torch.half),
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

    def test_used_indices_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_max_pool2d.default(x, (3, 3))

        inputs = [torch.randn(1, 2, 7, 7)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, enable_passes=True)


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestAdaptiveMaxPoolEndToEnd(TestCase):
    @parameterized.expand(
        [
            ("global", (1, 1)),
            ("not_divisible", (7, 3)),
        ]
    )
    def test_one_engine(self, _, output_size):
        model = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d(output_size),
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
