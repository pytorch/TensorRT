# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRoundConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_round_float(self, input_shape, dtype):
        class round(nn.Module):
            def forward(self, input):
                return torch.ops.aten.round.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            round(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), (11,), (12,)),
            ((1, 3, 4), (2, 3, 5), (3, 4, 6)),
            ((2, 3, 4, 5), (3, 5, 4, 5), (4, 6, 4, 5)),
        ]
    )
    def test_round_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class round(nn.Module):
            def forward(self, input):
                return torch.ops.aten.round.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            round(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_round_int(self, input_shape, dtype, low, high):
        class round(nn.Module):
            def forward(self, input):
                return torch.ops.aten.round.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            round(),
            inputs,
            check_dtype=False,
        )

    @parameterized.expand(
        [
            (torch.int8, [-56, 100]),
            (torch.int32, [2**25 + 1, -(2**25) - 3]),
            (torch.int64, [2**25 + 1, -(2**25) - 3]),
        ]
    )
    def test_round_int_is_exact(self, dtype, values):
        class round(nn.Module):
            def forward(self, input):
                return torch.ops.aten.round.default(input)

        # float32 would round the int32 and int64 values, and TensorRT treats an int8
        # engine input as quantized, so a rounding layer after a cast fails the build.
        inputs = [torch.tensor(values, dtype=dtype)]
        self.run_test(round(), inputs, rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
