# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestAddmmConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 2), (2, 3), (3, 2)),
            ((4, 6), (4, 5), (5, 6)),
            ((2, 1), (2, 3), (3, 1)),
            ((4, 1), (4, 1), (1, 1)),
            ((1, 2), (1, 3), (3, 2)),
        ]
    )
    def test_addmm(self, input_shape, mat1_shape, mat2_shape):
        skip_if_trt_rtx_turing(self, "aten.addmm (an FP32 GEMM)")

        class Addmm(nn.Module):
            def forward(self, input, mat1, mat2):
                return torch.ops.aten.addmm.default(input, mat1, mat2)

        inputs = [
            torch.randn(input_shape),
            torch.randn(mat1_shape),
            torch.randn(mat2_shape),
        ]

        self.run_test(
            Addmm(),
            inputs,
        )

    @parameterized.expand(
        [
            ((2, 2), (2, 3), (3, 2), 1.0, 1.0),
            ((4, 6), (4, 5), (5, 6), 1.2, 0.8),
            ((2, 1), (2, 3), (3, 1), 3, 2),
            ((4, 1), (4, 1), (1, 1), 1, 1),
            ((1, 2), (1, 3), (3, 2), 2, 1.0),
            ((1, 2), (1, 3), (3, 2), 1, 2.0),
        ]
    )
    def test_addmm_scale(self, input_shape, mat1_shape, mat2_shape, beta, alpha):
        skip_if_trt_rtx_turing(self, "aten.addmm (an FP32 GEMM)")

        class Addmm(nn.Module):
            def forward(self, input, mat1, mat2):
                return torch.ops.aten.addmm.default(
                    input, mat1, mat2, beta=beta, alpha=alpha
                )

        inputs = [
            torch.randn(input_shape),
            torch.randn(mat1_shape),
            torch.randn(mat2_shape),
        ]

        self.run_test(
            Addmm(),
            inputs,
        )

    def test_addmm_bias_then_qkv_heads(self):
        # GPT-2's Conv1D QKV projection split into heads: with the bias broadcast to
        # the matmul's full shape, TensorRT failed to build the fused projection.
        class QKV(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(64, 192) * 0.1)
                self.b = nn.Parameter(torch.randn(192) * 0.1)

            def forward(self, x):
                y = torch.ops.aten.addmm.default(self.b, x.view(-1, 64), self.w)
                return tuple(
                    t.view(1, 7, 2, 32).permute(0, 2, 1, 3)
                    for t in y.view(1, 7, 192).split(64, dim=2)
                )

        self.run_test(QKV(), [torch.randn(1, 7, 64)])


if __name__ == "__main__":
    run_tests()
