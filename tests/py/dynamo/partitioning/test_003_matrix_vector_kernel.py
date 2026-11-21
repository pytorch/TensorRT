# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl


def _kernel_spy():
    return mock.patch.object(
        impl.linear, "matrix_vector_kernel", wraps=impl.linear.matrix_vector_kernel
    )


def _compile(model, x):
    with torch.no_grad(), _kernel_spy() as spy:
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            inputs=[x],
            use_explicit_typing=True,
            min_block_size=1,
            matrix_vector_kernel=True,
        )
        return compiled(x), spy.call_count


class TestMatrixVectorKernel(TestCase):
    def setUp(self):
        reason = aten_ops_converters._grouped_mm_kernel_unavailable()
        if reason is not None:
            self.skipTest(reason)

    @parameterized.expand(
        [
            # name, rows, K, N, bias, dtype
            ("one_row_split_k", 1, 2304, 3072, False, torch.bfloat16),
            ("one_row_narrow", 1, 2304, 24, False, torch.bfloat16),
            ("one_row_no_split", 1, 2304, 8608, False, torch.bfloat16),
            ("rows_bias_fp16", 5, 1024, 1000, True, torch.float16),
            ("sixteen_rows_small_k", 16, 48, 64, False, torch.bfloat16),
        ]
    )
    def test_kernel_matches_pytorch(self, _, rows, k, n, bias, dtype):
        torch.manual_seed(0)
        model = torch.nn.Linear(k, n, bias=bias).to("cuda", dtype).eval()
        x = torch.randn(1, rows, k, dtype=dtype, device="cuda")
        out, calls = _compile(model, x)
        self.assertEqual(calls, 1)
        expected = torch.nn.functional.linear(
            x.float(),
            model.weight.float(),
            None if model.bias is None else model.bias.float(),
        )
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    @parameterized.expand(
        [
            ("seventeen_rows", 17, 256),
            ("k_not_multiple_of_16", 1, 100),
        ]
    )
    def test_declines(self, _, rows, k):
        model = torch.nn.Linear(k, 64, bias=False).to("cuda", torch.bfloat16).eval()
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        out, calls = _compile(model, x)
        self.assertEqual(calls, 0)
        torch.testing.assert_close(out, model(x), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    run_tests()
