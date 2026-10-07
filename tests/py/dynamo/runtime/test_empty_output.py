# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests


class _EmptyOutput(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.slice.Tensor(x, 0, 0, 0, 1)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestEmptyOutput(TestCase):
    def setUp(self):
        super().setUp()
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        torch._dynamo.reset()
        super().tearDown()

    @parameterized.expand(
        [
            ("direct", False),
            ("cudagraph", True),
        ]
    )
    def test_empty_output_has_valid_binding(self, _, use_cudagraphs):
        model = _EmptyOutput().eval().cuda()
        x = torch.randn(8, 4, device="cuda")
        exported_program = torch.export.export(model, (x,))
        compiled = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[x],
            min_block_size=1,
            require_full_compilation=True,
        )

        if use_cudagraphs:
            with torch_tensorrt.runtime.enable_cudagraphs(compiled) as runtime:
                output = runtime(x)
                replay_output = runtime(x)
                self.assertEqual(replay_output.shape, (0, 4))
                self.assertEqual(replay_output.numel(), 0)
        else:
            output = compiled(x)

        self.assertEqual(output.shape, (0, 4))
        self.assertEqual(output.numel(), 0)


if __name__ == "__main__":
    run_tests()
