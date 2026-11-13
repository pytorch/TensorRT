# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._defaults import GROUPED_MM_DENSE_MAX_MACS


class ExpertsBetweenReLUs(torch.nn.Module):
    def __init__(self, experts, k, n):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(experts, k, n) * 0.05)

    def forward(self, x, offs):
        h = torch.ops.aten._grouped_mm.default(torch.relu(x), self.w, offs)
        return torch.relu(h)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGroupedMMEngineOnlyExport(TestCase):
    def test_engine_only_export_keeps_large_grouped_mm(self):
        # Few groups keep the engine small.
        experts, rows, k = 2, 4097, 5800
        self.assertGreater(experts * rows * k * k, GROUPED_MM_DENSE_MAX_MACS)
        model = ExpertsBetweenReLUs(experts, k, k).half().eval().cuda()
        offs = torch.linspace(rows // experts, rows, experts).to(torch.int32).cuda()
        x = torch.randn(rows, k, dtype=torch.float16).cuda()
        ep = torch.export.export(model, (x, offs))
        engine = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                ep, arg_inputs=[x, offs], min_block_size=1
            )
        )
        self.assertGreater(len(engine), 0)


if __name__ == "__main__":
    run_tests()
