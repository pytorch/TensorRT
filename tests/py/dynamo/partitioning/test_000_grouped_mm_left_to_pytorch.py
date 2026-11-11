# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests


class ExpertsBetweenReLUs(torch.nn.Module):
    def __init__(self, experts, k, n):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(experts, k, n) * 0.05)

    def forward(self, x, offs):
        h = torch.ops.aten._grouped_mm.default(torch.relu(x), self.w, offs)
        return torch.relu(h)


def _inputs(experts, rows, k):
    counts = torch.zeros(experts, dtype=torch.int64)
    counts[::3] = rows // len(counts[::3])
    counts[0] += rows - counts.sum()
    offs = torch.cumsum(counts, 0).to(torch.int32).cuda()
    return torch.randn(rows, k, dtype=torch.float16).cuda(), offs


def _grouped_mm_in_pytorch(compiled):
    return [
        node
        for name, sub in compiled.named_children()
        if name.startswith("_run_on_gpu")
        for node in sub.graph.nodes
        if node.target == torch.ops.aten._grouped_mm.default
    ]


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGroupedMMLeftToPyTorch(TestCase):
    # 8 * 256 * 64 * 64 multiply-adds, above this limit.
    small_limit = 2**20

    def setUp(self):
        self.model = ExpertsBetweenReLUs(8, 64, 64).half().eval().cuda()
        self.x, self.offs = _inputs(8, 256, 64)

    def test_grouped_mm_over_limit_runs_in_pytorch(self):
        compiled = torch_tensorrt.compile(
            self.model,
            ir="dynamo",
            inputs=[self.x, self.offs],
            min_block_size=1,
            grouped_mm_dense_max_macs=self.small_limit,
        )
        self.assertEqual(len(_grouped_mm_in_pytorch(compiled)), 1)
        torch.testing.assert_close(
            compiled(self.x, self.offs),
            self.model(self.x, self.offs),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_each_compile_warns_once_per_node(self):
        # The registry runs the validator more than once per node, and a second compile
        # in the same process leaves the op to PyTorch again.
        for _ in range(2):
            with self.assertLogs("torch_tensorrt", level="WARNING") as logs:
                torch_tensorrt.compile(
                    self.model,
                    ir="dynamo",
                    inputs=[self.x, self.offs],
                    min_block_size=1,
                    grouped_mm_dense_max_macs=self.small_limit,
                )
            warnings = [
                record
                for record in logs.records
                if "grouped_mm_dense_max_macs" in record.getMessage()
            ]
            self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    run_tests()
