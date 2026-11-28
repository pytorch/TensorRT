# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.lowering.passes._FakeTensorUpdater import FakeTensorUpdater


class TestFakeTensorUpdater(TestCase):
    def test_fills_missing_fake_tensor_metadata(self):
        class AddRelu(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1)

        gm = torch.export.export(AddRelu(), (torch.randn(2, 3),)).graph_module
        add = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.add.Tensor)
        del add.meta["val"]

        FakeTensorUpdater(gm).incremental_update(_detect_fake_mode_from_gm(gm))

        self.assertEqual(add.meta["val"].shape, torch.Size([2, 3]))
        self.assertEqual(add.meta["val"].dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
