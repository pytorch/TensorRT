# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests


class EmptyAndFullOutputs(nn.Module):
    """One output has no elements, the other is a normal tensor from the same engine."""

    def forward(self, x):
        return x[:, :0].relu(), x * 2 + 1


class TestEmptyEngineOutput(TestCase):
    def test_empty_output_does_not_break_other_outputs(self):
        """TensorRT rejects a null output address even for a tensor with no elements, and
        an empty tensor has a null address. The engine then refused to run, the failure
        was ignored, and the second output came back unwritten."""
        x = torch.randn(3, 4, device="cuda")
        model = EmptyAndFullOutputs().eval().cuda()

        compiled = torchtrt.dynamo.compile(
            torch.export.export(model, (x,)),
            arg_inputs=[x],
            min_block_size=1,
        )
        engines = [
            name for name, _ in compiled.named_children() if "_run_on_acc" in name
        ]
        self.assertEqual(engines, ["_run_on_acc_0"], "expected one engine")

        empty, full = compiled(x)
        ref_empty, ref_full = model(x)
        self.assertEqual(empty.shape, ref_empty.shape)
        torch.testing.assert_close(full, ref_full)


if __name__ == "__main__":
    run_tests()
