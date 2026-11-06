# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.export import Dim
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import post_lowering
from torch_tensorrt.dynamo.lowering.passes.fold_slice_into_constant_pad import (
    fold_slice_into_constant_pad,
)

PAD = torch.ops.aten.constant_pad_nd.default
SLICE = torch.ops.aten.slice.Tensor


class PadThenSlice(torch.nn.Module):
    def __init__(self, pad, dim, start, end, step=1, mode="constant", keep_pad=False):
        super().__init__()
        self.pad, self.dim, self.start, self.end, self.step = pad, dim, start, end, step
        self.mode, self.keep_pad = mode, keep_pad

    def forward(self, x):
        padded = F.pad(x, self.pad, mode=self.mode)
        index = [slice(None)] * x.dim()
        index[self.dim] = slice(self.start, self.end, self.step)
        out = padded[tuple(index)]
        return (out, padded) if self.keep_pad else out


def _targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _lower(module, x, dynamic_shapes=None):
    ep = torch.export.export(module, (x,), dynamic_shapes=dynamic_shapes)
    gm = ep.run_decompositions().graph_module
    before = _targets(gm)
    gm = fold_slice_into_constant_pad(gm, CompilationSettings())
    return gm, before, _targets(gm)


class TestFoldSliceIntoConstantPad(unittest.TestCase):
    @parameterized.expand(
        [
            ("shift_right", (3, 0), -1, None, 8),
            ("shift_left", (0, 3), -1, 3, None),
            ("crop_both_ends", (2, 2), -1, 1, -1),
            ("keep_part_of_padding", (3, 2), -1, 1, 10),
            ("second_to_last_dim", (0, 0, 2, 1), 2, 1, 7),
        ]
    )
    def test_folds(self, _, pad, dim, start, end):
        module = PadThenSlice(pad, dim, start, end)
        x = torch.randn(2, 4, 6, 8)
        gm, _, targets = _lower(module, x)
        self.assertNotIn(SLICE, targets)
        self.assertEqual(targets.count(PAD), 1)
        torch.testing.assert_close(gm(x)[0], module(x), rtol=0, atol=0)

    def test_runs_in_post_lowering(self):
        ep = torch.export.export(
            PadThenSlice((3, 0), -1, None, 8), (torch.randn(2, 8),)
        )
        gm = post_lowering(ep.run_decompositions().graph_module, CompilationSettings())
        self.assertNotIn(SLICE, _targets(gm))

    @parameterized.expand(
        [
            # The slice is on a dim that the pad does not pad.
            ("other_dim", PadThenSlice((2, 0), 1, 1, 3)),
            ("dim_padded_by_zero", PadThenSlice((2, 0, 0, 0), 2, 1, 5)),
            ("strided", PadThenSlice((3, 0), -1, 0, 8, step=2)),
            ("empty", PadThenSlice((3, 0), -1, 5, 5)),
            # The slice keeps none of the input, which a negative amount cannot express.
            ("inside_the_padding", PadThenSlice((6, 0), -1, 0, 2)),
            ("shift_past_the_width", PadThenSlice((10, 0), -1, None, 8)),
            ("pad_has_other_users", PadThenSlice((3, 0), -1, None, 8, keep_pad=True)),
            ("reflect", PadThenSlice((3, 0, 0, 0), -1, None, 8, mode="reflect")),
            ("replicate", PadThenSlice((3, 0, 0, 0), -1, None, 8, mode="replicate")),
        ]
    )
    def test_leaves_other_patterns_alone(self, _, module):
        x = torch.randn(2, 4, 6, 8)
        gm, before, after = _lower(module, x)
        self.assertIn(SLICE, before)
        self.assertEqual(before, after)

    def test_leaves_symbolic_amounts_alone(self):
        class SymbolicPad(torch.nn.Module):
            def forward(self, x, y):
                return F.pad(x, (y.shape[0], 0))[..., :8]

        x, y = torch.randn(2, 8), torch.randn(3)
        ep = torch.export.export(
            SymbolicPad(), (x, y), dynamic_shapes=(None, {0: Dim("n", min=2)})
        )
        gm = ep.run_decompositions().graph_module
        before = _targets(gm)
        gm = fold_slice_into_constant_pad(gm, CompilationSettings())
        self.assertIn(SLICE, before)
        self.assertEqual(before, _targets(gm))

    def test_stacked_shifts_compile(self):
        # Shifted copies of one tensor, stacked, as in a stereo cost volume.
        class Shifts(torch.nn.Module):
            def forward(self, x):
                width = x.shape[-1]
                return torch.stack(
                    [F.pad(x, (d, 0))[..., :width] for d in range(4)], dim=2
                )

        x = torch.randn(1, 3, 5, 8).cuda()
        module = Shifts().eval().cuda()
        trt_module = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (x,)), arg_inputs=[x], min_block_size=1
        )
        torch.testing.assert_close(trt_module(x), module(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
