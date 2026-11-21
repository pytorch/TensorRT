# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion import (
    aten_ops_converters,
    experts_custom_op,
    impl,
)

E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _nvfp4(groups, n, k):
    weight = torch.randint(0, 256, (groups, n, k // 2), dtype=torch.uint8)
    scale = (torch.rand(groups, n, k // 16) * 4 + 0.25).to(torch.float8_e4m3fn)
    scale_2 = torch.rand(groups) * 0.01 + 0.001
    return weight, scale, scale_2


def _mxfp8(groups, n, k):
    weight = (torch.randn(groups, n, k) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 124, (groups, n, k // 32), dtype=torch.uint8)
    return weight, scale


def _nvfp4_reference(weight, scale, scale_2):
    """One value at a time, straight from the format's definition."""
    groups, n, half_k = weight.shape
    out = torch.empty(groups, n, half_k * 2)
    for i in range(half_k * 2):
        byte = weight[..., i // 2].long()
        code = byte & 0xF if i % 2 == 0 else byte >> 4
        value = torch.tensor(E2M1)[code]
        out[..., i] = value * scale[..., i // 16].float() * scale_2[:, None]
    return out


class Experts(torch.nn.Module):
    """Packed expert weights dequantized before each grouped matmul, around an activation."""

    def __init__(self, fmt, groups, k, n):
        super().__init__()
        self.fmt = fmt
        for name, (rows, cols) in (("up", (n, k)), ("down", (k, n))):
            tensors = (
                _nvfp4(groups, rows, cols)
                if fmt == "nvfp4"
                else _mxfp8(groups, rows, cols)
            )
            for i, tensor in enumerate(tensors):
                self.register_buffer(f"{name}_{i}", tensor)

    def weight(self, name):
        tensors = [
            getattr(self, f"{name}_{i}") for i in range(3 if self.fmt == "nvfp4" else 2)
        ]
        if self.fmt == "nvfp4":
            return torch.ops.tensorrt.dequantize_nvfp4_experts(*tensors, torch.bfloat16)
        return torch.ops.tensorrt.dequantize_mxfp8_experts(*tensors, torch.bfloat16)

    def forward(self, x, offs):
        h = torch._grouped_mm(x, self.weight("up").transpose(-2, -1), offs=offs)
        return torch._grouped_mm(
            torch.relu(h), self.weight("down").transpose(-2, -1), offs=offs
        )


def _offsets(counts):
    return torch.cumsum(torch.tensor(counts), 0).to(torch.int32).cuda()


def _kernel_spy():
    return mock.patch.object(
        impl.moe, "quantized_grouped_mm", wraps=impl.moe.quantized_grouped_mm
    )


def _in_pytorch(compiled):
    return [
        node.target
        for name, sub in compiled.named_children()
        if name.startswith("_run_on_gpu")
        for node in sub.graph.nodes
        if node.op == "call_function"
    ]


class TestDequantizeExperts(TestCase):
    def test_nvfp4_matches_the_format(self):
        tensors = _nvfp4(3, 4, 64)
        out = torch.ops.tensorrt.dequantize_nvfp4_experts(*tensors, torch.float32)
        torch.testing.assert_close(out, _nvfp4_reference(*tensors), rtol=0, atol=0)

    def test_mxfp8_matches_the_format(self):
        weight, scale = _mxfp8(3, 4, 64)
        out = torch.ops.tensorrt.dequantize_mxfp8_experts(weight, scale, torch.float32)
        expected = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(
            32, -1
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_fused_op_equals_dequantize_then_grouped_mm(self):
        weight, scale, scale_2 = (t.cuda() for t in _nvfp4(4, 32, 64))
        x = torch.randn(6, 64, dtype=torch.bfloat16, device="cuda")
        offs = _offsets([2, 0, 3, 1])
        w = torch.ops.tensorrt.dequantize_nvfp4_experts(
            weight, scale, scale_2, torch.bfloat16
        )
        expected = torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
        out = torch.ops.tensorrt.grouped_mm_nvfp4(x, weight, scale, scale_2, offs)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rejects_mismatched_scales(self):
        weight, scale, scale_2 = _nvfp4(2, 4, 64)
        with self.assertRaisesRegex(ValueError, "NVFP4 scales"):
            torch.ops.tensorrt.dequantize_nvfp4_experts(
                weight, scale[..., :-1], scale_2, torch.bfloat16
            )


class TestQuantizedGroupedMMKernel(TestCase):
    def setUp(self):
        reason = aten_ops_converters._grouped_mm_kernel_unavailable()
        if reason is not None:
            self.skipTest(reason)

    def _compile(self, model, inputs, **settings):
        exported = torch.export.export(model, tuple(inputs))
        return torch_tensorrt.dynamo.compile(
            exported,
            inputs=inputs,
            use_explicit_typing=True,
            min_block_size=1,
            **settings,
        )

    @parameterized.expand(
        [
            ("nvfp4_decode", "nvfp4", [1, 0, 1, 1, 0, 1, 1, 1], 128, 256),
            ("nvfp4_several_rows", "nvfp4", [3, 0, 2, 5], 192, 64),
            ("nvfp4_rows_past_last_offset", "nvfp4", [1, 2, 0, 1], 64, 128),
            ("mxfp8_decode", "mxfp8", [1, 1, 0, 1, 1, 1, 0, 1], 256, 128),
            ("mxfp8_several_rows", "mxfp8", [4, 1, 0, 2], 128, 384),
        ]
    )
    def test_kernel_matches_pytorch(self, _, fmt, counts, k, n):
        torch.manual_seed(0)
        model = Experts(fmt, len(counts), k, n).cuda().eval()
        rows = sum(counts) + (2 if "past" in _ else 0)
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        offs = _offsets(counts)
        with torch.no_grad(), _kernel_spy() as spy:
            compiled = self._compile(model, [x, offs], grouped_mm_kernel=True)
            out = compiled(x, offs)
            expected = model(x, offs)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(_in_pytorch(compiled), [])
        # aten._grouped_mm leaves rows past the last offset undefined; the kernel zeroes them.
        grouped = sum(counts)
        self.assertTrue(bool((out[grouped:] == 0).all()))
        # Both sides round the weights to the same bfloat16 values; only the summation order differs.
        out, expected = out[:grouped], expected[:grouped]
        torch.testing.assert_close(
            out, expected, rtol=2e-2, atol=2e-2 * expected.abs().max().item()
        )

    def test_without_the_setting_the_weights_are_dequantized_at_build_time(self):
        torch.manual_seed(0)
        model = Experts("nvfp4", 4, 64, 64).cuda().eval()
        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        offs = _offsets([1, 1, 1, 1])
        with torch.no_grad(), _kernel_spy() as spy:
            compiled = self._compile(
                model,
                [x, offs],
                grouped_mm_kernel=False,
                grouped_mm_dense_max_macs=None,
            )
            out = compiled(x, offs)
        self.assertEqual(spy.call_count, 0)
        torch.testing.assert_close(out, model(x, offs), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    run_tests()
