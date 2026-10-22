# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.lowering.passes.fold_shared_kv_attention import (
    fold_shared_kv_attention,
)

LOGGER = "torch_tensorrt.dynamo.lowering.passes.fold_shared_kv_attention"


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Hugging Face's ``repeat_kv``: one copy of each key/value head per query head."""
    batch, kv_heads, seq, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, groups, seq, dim)
    return x.reshape(batch, kv_heads * groups, seq, dim)


class RepeatKVAttention(torch.nn.Module):
    """Eager attention as Hugging Face writes it, optionally with RMSNorm on queries and keys."""

    def __init__(
        self, heads: int, kv_heads: int, head_dim: int, qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.groups = heads // kv_heads
        self.scaling = head_dim**-0.5
        self.qk_norm = qk_norm

    def forward(self, q, k, v, mask):
        if self.qk_norm:
            q, k = F.rms_norm(q, q.shape[-1:]), F.rms_norm(k, k.shape[-1:])
        k, v = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling + mask
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v)


class SdpaAttention(torch.nn.Module):
    """Attention through ``scaled_dot_product_attention``, as Hugging Face's SDPA path writes it."""

    def __init__(self, heads: int, kv_heads: int, enable_gqa: bool) -> None:
        super().__init__()
        self.groups = heads // kv_heads
        self.enable_gqa = enable_gqa

    def forward(self, q, k, v, mask):
        if not self.enable_gqa:
            k, v = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=self.enable_gqa
        )


def _inputs(heads, kv_heads, head_dim=16, q_len=7, kv_len=11, dtype=torch.float32):
    q = torch.randn(2, heads, q_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(2, kv_heads, kv_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(2, kv_heads, kv_len, head_dim, dtype=dtype, device="cuda")
    mask = torch.zeros(2, 1, q_len, kv_len, dtype=dtype, device="cuda")
    mask[..., -3:] = torch.finfo(dtype).min
    return q, k, v, mask


def _lowered(module, inputs, settings=CompilationSettings()) -> torch.fx.GraphModule:
    program = torch.export.export(module, inputs)
    decompositions = get_decompositions(
        decompose_attention=settings.decompose_attention,
        use_fp32_acc=settings.use_fp32_acc,
    )
    return program.run_decompositions(decompositions).module()


def _targets(gm: torch.fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


ON = CompilationSettings(fold_shared_kv_attention=True)


class TestFoldSharedKVAttention(unittest.TestCase):
    @parameterized.expand([("plain", False), ("rms_norm_on_q_and_k", True)])
    def test_lowering_folds_both_matmuls_of_multi_query_attention(
        self, _, qk_norm
    ) -> None:
        inputs = _inputs(heads=8, kv_heads=1)
        module = RepeatKVAttention(8, 1, 16, qk_norm).eval()
        unfolded = post_lowering(_lowered(module, inputs), CompilationSettings())
        self.assertEqual(_targets(unfolded).count(torch.ops.aten.expand.default), 2)

        folded = post_lowering(_lowered(module, inputs), ON)

        self.assertNotIn(torch.ops.aten.expand.default, _targets(folded))
        matmuls = [
            n for n in folded.graph.nodes if n.target == torch.ops.aten.matmul.default
        ]
        # The 8 query heads of 7 rows become 56 rows against the single head.
        self.assertEqual(
            [[tuple(arg.meta["val"].shape) for arg in n.args] for n in matmuls],
            [[(2, 1, 56, 16), (2, 1, 16, 11)], [(2, 1, 56, 11), (2, 1, 11, 16)]],
        )
        self.assertTrue(all("nn_module_stack" in n.meta for n in matmuls))
        with torch.no_grad():
            torch.testing.assert_close(folded(*inputs), unfolded(*inputs))

    @parameterized.expand(
        [
            ("repeat_kv", False, False),
            ("repeat_kv_fp32_acc", False, True),
            ("enable_gqa", True, False),
        ]
    )
    def test_folds_decomposed_sdpa(self, _, enable_gqa, use_fp32_acc) -> None:
        settings = CompilationSettings(
            fold_shared_kv_attention=True,
            decompose_attention=True,
            use_fp32_acc=use_fp32_acc,
        )
        inputs = _inputs(heads=8, kv_heads=1, dtype=torch.float16)
        module = SdpaAttention(8, 1, enable_gqa).eval()
        gm = _lowered(module, inputs, settings)
        self.assertEqual(_targets(gm).count(torch.ops.aten.expand.default), 2)

        folded = fold_shared_kv_attention(gm, settings)

        self.assertNotIn(torch.ops.aten.expand.default, _targets(folded))
        with torch.no_grad():
            expected = module(*(x.float() for x in inputs))
            torch.testing.assert_close(
                folded(*inputs).float(), expected, atol=1e-2, rtol=1e-2
            )

    def test_does_nothing_unless_enabled(self) -> None:
        inputs = _inputs(heads=8, kv_heads=1)
        gm = _lowered(RepeatKVAttention(8, 1, 16).eval(), inputs)
        before = _targets(gm)

        folded = fold_shared_kv_attention(gm, CompilationSettings())

        self.assertEqual(_targets(folded), before)

    @parameterized.expand([("grouped_query", 8, 2), ("multi_head", 8, 8)])
    def test_leaves_attention_with_several_kv_heads_alone(
        self, _, heads, kv_heads
    ) -> None:
        inputs = _inputs(heads=heads, kv_heads=kv_heads)
        gm = _lowered(RepeatKVAttention(heads, kv_heads, 16).eval(), inputs)
        before = _targets(gm)

        with self.assertLogs(LOGGER, level="INFO") as logs:
            folded = fold_shared_kv_attention(gm, ON)

        self.assertEqual(_targets(folded), before)
        self.assertIn("found no matmul", "\n".join(logs.output))

    def test_leaves_expand_that_also_repeats_a_size_one_dim_alone(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x, kv):
                # expand also repeats the last dim, so these are not 6 copies of kv.
                repeated = kv.unsqueeze(2).expand(2, 1, 1, 6, 6).reshape(2, 6, 6, 1)
                return torch.matmul(x, repeated)

        inputs = (
            torch.randn(2, 6, 3, 6, device="cuda"),
            torch.randn(2, 1, 6, 1, device="cuda"),
        )
        gm = _lowered(Module(), inputs)
        before = _targets(gm)

        folded = fold_shared_kv_attention(gm, ON)

        self.assertEqual(_targets(folded), before)
        with torch.no_grad():
            torch.testing.assert_close(folded(*inputs), Module()(*inputs))

    def test_compiled_multi_query_attention_matches_pytorch(self) -> None:
        inputs = _inputs(
            heads=8, kv_heads=1, head_dim=256, q_len=50, kv_len=96, dtype=torch.float16
        )
        module = RepeatKVAttention(8, 1, 256).eval()
        program = torch.export.export(module, inputs)
        with self.assertLogs(LOGGER, level="DEBUG") as logs:
            compiled = torch_tensorrt.dynamo.compile(
                program,
                arg_inputs=inputs,
                min_block_size=1,
                fold_shared_kv_attention=True,
            )
        self.assertIn("Folded 2 matmul(s)", "\n".join(logs.output))
        with torch.no_grad():
            expected = module(*(x.float() for x in inputs))
            # The float16 engine differs from the float32 reference by float16 rounding.
            torch.testing.assert_close(
                compiled(*inputs).float(), expected, atol=1e-2, rtol=1e-2
            )


if __name__ == "__main__":
    unittest.main()
