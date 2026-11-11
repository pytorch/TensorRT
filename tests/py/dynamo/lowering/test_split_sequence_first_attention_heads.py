# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import copy
import importlib
import unittest
from unittest import mock

import torch
import torch_tensorrt
from parameterized import parameterized
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering

EMBED, HEADS, BATCH, QUERY_LEN, MEMORY_LEN = 64, 4, 2, 10, 16
PASS_LOGGER = (
    "torch_tensorrt.dynamo.lowering.passes.split_sequence_first_attention_heads"
)
attention_pass = importlib.import_module(PASS_LOGGER)


class CrossAttention(torch.nn.Module):
    """ACT's decoder cross-attention: the key has a positional term of its own."""

    def __init__(
        self, heads: int = HEADS, bias: bool = True, return_weights: bool = False
    ) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(EMBED, heads, bias=bias)
        self.return_weights = return_weights

    def forward(self, x, query_pos, memory, memory_pos):
        out, weights = self.attn(x + query_pos, memory + memory_pos, memory)
        return (out, weights) if self.return_weights else out


class ConstantQueryAttention(torch.nn.Module):
    """ACT's decoder cross-attention: the query depends on weights only, so constant
    folding turns it into a constant."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(EMBED, HEADS)
        self.query = torch.nn.Parameter(torch.randn(QUERY_LEN, BATCH, EMBED))

    def forward(self, memory, memory_pos):
        return self.attn(self.query, memory + memory_pos, memory)[0]


class SelfAttention(torch.nn.Module):
    """ACT's encoder self-attention: the query and the key share one positional term."""

    def __init__(self, packed: bool = False) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(EMBED, HEADS)
        self.packed = packed

    def forward(self, x, pos):
        query = key = x + pos
        return self.attn(query, key, query if self.packed else x)[0]


class HeadSplitInChild(torch.nn.Module):
    """Attention whose head reshapes, transposes and softmax run in child modules."""

    class View(torch.nn.Module):
        def forward(self, x):
            return x.clone().view(x.shape[0], x.shape[1] * HEADS, EMBED // HEADS)

    class Transpose(torch.nn.Module):
        def forward(self, x):
            return x.transpose(0, 1)

    class Softmax(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    def __init__(self) -> None:
        super().__init__()
        self.q, self.k, self.v = (torch.nn.Linear(EMBED, EMBED) for _ in range(3))
        self.view, self.transpose = self.View(), self.Transpose()
        self.softmax = self.Softmax()

    def split(self, x):
        return self.transpose(self.view(x))

    def forward(self, x, query_pos, memory, memory_pos):
        q = self.split(self.q(x + query_pos)) * (EMBED // HEADS) ** -0.5
        k = self.split(self.k(memory + memory_pos))
        v = self.split(self.v(memory))
        return torch.bmm(self.softmax(torch.bmm(q, k.transpose(1, 2))), v)


class SharedAttentionInput(HeadSplitInChild):
    """Attention whose scaled query or key transpose another op also reads."""

    def __init__(self, shared: str) -> None:
        super().__init__()
        self.shared = shared

    def forward(self, x, query_pos, memory, memory_pos):
        q = self.split(self.q(x + query_pos)) * (EMBED // HEADS) ** -0.5
        k_t = self.split(self.k(memory + memory_pos)).transpose(1, 2)
        v = self.split(self.v(memory))
        out = torch.bmm(self.softmax(torch.bmm(q, k_t)), v)
        return out, (q if self.shared == "query" else k_t).sum()


class MaskedMemoryAttention(torch.nn.Module):
    """Attention whose key and value a mask selects, so they can be empty at run time."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 2)

    def forward(self, x, memory, memory_pos, mask):
        return self.attn(x, (memory + memory_pos)[mask], memory[mask])[0]


def _inputs(
    dtype=torch.float32, query_len=QUERY_LEN, batch=BATCH, memory_len=MEMORY_LEN
):
    shapes = (
        (query_len, batch, EMBED),
        (query_len, 1, EMBED),
        (memory_len, batch, EMBED),
        (memory_len, 1, EMBED),
    )
    return tuple(torch.randn(shape, dtype=dtype, device="cuda") for shape in shapes)


def _lowered(module, inputs) -> torch.fx.GraphModule:
    program = torch.export.export(module, inputs)
    return program.run_decompositions(get_decompositions()).module()


def _permutes(gm: torch.fx.GraphModule) -> list:
    return [
        list(n.args[1])
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.permute.default
    ]


class TestSplitSequenceFirstAttentionHeadsLowering(unittest.TestCase):
    def setUp(self) -> None:
        # The rewrite of the graph does not need a TensorRT version that has the bug.
        patcher = mock.patch.object(
            attention_pass, "_has_the_attention_bug", return_value=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @parameterized.expand([("cross", CrossAttention, 4)])
    def test_lowering_splits_query_key_and_value(self, _, make_module, count) -> None:
        torch.manual_seed(0)
        module = make_module().cuda().eval().half()
        inputs = _inputs(torch.float16)[:count]
        gm = _lowered(module, inputs)
        self.assertEqual(_permutes(gm).count([1, 0, 2]), 4)

        gm = post_lowering(gm)

        # Only the transpose of the attention output stays 3-D.
        self.assertEqual(_permutes(gm).count([1, 0, 2]), 1)
        self.assertEqual(_permutes(gm).count([1, 2, 0, 3]), 3)
        with torch.no_grad():
            torch.testing.assert_close(gm(*inputs), module(*inputs))

    @parameterized.expand(
        [
            ("float32", CrossAttention, 4, torch.float32),
            (
                "weights_returned",
                lambda: CrossAttention(return_weights=True),
                4,
                torch.float16,
            ),
            ("one_head", lambda: CrossAttention(heads=1), 4, torch.float16),
            (
                "packed_projection",
                lambda: SelfAttention(packed=True),
                2,
                torch.float16,
            ),
            (
                "shared_scaled_query",
                lambda: SharedAttentionInput("query"),
                4,
                torch.float16,
            ),
            (
                "shared_key_transpose",
                lambda: SharedAttentionInput("key"),
                4,
                torch.float16,
            ),
        ]
    )
    def test_lowering_skips(self, _, make_module, count, dtype) -> None:
        module = make_module().cuda().eval().to(dtype)

        gm = post_lowering(_lowered(module, _inputs(dtype)[:count]))

        self.assertNotIn([1, 2, 0, 3], _permutes(gm))

    def test_lowering_skips_a_key_and_value_selected_by_a_mask(self) -> None:
        module = MaskedMemoryAttention().cuda().eval().half()
        x, memory, memory_pos = (
            torch.randn(shape, dtype=torch.float16, device="cuda")
            for shape in ((3, 2, 8), (4, 2, 8), (4, 1, 8))
        )
        mask = torch.arange(4, device="cuda") < 2

        gm = post_lowering(_lowered(module, (x, memory, memory_pos, mask)))

        self.assertNotIn([1, 2, 0, 3], _permutes(gm))

    def test_lowering_follows_torch_executed_ops(self) -> None:
        module = CrossAttention().cuda().eval().half()
        inputs = _inputs(torch.float16)
        unused = {"torch.ops.aten.view.default"}
        head_split = {"torch.ops.aten._reshape_copy.default"}

        gm = post_lowering(
            _lowered(module, inputs), CompilationSettings(torch_executed_ops=unused)
        )
        self.assertEqual(_permutes(gm).count([1, 2, 0, 3]), 3)

        gm = post_lowering(
            _lowered(module, inputs), CompilationSettings(torch_executed_ops=head_split)
        )
        self.assertNotIn([1, 2, 0, 3], _permutes(gm))

    @parameterized.expand(
        [
            ("view", HeadSplitInChild.View),
            ("transpose", HeadSplitInChild.Transpose),
            ("softmax", HeadSplitInChild.Softmax),
        ]
    )
    def test_lowering_follows_torch_executed_modules(self, _, child) -> None:
        module = HeadSplitInChild().cuda().eval().half()
        inputs = _inputs(torch.float16)
        settings = CompilationSettings(
            torch_executed_modules={f"{child.__module__}.{child.__qualname__}"}
        )

        gm = post_lowering(_lowered(module, inputs))
        self.assertEqual(_permutes(gm).count([1, 2, 0, 3]), 3)

        gm = post_lowering(_lowered(module, inputs), settings)
        self.assertNotIn([1, 2, 0, 3], _permutes(gm))

    def test_lowering_keeps_a_dynamic_empty_sequence_working(self) -> None:
        torch.manual_seed(0)
        module = CrossAttention().cuda().eval().half()
        memory_len = torch.export.Dim("memory_len", min=0, max=32)
        program = torch.export.export(
            module,
            _inputs(torch.float16),
            dynamic_shapes=(None, None, {0: memory_len}, {0: memory_len}),
        )

        gm = post_lowering(program.run_decompositions(get_decompositions()).module())

        self.assertEqual(_permutes(gm).count([1, 2, 0, 3]), 3)
        for length in (MEMORY_LEN, 0):
            inputs = _inputs(torch.float16, memory_len=length)
            with torch.no_grad():
                torch.testing.assert_close(gm(*inputs), module(*inputs))


class TestSplitSequenceFirstAttentionHeads(unittest.TestCase):
    # The value checks run with every TensorRT version, so they fail with a version
    # that has the bug but that _has_the_attention_bug does not list. They fail without
    # the pass only where TensorRT builds the original attention wrong, so where the
    # pass runs, each also checks that the pass rewrote the attention.
    @contextlib.contextmanager
    def _rewrites_attention(self):
        if not attention_pass._has_the_attention_bug():
            yield
            return
        with self.assertLogs(PASS_LOGGER, level="DEBUG") as logs:
            yield
        self.assertIn(
            "Split batch and heads of 1 sequence-first attention(s)",
            "\n".join(logs.output),
        )

    def _check_low_precision(self, module, inputs, dtype, rewrites=True) -> None:
        with torch.no_grad():
            expected = module(*inputs)
        module, inputs = module.to(dtype), tuple(x.to(dtype) for x in inputs)

        with self._rewrites_attention() if rewrites else contextlib.nullcontext():
            trt_module = torch_tensorrt.dynamo.compile(
                torch.export.export(module, inputs),
                arg_inputs=list(inputs),
                min_block_size=1,
            )

        with torch.no_grad():
            actual = trt_module(*inputs).float()
        tolerance = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    @parameterized.expand([("bfloat16", torch.bfloat16)])
    def test_low_precision_matches_pytorch(self, _, dtype) -> None:
        torch.manual_seed(0)
        self._check_low_precision(CrossAttention().cuda().eval(), _inputs(), dtype)

    def test_constant_query_matches_pytorch(self) -> None:
        torch.manual_seed(0)
        module = ConstantQueryAttention().cuda().eval()
        self._check_low_precision(module, _inputs()[2:], torch.float16)

    def test_empty_key_and_value_match_pytorch(self) -> None:
        torch.manual_seed(0)
        module = CrossAttention().cuda().eval()
        self._check_low_precision(
            module, _inputs(memory_len=0), torch.float16, rewrites=False
        )

    def test_key_and_value_selected_by_a_mask_match_pytorch(self) -> None:
        torch.manual_seed(0)
        module = MaskedMemoryAttention().cuda().eval().half()
        x, memory, memory_pos = (
            torch.randn(shape, dtype=torch.float16, device="cuda")
            for shape in ((3, 2, 8), (4, 2, 8), (4, 1, 8))
        )
        masks = [torch.arange(4, device="cuda") < length for length in (2, 1, 0)]
        trt_module = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (x, memory, memory_pos, masks[0])),
            arg_inputs=[x, memory, memory_pos, masks[0]],
            min_block_size=1,
        )

        for mask in masks:
            with torch.no_grad():
                torch.testing.assert_close(
                    trt_module(x, memory, memory_pos, mask),
                    module(x, memory, memory_pos, mask),
                    atol=1e-2,
                    rtol=1e-2,
                )

    def test_dynamic_shapes_match_pytorch(self) -> None:
        torch.manual_seed(0)
        module = CrossAttention().cuda().eval()
        half_module = copy.deepcopy(module).half()
        query_len = torch.export.Dim("query_len", min=2, max=32)
        batch = torch.export.Dim("batch", min=1, max=8)
        inputs = _inputs(torch.float16)
        program = torch.export.export(
            half_module,
            inputs,
            dynamic_shapes=(
                {0: query_len, 1: batch},
                {0: query_len},
                {1: batch},
                None,
            ),
        )

        with self._rewrites_attention():
            trt_module = torch_tensorrt.dynamo.compile(
                program, arg_inputs=list(inputs), min_block_size=1
            )

        for length, size in ((QUERY_LEN, BATCH), (23, 1), (5, 7)):
            inputs = _inputs(torch.float16, length, size)
            with torch.no_grad():
                expected = module(*(x.float() for x in inputs))
                actual = trt_module(*inputs).float()
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    # After torch.compile, which flattens the input of each projection on its own,
    # TensorRT 11.4 also gets self-attention whose query and key share an input wrong.
    # torch.autocast runs the softmax in float32, with a cast before it for bfloat16
    # and a cast after it.
    @parameterized.expand(
        [
            ("cross_without_bias", lambda: CrossAttention(bias=False), 4, None),
            ("self", SelfAttention, 2, None),
            ("cross_autocast_bfloat16", CrossAttention, 4, torch.bfloat16),
        ]
    )
    def test_torch_compile_matches_pytorch(
        self, _, make_module, count, autocast_dtype
    ) -> None:
        torch.manual_seed(0)
        module = make_module().cuda().eval()
        inputs = _inputs()[:count]
        with torch.no_grad():
            expected = module(*inputs)
        if autocast_dtype is None:
            module, inputs = module.half(), tuple(x.half() for x in inputs)
        torch._dynamo.reset()
        compiled = torch.compile(
            module,
            backend="torch_tensorrt",
            options={"min_block_size": 1, "pass_through_build_failures": True},
        )

        with (
            torch.no_grad(),
            torch.autocast(
                "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
            ),
            self._rewrites_attention(),
        ):
            actual = compiled(*inputs).float()
        torch._dynamo.reset()

        tolerance = 5e-2 if autocast_dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


class TestSplitSequenceFirstAttentionHeadsVersions(unittest.TestCase):
    def test_runs_only_with_the_tensorrt_versions_that_have_the_bug(self) -> None:
        lowered = _lowered(
            CrossAttention().cuda().eval().half(), _inputs(torch.float16)
        )
        trt = attention_pass.trt
        for package, version, rewrites in (
            ("tensorrt", "10.3.0", False),
            ("tensorrt", "11.1.0", False),
            ("tensorrt", "11.3.0", True),
            ("tensorrt", "11.4.0.106", True),
            ("tensorrt_rtx", "1.7.1", True),
        ):
            with (
                self.subTest(package=package, version=version),
                mock.patch.object(trt, "_package_name", package),
                mock.patch.object(trt, "__version__", version),
            ):
                gm = post_lowering(copy.deepcopy(lowered))
                self.assertEqual(
                    _permutes(gm).count([1, 2, 0, 3]), 3 if rewrites else 0
                )


if __name__ == "__main__":
    unittest.main()
