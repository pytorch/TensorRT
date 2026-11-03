import importlib.util
import platform
import unittest
from unittest import mock

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    scaled_dot_product_attention_validator,
)
from torch_tensorrt.dynamo.lowering.passes.decompose_unsupported_attention import (
    decompose_unsupported_attention,
)

from ..testing_utilities import lower_graph_testing


class TestDecomposeUnsupportedAttention(TestCase):
    def test_turing_fp32_attention_left_intact_for_pytorch_fallback(self):
        class SDPA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        # Also cover MLA shapes: shape-based decomposition must not override a
        # hardware rejection, even when K and V have different head dimensions.
        for value_head_dim in (8, 4):
            with (
                self.subTest(value_head_dim=value_head_dim),
                mock.patch(
                    "torch_tensorrt._features.ENABLED_FEATURES",
                    ENABLED_FEATURES._replace(tensorrt_rtx=True),
                ),
            ):
                inputs = (
                    torch.randn(1, 2, 4, 8),
                    torch.randn(1, 2, 4, 8),
                    torch.randn(1, 2, 4, value_head_dim),
                )
                gm = torch.export.export(SDPA(), inputs).module()
                settings = CompilationSettings(
                    decompose_attention=False,
                    target_compute_capabilities=[(7, 5)],
                )
                sdpa = next(
                    n
                    for n in gm.graph.nodes
                    if n.target == torch.ops.aten.scaled_dot_product_attention.default
                )
                self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))
                original_graph = str(gm.graph)

                gm = decompose_unsupported_attention(gm, settings)

                self.assertEqual(str(gm.graph), original_graph)
                self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))
                self.assertEqual(gm(*inputs), SDPA()(*inputs))

    def test_mla_kv_head_dim_mismatch_is_decomposed(self):
        # MLA: K head dim = nope + rope, V head dim = v only.
        b, h, s, d_k, d_v = 1, 2, 4, 6, 4

        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        inputs = [
            torch.randn(b, h, s, d_k),
            torch.randn(b, h, s, d_k),
            torch.randn(b, h, s, d_v),
        ]
        ep = torch.export.export(MLA(), tuple(inputs))
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        gm = ep.module()

        sdpa = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten.scaled_dot_product_attention.default
        )
        self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))

        gm = decompose_unsupported_attention(gm, settings)

        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)

    def test_equal_kv_shapes_left_intact(self):
        class MHA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        inputs = [torch.randn(1, 2, 4, 8) for _ in range(3)]
        ep = torch.export.export(MHA(), tuple(inputs))
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        gm = ep.module()
        gm = decompose_unsupported_attention(gm, settings)

        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.aten.scaled_dot_product_attention.default, targets)

    def test_attention_with_a_constant_key_is_left_for_pytorch(self):
        # TensorRT's attention layer gets a constant key wrong in FP16 and BF16. FP32 matches an
        # input key, and a key that takes its batch from an input keeps the layer as before.
        class ConstantKey(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.register_buffer("k", torch.randn(1, 2, 4, 8, dtype=dtype))

            def forward(self, q, v):
                k = self.k.expand(q.shape[0], -1, -1, -1)
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        batch = torch.export.Dim("batch")
        for dtype, dynamic_shapes, supported in (
            (torch.float16, None, False),
            (torch.bfloat16, None, False),
            (torch.float16, ({0: batch}, {0: batch}), True),
            (torch.float32, None, True),
        ):
            with self.subTest(dtype=dtype, dynamic=dynamic_shapes is not None):
                inputs = tuple(torch.randn(2, 2, 4, 8, dtype=dtype) for _ in range(2))
                ep = torch.export.export(
                    ConstantKey(dtype), inputs, dynamic_shapes=dynamic_shapes
                )
                gm = decompose_unsupported_attention(ep.module(), settings)
                nodes = [n for n in gm.graph.nodes if n.target == sdpa]
                self.assertEqual(len(nodes), 1)
                self.assertEqual(
                    scaled_dot_product_attention_validator(nodes[0], settings),
                    supported,
                )

    def test_post_lowering_mla_fully_supported(self):
        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        inputs = [
            torch.randn(1, 2, 4, 6, device="cuda"),
            torch.randn(1, 2, 4, 6, device="cuda"),
            torch.randn(1, 2, 4, 4, device="cuda"),
        ]
        unexpected_ops = {torch.ops.aten.scaled_dot_product_attention.default}
        fx_graph = torch.export.export(MLA(), tuple(inputs)).module()
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            decompose_attention=False,
        )
        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"SDPA should have been decomposed for MLA shapes: {unexpected_ops_seen}",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_decomposed_attention_runs_in_tensorrt(self):
        # The per-node fallback once left aten.transpose.int and aten._unsafe_view to
        # PyTorch, splitting the model across engines.
        import torch_tensorrt

        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        inputs = (
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16),
        )
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(MLA(), inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            decompose_attention=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(
            trt_model(*inputs), MLA()(*inputs), rtol=1e-2, atol=1e-2
        )
        self.assertEqual(
            [name for name, _ in trt_model.named_children()], ["_run_on_acc_0"]
        )

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_decomposed_attention_with_a_constant_bool_mask_builds(self):
        # The decomposition turns a bool mask into a logical_not after constant folding has run.
        import torch_tensorrt

        class MLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "mask", torch.ones(16, 16, device="cuda").tril() > 0
                )

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=self.mask
                )

        inputs = tuple(
            torch.randn(1, 2, 16, dim, device="cuda", dtype=torch.float16)
            for dim in (64, 64, 32)
        )
        model = MLA()
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(
            trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
        )

    def test_attention_on_inputs_cast_to_fp32_is_decomposed_where_tensorrt_fails(self):
        # TensorRT folds the casts into its attention layer and builds only when the whole layer can run in
        # one 16-bit type: all inputs folded (the query only with a given scale), no float mask, result cast back.
        class Attention(nn.Module):
            def __init__(
                self,
                cast,
                op="sdpa",
                scale=None,
                mask=False,
                out=None,
                view=None,
                lse=False,
            ):
                super().__init__()
                self.cast, self.op, self.scale, self.mask = cast, op, scale, mask
                self.out, self.view, self.lse = out, view, lse
                self.register_buffer("bias", torch.randn(1, 2, 6, 8))

            def forward(self, q, k, v):
                q, k, v = (
                    (
                        torch.ops.aten._to_copy.default(t, dtype=torch.float32)
                        if name in self.cast
                        else t
                    )
                    for name, t in zip("qkv", (q, k, v))
                )
                if self.view == "transpose":
                    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
                if self.view == "expand":
                    k, v = (t.expand(1, 2, 8, 8) for t in (k, v))
                mask = self.bias if self.mask else None
                if self.op == "sdpa":
                    out = torch.ops.aten.scaled_dot_product_attention.default(
                        q, k, v, mask, scale=self.scale
                    )
                else:
                    out = (
                        torch.ops.aten._scaled_dot_product_efficient_attention.default(
                            q, k, v, mask, self.lse, scale=self.scale
                        )[0]
                    )
                if self.out is not None:
                    out = torch.ops.aten._to_copy.default(out, dtype=self.out)
                return out

        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        for cast, options, decomposed in (
            ("qkv", {}, True),
            ("k", {}, True),
            ("v", {}, True),
            ("q", {}, False),
            ("q", {"scale": 0.5}, True),
            ("qkv", {"scale": 0.5, "out": torch.bfloat16}, False),
            ("qkv", {"scale": 0.5}, True),
            ("qkv", {"scale": 0.5, "out": torch.float16}, True),
            ("qkv", {"scale": 0.5, "out": torch.bfloat16, "fp16_kv": True}, True),
            ("qkv", {"scale": 0.5, "mask": True, "out": torch.bfloat16}, True),
            ("qkv", {"view": "transpose"}, True),
            ("qkv", {"view": "expand"}, False),
            ("qkv", {"op": "efficient", "mask": True}, True),
            ("qkv", {"op": "efficient", "mask": True, "lse": True}, False),
        ):
            with self.subTest(cast=cast, **options):
                fp16_kv = options.pop("fp16_kv", False)
                model = Attention(cast, **options)
                view = options.get("view")
                shapes = {
                    "transpose": ((1, 6, 2, 8), (1, 8, 2, 8), (1, 8, 2, 8)),
                    "expand": ((1, 2, 6, 8), (1, 1, 8, 8), (1, 1, 8, 8)),
                }.get(view, ((1, 2, 6, 8), (1, 2, 8, 8), (1, 2, 8, 8)))
                half = dict.fromkeys(cast, torch.bfloat16)
                if fp16_kv:
                    half.update(k=torch.float16, v=torch.float16)
                inputs = tuple(
                    torch.randn(shape, dtype=half.get(name, torch.float32))
                    for name, shape in zip("qkv", shapes)
                )
                gm = torch.export.export(model, inputs).module()
                gm = decompose_unsupported_attention(gm, settings)
                attention = [n for n in gm.graph.nodes if "attention" in str(n.target)]
                self.assertEqual(not attention, decomposed)
                if options.get("op") == "efficient" and not decomposed:
                    continue  # the efficient op has no CPU kernel
                q, k, v = (t.float() for t in inputs)
                if view == "transpose":
                    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k.expand(1, 2, 8, 8),
                    v.expand(1, 2, 8, 8),
                    attn_mask=model.bias if model.mask else None,
                    scale=model.scale,
                )
                if model.out is not None:
                    expected = expected.to(model.out)
                torch.testing.assert_close(gm(*inputs), expected)

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_attention_on_inputs_cast_to_fp32_builds(self):
        import torch_tensorrt

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(1, 2, 8, 16))

            def forward(self, q, k, v):
                q, k, v = (
                    torch.ops.aten._to_copy.default(t, dtype=torch.float32)
                    for t in (q, k, v)
                )
                return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, self.bias, False
                )[0]

        model = Attention().cuda()
        inputs = [
            torch.randn(1, 2, length, 32, device="cuda", dtype=torch.bfloat16)
            for length in (8, 16, 16)
        ]
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, tuple(inputs)),
            arg_inputs=inputs,
            min_block_size=1,
            disable_tf32=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        # TensorRT-RTX ignores disable_tf32, so it can only match to TF32 precision. That
        # still fails attention run in BF16, which is off by more than 2e-3.
        tol = 1e-3 if ENABLED_FEATURES.tensorrt_rtx else 1e-4
        torch.testing.assert_close(
            trt_model(*inputs), model(*inputs), rtol=tol, atol=tol
        )

    def test_attentions_sharing_a_bf16_query_are_decomposed(self):
        # Wan2.1's image-to-video cross-attention reads one query against text and image keys.
        class TwoAttentions(nn.Module):
            def forward(self, q, text, image):
                q = q.transpose(1, 2)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, text, text
                ) + torch.nn.functional.scaled_dot_product_attention(q, image, image)

        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        for dtype, remaining in ((torch.bfloat16, 0), (torch.float16, 2)):
            with self.subTest(dtype=dtype):
                inputs = (
                    torch.randn(1, 6, 2, 8, dtype=dtype),
                    torch.randn(1, 2, 5, 8, dtype=dtype),
                    torch.randn(1, 2, 3, 8, dtype=dtype),
                )
                gm = torch.export.export(TwoAttentions(), inputs).module()
                gm = decompose_unsupported_attention(gm, settings)
                self.assertEqual(
                    sum(n.target == sdpa for n in gm.graph.nodes), remaining
                )
                if dtype == torch.bfloat16:
                    torch.testing.assert_close(
                        gm(*inputs), TwoAttentions()(*inputs), rtol=1e-2, atol=1e-2
                    )

    def test_attentions_reading_one_tensor_in_different_dtypes_are_kept(self):
        # Only one of the two queries is bfloat16, so they cannot become one tensor in TensorRT.
        class TwoDtypes(nn.Module):
            def forward(self, q, k16, kbf16):
                cast = torch.ops.aten._to_copy.default
                a = torch.nn.functional.scaled_dot_product_attention(
                    cast(q, dtype=torch.float16), k16, k16
                )
                b = torch.nn.functional.scaled_dot_product_attention(
                    cast(q, dtype=torch.bfloat16), kbf16, kbf16
                )
                return a.float() + b.float()

        inputs = (
            torch.randn(1, 2, 6, 8),
            torch.randn(1, 2, 5, 8, dtype=torch.float16),
            torch.randn(1, 2, 5, 8, dtype=torch.bfloat16),
        )
        gm = torch.export.export(TwoDtypes(), inputs).module()
        gm = decompose_unsupported_attention(
            gm, CompilationSettings(min_block_size=1, decompose_attention=False)
        )
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        self.assertEqual(sum(n.target == sdpa for n in gm.graph.nodes), 2)

    @unittest.skipIf(
        importlib.util.find_spec("modelopt") is None or platform.system() != "Linux",
        "needs NVIDIA Model Optimizer's quantize op, which is only supported on Linux",
    )
    def test_attention_whose_output_is_quantized_to_fp8_is_decomposed(self):
        import modelopt.torch.quantization  # noqa: F401  # registers tensorrt.quantize_op

        quantize = torch.ops.tensorrt.quantize_op.default

        def graph(fp8_inputs: bool) -> torch.fx.GraphModule:
            g = torch.fx.Graph()
            q, k, v, amax = (g.placeholder(n) for n in ("q", "k", "v", "amax"))
            if fp8_inputs:
                q, k, v = (
                    g.call_function(quantize, (t, amax, 8, 4, False, False))
                    for t in (q, k, v)
                )
            out = g.call_function(
                torch.ops.aten.scaled_dot_product_attention.default, (q, k, v)
            )
            out = g.call_function(torch.ops.aten.transpose.int, (out, 1, 2))
            g.output(g.call_function(quantize, (out, amax, 8, 4, False, False)))
            return torch.fx.GraphModule({}, g)

        for fp8_inputs, supported in ((False, False), (True, True)):
            with self.subTest(fp8_inputs=fp8_inputs):
                sdpa = next(
                    n
                    for n in graph(fp8_inputs).graph.nodes
                    if n.target == torch.ops.aten.scaled_dot_product_attention.default
                )
                self.assertEqual(
                    scaled_dot_product_attention_validator(sdpa, None), supported
                )

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_attentions_sharing_a_query_build(self):
        # TensorRT crashed or failed building these before the attentions were decomposed.
        import torch_tensorrt

        class TwoAttentions(nn.Module):
            def __init__(self, combine):
                super().__init__()
                self.combine = combine

            def forward(self, q, text, image):
                return self.combine(
                    torch.nn.functional.scaled_dot_product_attention(q, text, text),
                    torch.nn.functional.scaled_dot_product_attention(q, image, image),
                )

        inputs = tuple(
            torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16)
            for n in (256, 512, 257)
        )
        for combine in (torch.add, lambda a, b: torch.cat([a, b], dim=-1)):
            with self.subTest(combine=combine):
                model = TwoAttentions(combine)
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=list(inputs),
                    min_block_size=1,
                    decompose_attention=False,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                )
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
                )
                # The decomposed attentions run in TensorRT, not in PyTorch between engines.
                self.assertEqual(
                    [name for name, _ in trt_model.named_children()], ["_run_on_acc_0"]
                )

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() < (8, 9)
        or importlib.util.find_spec("modelopt") is None
        or platform.system() != "Linux",
        "needs an FP8 GPU and NVIDIA Model Optimizer's quantize op (Linux only)",
    )
    def test_attention_whose_output_is_quantized_to_fp8_builds(self):
        # TensorRT 11.4 failed to build this before the attention was decomposed.
        import modelopt.torch.quantization  # noqa: F401  # registers tensorrt.quantize_op
        import torch_tensorrt

        class AttentionThenFP8(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("amax", torch.tensor(4.0))
                self.out = nn.Linear(256, 256)

            def forward(self, q, k, v):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2).reshape(1, 128, 256)
                x = torch.ops.tensorrt.quantize_op(x, self.amax, 8, 4, False, False)
                return self.out(x)

        model = AttentionThenFP8().cuda().half().eval()
        inputs = tuple(
            torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
            for _ in range(3)
        )
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            decompose_attention=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(
            trt_model(*inputs), model(*inputs), rtol=0.1, atol=0.1
        )


if __name__ == "__main__":
    run_tests()
