import importlib.util
import platform
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    bitwise_not_type_validator,
    bitwise_type_validator,
    scaled_dot_product_attention_validator,
)
from torch_tensorrt.dynamo.lowering.passes.decompose_unsupported_attention import (
    decompose_unsupported_attention,
)
from torch_tensorrt.dynamo.lowering.passes.mark_merged_bf16_attention_queries import (
    mark_merged_bf16_attention_queries,
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

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_decomposed_attention_gets_its_bool_mask_in_its_own_engine(self):
        # When the decomposed nodes carried the trace's tensor_meta, the first engine computed the
        # float bias from the mask and passed it, with a scalar and copies of K and V, to the
        # attention's engine.
        import torch_tensorrt

        class Attention(nn.Module):
            def forward(self, q, k, v, mask):
                return torch.nn.functional.scaled_dot_product_attention(
                    (q + 1) * 2, k, v, attn_mask=mask
                )

        # K and V head dims differ, so the attention is decomposed.
        inputs = [
            torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16),
            torch.rand(1, 2, 1, 16, device="cuda") > 0.3,
        ]
        model = Attention()
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, tuple(inputs)),
            arg_inputs=inputs,
            min_block_size=1,
            # Puts q + 1 in an engine before the attention.
            torch_executed_ops={torch.ops.aten.mul.Tensor},
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        engines = []
        for name, engine in trt_model.named_children():
            if "_run_on_acc" in name:
                engine.register_forward_hook(
                    lambda _, args, out: engines.append(
                        (
                            [arg.dtype for arg in args],
                            len(out) if isinstance(out, (list, tuple)) else 1,
                        )
                    )
                )
        torch.testing.assert_close(
            trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
        )
        self.assertEqual(
            engines,
            [([torch.float16], 1), ([torch.bool] + [torch.float16] * 3, 1)],
        )

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_readers_of_decomposed_attention_stay_in_tensorrt(self):
        # The sort and diagonal validators read the shape of their input, which the
        # decomposed nodes carry in "val" only.
        import torch_tensorrt

        class Attention(nn.Module):
            def __init__(self, reader):
                super().__init__()
                self.reader = reader

            def forward(self, q, k, v, mask):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )
                return self.reader(out) + 1

        # K and V head dims differ, so the attention is decomposed.
        inputs = [
            torch.randn(1, 2, 4, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16),
            torch.rand(1, 2, 4, 16, device="cuda") > 0.3,
        ]
        for name, reader in (
            ("sort", lambda out: torch.sort(out, dim=-1).values),
            ("diagonal", lambda out: torch.diagonal(out, 0, -2, -1)),
        ):
            with self.subTest(name):
                model = Attention(reader)
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                    require_full_compilation=True,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                )
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
                )

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_float64_cat_of_decomposed_attention_runs_in_pytorch(self):
        # The cat validator reads the dtypes of its operands, which the decomposed nodes
        # carry in "val" only. An empty float64 operand makes the result float64, which
        # TensorRT has only with truncate_double, so the cat runs in PyTorch.
        import torch_tensorrt

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("empty", torch.empty(0, dtype=torch.float64))

            def forward(self, q, k, v):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return torch.cat([out, self.empty])

        # K and V head dims differ, so the attention is decomposed.
        inputs = [
            torch.randn(1, 2, 4, 64, device="cuda"),
            torch.randn(1, 2, 16, 64, device="cuda"),
            torch.randn(1, 2, 16, 32, device="cuda"),
        ]
        model = Attention().cuda()
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, tuple(inputs)),
            arg_inputs=inputs,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        result = trt_model(*inputs)
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result, model(*inputs), rtol=1e-2, atol=1e-2)

    def test_bitwise_validators_read_val(self):
        # Like a decomposed node, a node can carry its type in "val" only.
        aten = torch.ops.aten
        mask = SimpleNamespace(meta={"val": torch.empty(4, 4, dtype=torch.bool)})
        for validator, node in (
            (
                bitwise_not_type_validator,
                SimpleNamespace(target=aten.bitwise_not.default, args=(mask,)),
            ),
            (
                bitwise_type_validator,
                SimpleNamespace(target=aten.bitwise_and.Tensor, args=(mask, mask)),
            ),
            (
                bitwise_type_validator,
                SimpleNamespace(target=aten.bitwise_xor.Scalar, args=(mask, True)),
            ),
            (
                bitwise_type_validator,
                SimpleNamespace(
                    target=aten.bitwise_xor.Scalar_Tensor, args=(True, mask)
                ),
            ),
        ):
            with self.subTest(str(node.target)):
                self.assertTrue(validator(node))

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
                self.bias[0, 0, 0] = float("-inf")  # One query sees no key.

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
            ("", {}, False),
            ("", {"mask": True}, True),
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

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_fully_masked_query_gives_zeros(self):
        # PyTorch gives zeros for a query that the mask hides all keys from.
        import torch_tensorrt

        class Attention(nn.Module):
            def __init__(self, op, upcast):
                super().__init__()
                self.op, self.upcast = op, upcast

            def forward(self, q, k, v, mask):
                if self.upcast:
                    q, k, v = (t.float() for t in (q, k, v))
                if self.op == "efficient":
                    return (
                        torch.ops.aten._scaled_dot_product_efficient_attention.default(
                            q, k, v, mask, False
                        )[0]
                    )
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, scale=0.5
                )
                return out.to(torch.bfloat16) if self.upcast else out

        for op, mask, dtype, decompose, fully_masked, upcast in (
            ("sdpa", "bool", torch.float32, False, True, False),
            ("sdpa", "additive", torch.float32, False, True, False),
            ("efficient", "additive", torch.float32, False, True, False),
            ("sdpa", "bool", torch.float32, True, True, False),
            ("sdpa", "additive", torch.float32, True, True, False),
            # truncate_double runs the FP64 decomposition in FP32.
            ("sdpa", "bool", torch.float64, True, True, False),
            # A control, right without the fix too.
            ("sdpa", "bool", torch.float32, False, False, False),
            # TensorRT's 16-bit attention already gives zeros.
            ("sdpa", "bool", torch.float16, False, True, False),
            ("sdpa", "additive", torch.float16, False, True, False),
            ("sdpa", "bool", torch.float16, True, True, False),
            # Runs as one BF16 attention layer, which builds only with the cast back right after it.
            ("sdpa", "bool", torch.bfloat16, False, True, True),
        ):
            with self.subTest(
                op=op,
                mask=mask,
                dtype=dtype,
                decompose=decompose,
                fully_masked=fully_masked,
                upcast=upcast,
            ):
                torch.manual_seed(0)
                inputs = [
                    torch.randn(2, 4, 16, 32, device="cuda", dtype=dtype)
                    for _ in range(3)
                ]
                visible = torch.ones(2, 4, 16, 16, dtype=torch.bool, device="cuda")
                visible[..., 12:] = False
                if fully_masked:
                    visible[0, :, :3] = False
                if mask == "additive":
                    visible = torch.zeros(
                        visible.shape, device="cuda", dtype=dtype
                    ).masked_fill(~visible, float("-inf"))
                inputs.append(visible)
                model = Attention(op, upcast)
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                    decompose_attention=decompose,
                    # The bias is not causal.
                    attn_bias_is_causal=False,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    truncate_double=dtype == torch.float64,
                )
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
                )

    class _MaskedAttention(nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )

    class _AroundAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = TestDecomposeUnsupportedAttention._MaskedAttention()

        def forward(self, q, k, v, mask):
            return self.attention(q * 2, k, v, mask) + 1

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_masked_attention_without_keys_gives_zeros(self):
        import torch_tensorrt

        # The decomposed attention is left to PyTorch: TensorRT cannot build its empty matmuls.
        for dtype, decompose, min_block_size in (
            (torch.bool, False, 5),
            (torch.float32, False, 5),
            (torch.bool, True, 1000),
            (torch.float32, True, 1000),
        ):
            with self.subTest(mask=dtype, decompose=decompose):
                q = torch.randn(1, 1, 3, 32, device="cuda")
                k = v = torch.randn(1, 1, 0, 32, device="cuda")
                mask = torch.zeros(3, 0, device="cuda", dtype=dtype)
                inputs = (q, k, v, mask)
                model = self._MaskedAttention()
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=inputs,
                    decompose_attention=decompose,
                    min_block_size=min_block_size,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                )
                torch.testing.assert_close(trt_model(*inputs), model(*inputs))

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_masked_fp32_attention_left_to_pytorch_stays_there(self):
        import torch_tensorrt

        inputs = [torch.randn(2, 4, 16, 32, device="cuda") for _ in range(3)]
        inputs.append(torch.rand(2, 4, 16, 16, device="cuda") > 0.2)
        model = self._AroundAttention()
        attention = (
            f"{self._MaskedAttention.__module__}.{self._MaskedAttention.__qualname__}"
        )
        for settings in (
            {
                "torch_executed_ops": {
                    torch.ops.aten.scaled_dot_product_attention.default
                }
            },
            {"torch_executed_modules": {attention}},
        ):
            with self.subTest(settings=settings):
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                    attn_bias_is_causal=False,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    **settings,
                )
                kept = [
                    node
                    for name, part in trt_model.named_children()
                    if "_run_on_gpu" in name
                    for node in part.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                ]
                self.assertEqual(len(kept), 1)
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-3, atol=1e-3
                )

    class _MlaAttention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    class _AroundMla(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = TestDecomposeUnsupportedAttention._MlaAttention()

        def forward(self, q, k, v):
            return self.attention(q * 2, k, v) + 1

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_declined_attention_left_to_pytorch_stays_there(self):
        import torch_tensorrt

        # K and V head dims differ, so the converter declines this attention.
        inputs = [
            torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.half),
            torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.half),
            torch.randn(1, 2, 4, 4, device="cuda", dtype=torch.half),
        ]
        model = self._AroundMla()
        attention = f"{self._MlaAttention.__module__}.{self._MlaAttention.__qualname__}"
        for settings in (
            {
                "torch_executed_ops": {
                    torch.ops.aten.scaled_dot_product_attention.default
                }
            },
            {"torch_executed_modules": {attention}},
        ):
            with self.subTest(settings=settings):
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    **settings,
                )
                kept = [
                    node
                    for name, part in trt_model.named_children()
                    if "_run_on_gpu" in name
                    for node in part.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                ]
                self.assertEqual(len(kept), 1)
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
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

    class _QueriesComputedTwice(nn.Module):
        # Two passes that compute the same query separately, as classifier-free guidance does for
        # the work before the text joins it. TensorRT merges them into one query.
        def __init__(self, second_scale=2.0):
            super().__init__()
            self.second_scale = second_scale

        def forward(self, q, k1, k2):
            first = torch.nn.functional.scaled_dot_product_attention(
                q * 2.0 + 1.0, k1, k1
            )
            second = torch.nn.functional.scaled_dot_product_attention(
                q * self.second_scale + 1.0, k2, k2
            )
            return first + second

    def test_attentions_computing_the_same_bf16_query_are_decomposed(self):
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        for dtype, second_scale, remaining in (
            (torch.bfloat16, 2.0, 0),
            (torch.bfloat16, 3.0, 2),  # different queries
            (torch.float16, 2.0, 2),  # float16 builds
        ):
            with self.subTest(dtype=dtype, second_scale=second_scale):
                model = self._QueriesComputedTwice(second_scale)
                inputs = tuple(torch.randn(1, 2, 6, 8, dtype=dtype) for _ in range(3))
                gm = torch.export.export(model, inputs).module()
                gm = mark_merged_bf16_attention_queries(gm, settings)
                gm = decompose_unsupported_attention(gm, settings)
                self.assertEqual(
                    sum(n.target == sdpa for n in gm.graph.nodes), remaining
                )
                torch.testing.assert_close(
                    gm(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
                )

    def test_attentions_on_random_queries_are_kept(self):
        # Two draws of random noise are different queries, though the same ops make them.
        class RandomQueries(nn.Module):
            def forward(self, k1, k2):
                return torch.nn.functional.scaled_dot_product_attention(
                    torch.randn_like(k1), k1, k1
                ) + torch.nn.functional.scaled_dot_product_attention(
                    torch.randn_like(k1), k2, k2
                )

        inputs = tuple(torch.randn(1, 2, 6, 8, dtype=torch.bfloat16) for _ in range(2))
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        gm = torch.export.export(RandomQueries(), inputs).module()
        gm = mark_merged_bf16_attention_queries(gm, settings)
        gm = decompose_unsupported_attention(gm, settings)
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        self.assertEqual(sum(n.target == sdpa for n in gm.graph.nodes), 2)

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_attentions_computing_the_same_bf16_query_build(self):
        # TensorRT 11.4 crashes (a segfault) while building these as two attention layers.
        import torch_tensorrt

        model = self._QueriesComputedTwice()
        inputs = tuple(
            torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(
            trt_model(*inputs), model(*inputs), rtol=2e-2, atol=2e-2
        )

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
