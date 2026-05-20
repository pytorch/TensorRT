import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestInputAsOutput(TestCase):
    def test_input_as_output(self):
        class InputAsOutput(torch.nn.Module):
            def forward(self, x, y):
                y_new = y + x + 1
                y_new = y_new * 7
                return (y_new, x, y)

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InputAsOutput())
        lower_graph_testing(fx_graph, inputs, min_block_size=1)
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"InputAsOutput TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestLoweringPassMembership(TestCase):
    def insert_at_end(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[-1])

        _remove_lowering_pass(-1)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)

    def insert_at_index(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass(index=0)
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[0])

        _remove_lowering_pass(0)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)


class TestPrimBroadcastFusion(TestCase):
    def test_broadcast_fusion(self):
        class BroadcastFusion(torch.nn.Module):
            def forward(self, x):
                return torch.var_mean(x, keepdim=True)[1]

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(BroadcastFusion())
        expected_ops = {torch.ops.aten.sum.dim_IntList}
        unexpected_ops = {torch.ops.aten.var.default, torch.ops.prims.var.default}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"BroadcastFusion TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestComplexSubgraph(TestCase):
    def test_complex_subgraph(self):
        BATCH = 1
        SEQ_LEN = 2
        HEADS = 1
        DIM = 2

        class RotaryAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = DIM
                self.wq = torch.nn.Linear(self.dim, self.dim)
                self.seq_len = SEQ_LEN

                self.register_buffer(
                    "freqs_ex_tensor",
                    self._freqs_ex_tensor(),
                    persistent=True,
                )

            def rotary_embedding(self, x, dim, freqs_cis=None):
                x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                x_out_flatten = torch.view_as_real(x_ * freqs_cis)
                return x_out_flatten.type_as(x)

            def _freqs_ex_tensor(self):
                real = torch.tensor([[[[1.0000]], [[2.0000]]]], device="cuda")
                imag = torch.tensor([[[[0.0000]], [[3.0000]]]], device="cuda")

                z = torch.complex(real, imag)
                return z

            def forward(self, x):
                q = self.wq(x)
                freqs_cis = self._freqs_ex_tensor().to(q.device)
                q_out = self.rotary_embedding(q, self.dim, freqs_cis=freqs_cis)
                return q_out

        inputs = [torch.randn(BATCH, SEQ_LEN, HEADS, DIM).cuda()]
        model = RotaryAttention()
        model = model.cuda()

        expected_ops = {torch.ops.aten.mul.Tensor}
        unexpected_ops = {
            torch.ops.aten.view_as_complex.default,
            torch.ops.aten.view_as_real.default,
        }

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            model,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            model,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs)[0].detach().cpu()
        torch_model_results = model(*inputs)[0].detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"ComplexSubgraph TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestRemoveSymIntNodes(TestCase):
    def test_remove_sym_nodes(self):
        class ModelContainSymIntNodes(torch.nn.Module):
            def __init__(self, embed_dim: int):
                super().__init__()
                self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
                self.embed_dim = embed_dim
                self.qkv_proj = torch.nn.Linear(self.embed_dim, self.embed_dim * 3)

            def forward(self, x: torch.Tensor):
                batch_size = x.shape[0]
                cls_token = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = self.qkv_proj(x)
                reshaped_qkv = x.reshape(batch_size, x.size(1), 3, 12, -1)
                return reshaped_qkv

        model = ModelContainSymIntNodes(embed_dim=768).cuda().eval()
        inputs = torch.randn(4, 196, 768).cuda()
        torch._dynamo.mark_dynamic(inputs, index=0, min=2, max=32)
        trt_module = torch.compile(
            model,
            backend="tensorrt",
            options={"use_python_runtime": False, "min_block_size": 1},
        )
        out = trt_module(inputs)
        # if the model can be successfully compiled, we regard the test as passed
        self.assertTrue(True)


class TestRewriteEfficientAttention(TestCase):
    def test_force_causal_efficient_attention(self):
        class RewriteEfficientAttention(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                attn_bias=None,
                compute_log_sumexp=False,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            ):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    compute_log_sumexp,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return out[0]

        attn_bias = torch.zeros(4, 8, 32, 32, device="cuda")
        upper = torch.triu(
            torch.ones((32, 32), dtype=torch.bool, device="cuda"), diagonal=1
        )
        attn_bias = attn_bias.masked_fill(upper, float("-inf"))

        inputs = [
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            attn_bias,
            True,
            0.0,
            True,
        ]
        model = RewriteEfficientAttention().cuda()
        pytorch_out = model(*inputs)
        ep = torch.export.export(model, tuple(inputs))
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            decompose_attention=False,
            attn_bias_is_causal=True,
        )
        trt_out = trt_module(*inputs)
        torch.testing.assert_close(pytorch_out, trt_out, rtol=1e-2, atol=1e-2)

    def test_force_causal_efficient_attention_with_non_causal_attn_bias(self):
        class RewriteEfficientAttention(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                attn_bias=None,
                compute_log_sumexp=False,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            ):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    compute_log_sumexp,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return out[0]

        attn_bias = torch.randn(4, 8, 32, 32).cuda()

        inputs = [
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            attn_bias,
            True,
            0.0,
            False,
        ]
        model = RewriteEfficientAttention().cuda()
        pytorch_out = model(*inputs)
        ep = torch.export.export(model, tuple(inputs))
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )
        trt_out = trt_module(*inputs)
        torch.testing.assert_close(pytorch_out, trt_out, rtol=1e-2, atol=1e-2)


class TestConstantDuplication(TestCase):
    def _make_shared_constant_module(self):
        """Module where ``reshape(weight) -> permute`` feeds two distinct matmuls.

        The intermediate ``permute`` is a constant subgraph with two users, the
        case ``constant_duplication`` is designed for.
        """

        class SharedConstantSubgraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("weight", torch.nn.Parameter(torch.randn(8, 4)))

            def forward(self, x, y):
                w = self.weight.reshape(4, 8)
                w = w.permute(1, 0)
                return x @ w + y @ w

        return SharedConstantSubgraph()

    def test_constant_duplication_pass_clones_shared_subgraph(self):
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering.passes.constant_duplication import (
            _compute_constant_nodes,
            _get_impure_targets,
            constant_duplication,
        )

        model = self._make_shared_constant_module().cuda().eval()
        inputs = (torch.randn(3, 8).cuda(), torch.randn(3, 8).cuda())
        ep = torch.export.export(model, inputs)
        gm = ep.module()

        before = _compute_constant_nodes(gm, _get_impure_targets())
        shared = [n for n in before if len(n.users) > 1]
        # Sanity: the test fixture must actually have a shared constant.
        self.assertGreater(
            len(shared),
            0,
            msg="Test fixture has no shared constant subgraph to duplicate.",
        )

        gm = constant_duplication(gm, CompilationSettings(constant_duplication=True))

        after = _compute_constant_nodes(gm, _get_impure_targets())
        remaining_shared = [n for n in after if len(n.users) > 1]
        self.assertEqual(
            len(remaining_shared),
            0,
            msg=(
                "After constant_duplication, no constant node should still have "
                f"multiple users; found: {remaining_shared}"
            ),
        )

    def test_constant_duplication_end_to_end(self):
        model = self._make_shared_constant_module().cuda().eval()
        inputs = (torch.randn(3, 8).cuda(), torch.randn(3, 8).cuda())
        pytorch_out = model(*inputs)
        ep = torch.export.export(model, inputs)
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            constant_duplication=True,
        )
        trt_out = trt_module(*inputs)
        torch.testing.assert_close(pytorch_out, trt_out, rtol=1e-3, atol=1e-3)

    def test_constant_duplication_nested_constants_no_resharing(self):
        """Regression: when a constant subgraph has *multiple* multi-user
        constant nodes (e.g. a shared weight W feeding two shared intermediates
        r2 and r1_t), the clones produced for the outer candidate must
        themselves be classified as constants so the inner candidate's
        duplication does not re-share them.

        We run only the duplication step (no trailing ``constant_fold``) so the
        invariant is checked directly on the post-duplication graph: every
        constant node has exactly one user.
        """
        from torch_tensorrt.dynamo.lowering.passes.constant_duplication import (
            _clone_constant_subgraph,
            _compute_constant_nodes,
            _get_impure_targets,
        )

        class ChainedSharedConstants(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(16, 16))

            def forward(self, x, y, p, q):
                # ``self.w`` has two reshape users → multi-user.
                # ``r1_t`` has two matmul users → multi-user.
                # ``r2`` has two matmul users → multi-user.
                r1 = self.w.reshape(8, 32)
                r1_t = r1.t()  # (32, 8)
                r2 = self.w.reshape(32, 8)  # (32, 8)
                return x @ r1_t + y @ r1_t + p @ r2 + q @ r2

        model = ChainedSharedConstants().cuda().eval()
        inputs = (
            torch.randn(2, 32).cuda(),
            torch.randn(2, 32).cuda(),
            torch.randn(2, 32).cuda(),
            torch.randn(2, 32).cuda(),
        )
        gm = torch.export.export(model, inputs).module()

        constant_nodes = _compute_constant_nodes(gm, _get_impure_targets())
        candidates = [
            n for n in list(gm.graph.nodes) if n in constant_nodes and len(n.users) > 1
        ]
        # Sanity: the fixture must contain a nested chain of multi-user
        # constants for this regression to be meaningful.
        self.assertGreaterEqual(
            len(candidates),
            2,
            msg=f"Test fixture has no nested multi-user constants: {candidates}",
        )

        for node in candidates:
            users = list(node.users.keys())
            for user in users[1:]:
                memo = {}
                new_root = _clone_constant_subgraph(
                    gm, node, user, constant_nodes, memo
                )
                user.replace_input_with(node, new_root)

        # Re-classify and verify no constant node ended up multi-user. Without
        # the fix, an outer-candidate clone (e.g. ``w_dup0``) is reused as-is
        # by an inner candidate's duplication and ends up with 2 users.
        post = _compute_constant_nodes(gm, _get_impure_targets())
        leftovers = [n for n in post if len(n.users) > 1]
        self.assertEqual(
            len(leftovers),
            0,
            msg=(
                "Duplication step left these constants multi-user "
                f"(should be impossible): {[(n.name, len(n.users)) for n in leftovers]}"
            ),
        )

    def test_constant_duplication_many_consumers(self):
        """A constant subgraph with N > 2 consumers should produce N - 1 clones
        (one extra chain per additional consumer), each carrying the original's
        shape / dtype metadata.
        """
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering.passes.constant_duplication import (
            _compute_constant_nodes,
            _get_impure_targets,
            constant_duplication,
        )

        N_CONSUMERS = 5

        class ManyConsumer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8, 4))

            def forward(self, *xs):
                w = self.weight.reshape(4, 8).permute(1, 0)
                return sum(x @ w for x in xs)

        model = ManyConsumer().cuda().eval()
        inputs = tuple(torch.randn(3, 8).cuda() for _ in range(N_CONSUMERS))
        gm = torch.export.export(model, inputs).module()

        # Snapshot the multi-user constant and its shape before the pass so we
        # can compare against the clones.
        before = _compute_constant_nodes(gm, _get_impure_targets())
        shared = [n for n in before if len(n.users) == N_CONSUMERS]
        self.assertEqual(
            len(shared),
            1,
            msg=f"Expected exactly one {N_CONSUMERS}-user constant, got {shared}",
        )
        original = shared[0]
        original_shape = tuple(original.meta["val"].shape)
        original_dtype = original.meta["val"].dtype

        gm = constant_duplication(gm, CompilationSettings(constant_duplication=True))

        # After the pass + internal constant_fold, each consumer's matmul
        # should consume an independent frozen constant of the right shape.
        matmul_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.matmul.default
        ]
        self.assertEqual(
            len(matmul_nodes),
            N_CONSUMERS,
            msg=f"Expected {N_CONSUMERS} matmuls, got {len(matmul_nodes)}",
        )

        seen_constants = set()
        for mm in matmul_nodes:
            const_input = mm.args[1]
            self.assertEqual(
                const_input.op,
                "get_attr",
                msg=(
                    f"Matmul {mm.name} should consume a get_attr after folding, "
                    f"got {const_input.op}={const_input.target}"
                ),
            )
            self.assertEqual(tuple(const_input.meta["val"].shape), original_shape)
            self.assertEqual(const_input.meta["val"].dtype, original_dtype)
            self.assertNotIn(
                const_input.target,
                seen_constants,
                msg="Each matmul should consume its own private frozen constant",
            )
            seen_constants.add(const_input.target)

    def test_constant_duplication_disabled_is_noop(self):
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering.passes.constant_duplication import (
            constant_duplication,
        )

        model = self._make_shared_constant_module().cuda().eval()
        inputs = (torch.randn(3, 8).cuda(), torch.randn(3, 8).cuda())
        ep = torch.export.export(model, inputs)
        gm = ep.module()

        node_count_before = sum(1 for _ in gm.graph.nodes)
        gm = constant_duplication(gm, CompilationSettings(constant_duplication=False))
        node_count_after = sum(1 for _ in gm.graph.nodes)
        self.assertEqual(node_count_before, node_count_after)


if __name__ == "__main__":
    run_tests()
