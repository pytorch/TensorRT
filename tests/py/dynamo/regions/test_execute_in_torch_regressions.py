# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Extended regression coverage for execute_in_torch.

Start with test_execute_in_torch.py for the beginner examples.
"""

import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_manager import DynamoPassManager
from torch_tensorrt.dynamo.regions import (
    audit_region_placement,
    get_region_records,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region._constants import (
    BACKEND_KEY,
    NAME_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    TOKEN_KEY,
)
from torch_tensorrt.region._errors import RegionCaptureError
from torch_tensorrt.region._marker_ops import BEGIN, END
from torch_tensorrt.region._session import RegionCompilationSession


class SharedBlock(torch.nn.Module):
    """All operations have converters; one invocation has an explicit Torch policy."""

    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())

    def forward(self, x):
        h = self.block(x)
        # "shared": these calls reuse the same module and weights.
        with region.execute_in_torch(name="shared"):
            y = self.block(h)
        return self.block(y) + h


class RepeatedScopes(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        for _ in range(2):
            # "repeated": one annotation used in separate loop occurrences.
            with region.execute_in_torch(name="repeated"):
                x = torch.relu(x * 1.5)
        return x - 3


class MultipleOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x, y):
        h = x + 1
        # "branches": two output computations, not Python control flow.
        with region.execute_in_torch(name="branches"):
            a = torch.relu(h * self.scale)
            b = torch.sigmoid(y - 1)
        return a + h, b * 3


class ComplexRegion(torch.nn.Module):
    def __init__(self, location):
        super().__init__()
        self.location = location

    def forward(self, x):
        if self.location == "parent":
            z = torch.complex(x, x + 1)
            with region.execute_in_torch():
                y = torch.relu(x)
            return y + z.abs()
        with region.execute_in_torch():
            if self.location == "body":
                x = torch.complex(x, x + 1)
            y = x * x
            if self.location == "body":
                y = y.abs()
        return y


class TensorLiteralRegion(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch():
            y = x + torch.tensor([1.0, 2.0], device=x.device)
        return y * 2


def capture_regions(model, *inputs):
    """Exercise the real export/decomposition boundary without building engines."""
    with RegionCompilationSession(strict=True):
        ep = torch.export.export(model.eval(), inputs, strict=True)
        ep = normalize_region_scopes(ep)
        ep = ep.run_decompositions()
        gm = materialize_regions(ep.module())
    gm.graph.lint()
    return gm


def region_children(gm):
    return [gm.get_submodule(record.child_target) for record in get_region_records(gm)]


def operator_targets(gm):
    return [node.target for node in gm.graph.nodes if node.op == "call_function"]


class TestExecuteInTorchEager(TestCase):
    def test_user_exception_is_not_suppressed(self):
        expected = RuntimeError("body failed")
        with self.assertRaises(RuntimeError) as caught:
            with region.execute_in_torch():
                raise expected
        self.assertIs(caught.exception, expected)


class TestExecuteInTorchCapture(TestCase):
    def test_membership_loss_is_rejected_instead_of_inferred_from_position(self):
        with RegionCompilationSession(strict=True):
            ep = torch.export.export(
                SharedBlock().eval(), (torch.randn(2, 8),), strict=True
            )
        member = next(
            node
            for node in ep.graph.nodes
            if TOKEN_KEY in node.meta.get("custom", {}) and node.target != END
        )
        member.meta["custom"] = {
            key: value
            for key, value in member.meta["custom"].items()
            if key != TOKEN_KEY
        }
        with self.assertRaisesRegex(RuntimeError, "membership.*disagree"):
            normalize_region_scopes(ep)

    @parameterized.expand([("top_level", False), ("cond_branch", True)])
    def test_lost_markers_cannot_silently_drop_region_policy(self, _, nested):
        class Conditional(torch.nn.Module):
            def forward(self, x, predicate):
                def true_branch(x):
                    with region.execute_in_torch():
                        return torch.sin(x)

                def false_branch(x):
                    return torch.cos(x)

                return torch.cond(predicate, true_branch, false_branch, (x,))

        model = Conditional() if nested else SharedBlock()
        inputs = (torch.randn(2, 8),) + ((torch.tensor(True),) if nested else ())
        with RegionCompilationSession(strict=True):
            ep = torch.export.export(model.eval(), inputs, strict=True)
        if nested:
            with self.assertRaisesRegex(RegionCaptureError, "higher-order"):
                normalize_region_scopes(ep)
        for child in ep.graph_module.modules():
            if not isinstance(child, torch.fx.GraphModule):
                continue
            for target in (END, BEGIN):
                for node in list(child.graph.nodes):
                    if node.target == target:
                        child.graph.erase_node(node)
            child.recompile()
        with self.assertRaisesRegex(RuntimeError, "without.*markers"):
            normalize_region_scopes(ep)

    def test_only_annotated_call_to_shared_module_is_outlined(self):
        x = torch.randn(2, 8)
        model = SharedBlock().eval()
        gm = capture_regions(model, x)
        children = region_children(gm)
        self.assertEqual(len(children), 1)
        self.assertEqual(
            operator_targets(children[0]).count(torch.ops.aten.relu.default), 1
        )
        self.assertEqual(operator_targets(gm).count(torch.ops.aten.relu.default), 2)
        for module in gm.modules():
            if isinstance(module, torch.fx.GraphModule):
                self.assertNotIn(BEGIN, operator_targets(module))
                self.assertNotIn(END, operator_targets(module))
        torch.testing.assert_close(gm(x), model(x))

    def test_independent_outside_operations_do_not_enter_region(self):
        class Independent(torch.nn.Module):
            def forward(self, x):
                before = torch.sin(x)
                with region.execute_in_torch():
                    inside = torch.relu(x + 2)
                after = torch.cos(x)
                return inside + before + after

        x = torch.randn(2, 8)
        model = Independent().eval()
        gm = capture_regions(model, x)
        child = region_children(gm)[0]
        self.assertNotIn(torch.ops.aten.sin.default, operator_targets(child))
        self.assertNotIn(torch.ops.aten.cos.default, operator_targets(child))
        self.assertIn(torch.ops.aten.sin.default, operator_targets(gm))
        self.assertIn(torch.ops.aten.cos.default, operator_targets(gm))
        torch.testing.assert_close(gm(x), model(x))

    def test_adjacent_repeated_labels_have_distinct_occurrences(self):
        x = torch.randn(2, 8)
        model = RepeatedScopes().eval()
        gm = capture_regions(model, x)
        records = get_region_records(gm)
        self.assertEqual(len(records), 2)
        self.assertEqual({record.name for record in records}, {"repeated"})
        self.assertEqual(len({record.id for record in records}), 2)
        torch.testing.assert_close(gm(x), model(x))

    def test_multiple_inputs_outputs_state_and_residual(self):
        x, y = torch.randn(2, 8), torch.randn(2, 8)
        model = MultipleOutputs().eval()
        gm = capture_regions(model, x, y)
        records = get_region_records(gm)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].output_names), 2)
        self.assertEqual(len(records[0].input_names), 3)
        torch.testing.assert_close(gm(x, y), model(x, y))

    def test_region_outputs_can_be_direct_model_outputs(self):
        class DirectOutput(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.relu(x + 1)
                    z = torch.sigmoid(x)
                return y, z

        model, x = DirectOutput().eval(), torch.randn(2, 8)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    def test_empty_and_dead_scopes_are_noops(self):
        class EmptyAndDead(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="empty"):
                    pass
                with region.execute_in_torch(name="dead"):
                    unused = torch.relu(x)
                return x * 2

        x = torch.randn(2, 8)
        model = EmptyAndDead().eval()
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 0)
        torch.testing.assert_close(gm(x), model(x))

    def test_tensor_literal_constructor_is_supported(self):
        model, x = TensorLiteralRegion().eval(), torch.randn(2)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    @parameterized.expand([("live", True), ("dead", False)])
    def test_literal_normalization_removes_only_capture_metadata(self, _, live):
        class Literal(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = x + torch.tensor([1.0, 2.0])
                return y if live else x * 2

        model, x = Literal().eval(), torch.randn(2)
        with RegionCompilationSession(strict=True):
            ep = torch.export.export(model, (x,), strict=True)
        lifted = next(
            node
            for node in ep.graph.nodes
            if node.op == "placeholder" and TOKEN_KEY in node.meta.get("custom", {})
        )
        lifted.meta["custom"]["application_label"] = "keep"
        normalize_region_scopes(ep)
        self.assertIs(normalize_region_scopes(ep), ep)
        self.assertEqual(lifted.meta["custom"], {"application_label": "keep"})
        capture_keys = {TOKEN_KEY, NAME_KEY, BACKEND_KEY, ROUTE_KEY, PLUGIN_KEY}
        for child in ep.graph_module.modules():
            if isinstance(child, torch.fx.GraphModule):
                for node in child.graph.nodes:
                    self.assertFalse(
                        capture_keys.intersection(node.meta.get("custom", {}))
                    )
        ep = normalize_region_scopes(ep.run_decompositions())
        torch.testing.assert_close(ep.module()(x), model(x))

    def test_tensor_literal_can_be_the_region_output(self):
        class LiteralOutput(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.tensor([1.0, 2.0], device=x.device)
                return y

        model, x = LiteralOutput().eval(), torch.randn(2)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    def test_dead_tensor_literal_scope_is_a_noop(self):
        class DeadLiteral(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    unused = torch.tensor([1.0, 2.0], device=x.device)
                return x * 2

        model, x = DeadLiteral().eval(), torch.randn(2)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 0)
        torch.testing.assert_close(gm(x), model(x))

    def test_tensor_literal_outputs_sharing_storage_are_rejected(self):
        class AliasedLiteralOutputs(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.tensor([1.0, 2.0], device=x.device)
                    z = y.view(2, 1)
                return y, z

        with self.assertRaisesRegex(RegionCaptureError, "alias each other.*clone"):
            capture_regions(AliasedLiteralOutputs(), torch.randn(2))

    def test_view_of_preceding_literal_region_output_is_rejected(self):
        class AdjacentLiteralView(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.tensor([1.0, 2.0], device=x.device)
                with region.execute_in_torch():
                    z = y.view(2, 1)
                return z

        with self.assertRaisesRegex(RegionCaptureError, "alias region inputs.*clone"):
            capture_regions(AdjacentLiteralView(), torch.randn(2))

    @parameterized.expand([("input", False), ("state", True)])
    def test_in_place_detach_of_input_or_state_is_rejected(self, _, use_state):
        class DetachMutation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.ones(2))

            def forward(self, x):
                with region.execute_in_torch():
                    target = self.bias if use_state else x
                    target.detach_()
                    y = target + 1
                return y

        with self.assertRaisesRegex(
            (RegionCaptureError, torch._dynamo.exc.Unsupported), "detach_"
        ):
            capture_regions(DetachMutation(), torch.randn(2))

    @parameterized.expand([("boundary",), ("body",), ("parent",)])
    def test_complex_tensors_anywhere_with_live_regions_are_rejected(self, location):
        dtype = torch.complex64 if location == "boundary" else torch.float32
        with self.assertRaisesRegex(RegionCaptureError, "complex tensors.*regions"):
            capture_regions(ComplexRegion(location), torch.randn(2, dtype=dtype))

    @parameterized.expand([("unannotated", False), ("dead_scopes", True)])
    def test_complex_models_without_live_regions_are_unchanged(self, _, dead_scopes):
        class ComplexNoop(torch.nn.Module):
            def forward(self, x):
                if dead_scopes:
                    with region.execute_in_torch():
                        pass
                    with region.execute_in_torch():
                        unused = x * x
                return x + 1

        model, x = ComplexNoop().eval(), torch.randn(2, dtype=torch.complex64)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 0)
        torch.testing.assert_close(gm(x), model(x))

    def test_noncontiguous_boundary_is_supported(self):
        x, y = torch.randn(8, 2).t(), torch.randn(8, 2).t()
        model = MultipleOutputs().eval()
        gm = capture_regions(model, x, y)
        torch.testing.assert_close(gm(x, y), model(x, y))

    def test_non_strict_capture_fails_explicitly(self):
        with self.assertRaisesRegex(Exception, "strict=True"):
            with RegionCompilationSession(strict=False):
                torch.export.export(
                    SharedBlock().eval(), (torch.randn(2, 8),), strict=False
                )

    def test_failed_capture_does_not_leak_session(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            with RegionCompilationSession(strict=True):
                raise RuntimeError("deliberate failure")
        model, x = SharedBlock().eval(), torch.randn(2, 8)
        ep = torch.export.export(model, (x,), strict=True)
        self.assertNotIn(BEGIN, operator_targets(ep.graph_module))
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)

    def test_nested_regions_are_rejected(self):
        class Nested(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="outer"):
                    x = x + 1
                    with region.execute_in_torch(name="inner"):
                        x = torch.relu(x)
                return x

        with self.assertRaisesRegex(RuntimeError, "[Nn]est|overlap"):
            capture_regions(Nested(), torch.randn(2, 8))

    def test_input_mutation_is_rejected(self):
        class Mutation(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    x.add_(1)
                    y = x * 2
                return y

        with self.assertRaisesRegex(RuntimeError, "[Mm]utat|functional"):
            capture_regions(Mutation(), torch.randn(2, 8))

    def test_alias_escape_is_rejected(self):
        class Aliasing(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = x.view(2, 8)
                return y

        with self.assertRaisesRegex(RegionCaptureError, "alias region inputs.*clone"):
            capture_regions(Aliasing(), torch.randn(2, 8))

    @parameterized.expand([("views", False), ("chunk", True)])
    def test_outputs_sharing_storage_are_rejected_before_decomposition(self, _, chunk):
        class AliasedOutputs(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="aliased_outputs"):
                    y = x + 1
                    if chunk:
                        a, b = y.chunk(2, dim=-1)
                    else:
                        a, b = y.view(2, 8), y.transpose(0, 1)
                return a, b

        with RegionCompilationSession(strict=True):
            ep = torch.export.export(
                AliasedOutputs().eval(), (torch.randn(2, 8),), strict=True
            )
        with self.assertRaisesRegex(
            RegionCaptureError, "aliased_outputs#0.*alias each other.*clone"
        ):
            normalize_region_scopes(ep)

    def test_cloned_chunk_outputs_are_supported(self):
        class ClonedOutputs(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    a, b = (x + 1).chunk(2, dim=-1)
                    a, b = a.clone(), b.clone()
                return a, b

        model, x = ClonedOutputs().eval(), torch.randn(2, 8)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    def test_rng_effects_are_rejected(self):
        class Random(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = x + torch.rand_like(x)
                return y

        with self.assertRaisesRegex(RuntimeError, "RNG|rand_like|nondetermin"):
            capture_regions(Random(), torch.randn(2, 8))

    def test_graph_break_inside_scope_is_rejected(self):
        class GraphBreak(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = x + 1
                    torch._dynamo.graph_break()
                    y = torch.relu(y)
                return y

        with self.assertRaisesRegex(RuntimeError, "[Gg]raph.break"):
            capture_regions(GraphBreak(), torch.randn(2, 8))

    def test_lowering_pass_cannot_erase_placement_policy(self):
        gm = capture_regions(SharedBlock().eval(), torch.randn(2, 8))

        def drop_region_policy(gm, settings):
            for node in gm.graph.nodes:
                node.meta.pop("torch_tensorrt_region", None)
            return gm

        with self.assertRaisesRegex(RuntimeError, "drop_region_policy.*invalidated"):
            DynamoPassManager([drop_region_policy])(gm, CompilationSettings())

    def test_unchanged_id_does_not_hide_a_modified_region_body(self):
        gm = capture_regions(SharedBlock().eval(), torch.randn(2, 8))
        child = region_children(gm)[0]
        activation = next(
            node
            for node in child.graph.nodes
            if node.target == torch.ops.aten.relu.default
        )
        activation.target = torch.ops.aten.sigmoid.default
        child.recompile()
        with self.assertRaisesRegex(RuntimeError, "reference body changed"):
            audit_region_placement(gm, get_region_records(gm))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestExecuteInTorchIntegration(TestCase):
    @staticmethod
    def _compile(model, inputs, use_fast_partitioner=True, **overrides):
        options = dict(
            ir="dynamo",
            strict=True,
            offload_module_to_cpu=False,
            min_block_size=1,
            use_fast_partitioner=use_fast_partitioner,
            enabled_precisions={torch.float32},
            disable_tf32=True,
        )
        options.update(overrides)
        return torch_tensorrt.compile(model, inputs=list(inputs), **options)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_supported_shared_module_stays_torch_only_inside_scope(
        self, _, use_fast_partitioner
    ):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        expected = model(x)
        compiled = self._compile(model, (x,), use_fast_partitioner)
        engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
        self.assertEqual(len(engines), 2)
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_multiple_outputs_and_live_buffer(self, _, use_fast_partitioner):
        model = MultipleOutputs().eval().cuda()
        inputs = (torch.randn(2, 8, device="cuda"), torch.randn(2, 8, device="cuda"))
        expected = model(*inputs)
        compiled = self._compile(model, inputs, use_fast_partitioner)
        self.assertEqual(len(get_region_records(compiled)), 1)
        self.assertTrue(
            any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
        )
        torch.testing.assert_close(compiled(*inputs), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_adjacent_scopes_survive_partitioning(self, _, use_fast_partitioner):
        model, x = RepeatedScopes().eval().cuda(), torch.randn(2, 8, device="cuda")
        compiled = self._compile(model, (x,), use_fast_partitioner)
        self.assertEqual(len(get_region_records(compiled)), 2)
        torch.testing.assert_close(compiled(x), model(x), atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_folded_parent_state_remains_live(self, _, use_fast_partitioner):
        class FoldedState(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(8))

            def forward(self, x):
                folded = self.bias + 2
                with region.execute_in_torch(name="live_constant"):
                    y = torch.relu(x + folded)
                return y * 3

        model, x = FoldedState().eval().cuda(), torch.randn(2, 8, device="cuda")
        expected = model(x)
        compiled = self._compile(model, (x,), use_fast_partitioner)
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_constant_input_region_survives_constant_folding(
        self, _, use_fast_partitioner
    ):
        class ConstantRegion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.randn(8))

            def forward(self, x):
                with region.execute_in_torch(name="constant_body"):
                    y = torch.relu(self.bias * 2)
                return x + y

        model, x = ConstantRegion().eval().cuda(), torch.randn(2, 8, device="cuda")
        expected = model(x)
        compiled = self._compile(model, (x,), use_fast_partitioner)
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_tensor_literal_constructor_compiles(self, _, use_fast_partitioner):
        model = TensorLiteralRegion().eval().cuda()
        x = torch.randn(2, device="cuda")
        compiled = self._compile(model, (x,), use_fast_partitioner)
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand([("legacy", False), ("decomposition", True)])
    def test_complex_regions_are_rejected_before_lowering(
        self, _, use_complex_decomposition
    ):
        # Capture tests cover each complex location. Here verify that both
        # public lowering modes reject before partitioning can begin.
        model = ComplexRegion("parent").eval().cuda()
        x = torch.randn(2, device="cuda")
        with self.assertRaisesRegex(RegionCaptureError, "complex tensors.*regions"):
            self._compile(
                model,
                (x,),
                use_complex_decomposition=use_complex_decomposition,
            )

    @parameterized.expand(
        [
            (
                "full_compilation",
                {"require_full_compilation": True},
                "full_compilation",
            ),
            ("offload", {"offload_module_to_cpu": True}, "offload_module_to_cpu"),
            ("autocast", {"enable_autocast": True}, "autocast"),
            ("resource", {"enable_resource_partitioning": True}, "resource"),
        ]
    )
    def test_conflicting_settings_are_rejected(self, _, options, message):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex((RuntimeError, ValueError), message):
            self._compile(model, (x,), **options)

    def test_public_non_strict_capture_fails_explicitly(self):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "strict=True"):
            self._compile(model, (x,), strict=False)

    def test_training_model_is_rejected(self):
        model, x = SharedBlock().train().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self._compile(model, (x,))

    def test_dynamic_shape_boundary_is_rejected(self):
        model = SharedBlock().eval().cuda()
        dynamic_input = torch_tensorrt.Input(
            min_shape=(1, 8),
            opt_shape=(2, 8),
            max_shape=(4, 8),
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "static shapes|dynamic"):
            self._compile(model, (dynamic_input,))

    def test_engine_only_output_is_rejected(self):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "serialized-engine-only"):
            torch_tensorrt.convert_method_to_trt_engine(
                model,
                inputs=[x],
                ir="dynamo",
                strict=True,
                offload_module_to_cpu=False,
            )

    @parameterized.expand(
        [
            ("torchscript", {"output_format": "torchscript"}),
            ("retrace", {"retrace": True}),
            ("new_exporter", {"use_legacy_exporter": False}),
        ]
    )
    def test_unqualified_save_modes_are_rejected(self, _, overrides):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        # The early return supplies a real region-bearing module without building
        # engines, keeping this API validation independent of serialization.
        compiled = self._compile(model, (x,), min_block_size=100)
        options = dict(
            output_format="exported_program", retrace=False, use_legacy_exporter=True
        )
        options.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "unsupported.ep"
            with self.assertRaisesRegex(RuntimeError, "Saving annotated regions"):
                torch_tensorrt.save(compiled, str(destination), inputs=[x], **options)
            self.assertFalse(destination.exists())

    @parameterized.expand(
        [
            ("fast_hybrid", True, False),
            ("global_hybrid", False, False),
            ("direct_torch", False, True),
        ]
    )
    def test_qualified_save_load_in_fresh_process(
        self, _, use_fast_partitioner, all_torch
    ):
        class DirectTorch(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="direct"):
                    y = torch.relu(x + 1)
                return y

        model = (DirectTorch() if all_torch else SharedBlock()).eval().cuda()
        x = torch.randn(2, 8, device="cuda")
        expected = model(x)
        compiled = self._compile(model, (x,), use_fast_partitioner)
        expected_engines = 0 if all_torch else 2
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "region.ep"
            data_file = Path(directory) / "inputs.pt"
            torch_tensorrt.save(
                compiled,
                str(artifact),
                inputs=[x],
                output_format="exported_program",
                retrace=False,
                use_legacy_exporter=True,
            )
            # Saving must not inline or alter the caller's compiled module.
            self.assertEqual(len(get_region_records(compiled)), 1)
            audit_region_placement(compiled, get_region_records(compiled))
            torch.testing.assert_close(compiled(x), expected, atol=5e-3, rtol=5e-3)
            torch.save(
                {"input": x.cpu(), "expected": expected.detach().cpu()}, data_file
            )
            script = """
import sys
import torch
import torch_tensorrt
from torch_tensorrt.region._session import current_session

assert current_session() is None
assert not hasattr(torch.ops.torch_tensorrt_region, "begin")
loaded = torch_tensorrt.load(sys.argv[1])
data = torch.load(sys.argv[2], weights_only=True)
actual = loaded.module()(data["input"].cuda())
assert actual.device.type == "cuda"
torch.testing.assert_close(actual.cpu(), data["expected"], atol=5e-3, rtol=5e-3)
engine_calls = [node for node in loaded.graph.nodes if "execute_engine" in str(node.target)]
assert len(engine_calls) == int(sys.argv[3]), (len(engine_calls), sys.argv[3])
assert not hasattr(torch.ops.torch_tensorrt_region, "begin")
assert loaded.graph_module.meta["custom"]["torch_tensorrt_regions"]
"""
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(artifact),
                    str(data_file),
                    str(expected_engines),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @parameterized.expand(
        [
            ("full_compilation", {"require_full_compilation": True}),
            ("offload", {"offload_module_to_cpu": True}),
            ("autocast", {"enable_autocast": True}),
            ("resource", {"enable_resource_partitioning": True}),
        ]
    )
    def test_dryrun_reports_conflicts_without_running_unsafe_stages(self, _, options):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        forbidden_stages = (
            "torch_tensorrt.dynamo._compiler.convert_module",
            "torch_tensorrt.dynamo._compiler.deallocate_module",
            "torch_tensorrt.dynamo._compiler.resource_partition",
            "torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass.trace_intermediate_node_outputs",
        )
        with ExitStack() as stack:
            for target in forbidden_stages:
                stack.enter_context(
                    mock.patch(
                        target, side_effect=AssertionError(f"Unexpected {target}")
                    )
                )
            compiled = self._compile(model, (x,), dryrun=True, **options)
        self.assertEqual(len(get_region_records(compiled)), 1)
        self.assertTrue(compiled.meta.get("torch_tensorrt_region_conflicts"))
        self.assertFalse(
            any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
        )
        torch.testing.assert_close(compiled(x), model(x))


if __name__ == "__main__":
    run_tests()
