# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture boundaries, TensorRT placement, and serialization of PyTorch regions."""

import copy
import operator
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes.pass_manager import DynamoPassManager
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    check_decomposed_regions,
    get_region_records,
    is_region_hop,
    normalize_region_scopes,
    region_hop_records,
)
from torch_tensorrt.dynamo.region_compilation._types import REGION_META_KEY
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region._constants import (
    BACKEND_KEY,
    NAME_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    TOKEN_KEY,
)
from torch_tensorrt.region._errors import RegionCaptureError, RegionError
from torch_tensorrt.region._marker_ops import BEGIN, END
from torch_tensorrt.region._session import RegionCompilationSession

from ._utils import capture_regions, compile_regions, run_python, save_and_load


class SharedBlock(torch.nn.Module):
    """All operations have converters; one invocation has an explicit Torch policy."""

    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())

    def forward(self, x):
        h = self.block(x)
        with region.execute_in_torch(name="shared"):
            y = self.block(h)
        return self.block(y) + h


class RepeatedScopes(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        for _ in range(2):
            with region.execute_in_torch(name="repeated"):
                x = torch.relu(x * 1.5)
        return x - 3


class MultipleOutputs(torch.nn.Module):
    def __init__(self, direct_output=False):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))
        self.direct_output = direct_output

    def forward(self, x, y):
        h = x + 1
        with region.execute_in_torch(name="branches"):
            a = torch.relu(h * self.scale)
            b = torch.sigmoid(y - 1)
        return (a, b) if self.direct_output else (a + h, b * 3)


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


class CpuScalarRegion(torch.nn.Module):
    """A literal and a scalar buffer that stay on the CPU, as PyTorch allows."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        with region.execute_in_torch():
            y = torch.relu(x) * self.scale + torch.tensor(0.5)
        return y + 1


class BufferRegion(torch.nn.Module):
    def __init__(self, constant_only=False):
        super().__init__()
        self.register_buffer("bias", torch.randn(8))
        self.constant_only = constant_only

    def forward(self, x):
        folded = self.bias + 2
        with region.execute_in_torch():
            y = torch.relu(self.bias * 2 if self.constant_only else x + folded)
        return x + y if self.constant_only else y * 3


class DeadRegion(torch.nn.Module):
    """Torch-TensorRT decomposes full_like into full(shape), so the region dies."""

    def forward(self, x):
        with region.execute_in_torch(name="shape_only"):
            y = torch.relu(x) * 2
        return torch.full_like(y, 3.0) + x


class ShapeOnlyRegion(DeadRegion):
    """A live region follows the region that decomposition removes."""

    def forward(self, x):
        z = super().forward(x)
        with region.execute_in_torch(name="live"):
            w = torch.sin(z)
        return w + 1


class Fp32Island(torch.nn.Module):
    """An fp16 model that keeps an fp32 RMSNorm in PyTorch."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, x):
        h = x + 1
        with region.execute_in_torch(backend=self.backend):
            f = h.float()
            f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
        return f.half() * 2


class SeededRegion(torch.nn.Module):
    """Operations tagged nondeterministic_seeded that draw no random numbers in eval."""

    def __init__(self, kind, backend="eager"):
        super().__init__()
        self.kind, self.backend = kind, backend
        self.attention = torch.nn.MultiheadAttention(
            8, 2, dropout=0.1, batch_first=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Dropout(0.1),
            torch.nn.Dropout1d(0.1),
            torch.nn.AlphaDropout(0.1),
            torch.nn.FeatureAlphaDropout(0.1),
            torch.nn.RReLU(),
        )
        self.lstm = torch.nn.LSTM(8, 8, batch_first=True)
        self.gru = torch.nn.GRU(8, 8, batch_first=True)
        self.rnn_tanh = torch.nn.RNN(8, 8, batch_first=True)
        self.rnn_relu = torch.nn.RNN(8, 8, nonlinearity="relu", batch_first=True)

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(backend=self.backend):
            if self.kind == "sdpa":
                y = F.scaled_dot_product_attention(x, x, x)
            elif self.kind == "native_dropout":
                # Traced, it returns its input, which a region must not output.
                y = torch.native_dropout(x * 2, 0.1, False)[0]
            elif self.kind == "attention":
                y = self.mlp(self.attention(x, x, x, need_weights=False)[0])
            elif self.kind == "mlp":
                y = self.mlp(x)
            else:
                y = getattr(self, self.kind)(x)[0]
        return y * 2


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
    @parameterized.expand([("token", TOKEN_KEY), ("backend", BACKEND_KEY)])
    def test_membership_loss_is_rejected_instead_of_inferred_from_position(
        self, _, key
    ):
        class InductorRegion(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(backend="inductor"):
                    return torch.relu(x)

        with RegionCompilationSession(strict=True):
            ep = torch.export.export(
                InductorRegion().eval(), (torch.randn(2, 8),), strict=True
            )
        member = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.relu.default
        )
        del member.meta["custom"][key]
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
            with self.assertRaisesRegex(RegionCaptureError, "'cond' around a region"):
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

    @parameterized.expand(
        [
            ("cond_inside", "cond", True, "'cond' inside a region"),
            ("flex_inside", "flex_attention", True, "'flex_attention' inside"),
            ("autocast_inside", "autocast", True, "'wrap_with_autocast' inside"),
            ("autocast_around", "autocast", False, "'wrap_with_autocast' around"),
            ("no_grad_inside", "no_grad", True, "'wrap_with_set_grad_enabled' inside"),
            ("no_grad_around", "no_grad", False, "'wrap_with_set_grad_enabled' around"),
        ]
    )
    def test_higher_order_nesting_is_named(self, _, construct, inside, message):
        class Nested(torch.nn.Module):
            def forward(self, x):
                if construct == "cond":
                    with region.execute_in_torch():
                        y = torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,))
                elif construct == "flex_attention":
                    q = x.view(1, 1, 2, 8)
                    with region.execute_in_torch():
                        y = flex_attention(q, q, q).view(2, 8)
                else:
                    scope = (
                        torch.autocast("cpu", enabled=False)
                        if construct == "autocast"
                        else torch.no_grad()
                    )
                    if inside:
                        with region.execute_in_torch(), scope:
                            y = torch.relu(x)
                    else:
                        with scope, region.execute_in_torch():
                            y = torch.relu(x)
                return y + 1

        model, x = Nested().eval(), torch.randn(2, 8)
        with torch.enable_grad(), RegionCompilationSession(strict=True):
            ep = torch.export.export(model, (x,), strict=True)
        with self.assertRaisesRegex(
            RegionCaptureError, f"{message}.*Compile under torch.no_grad"
        ):
            normalize_region_scopes(ep)
        if construct == "no_grad":
            # Export records a grad-mode block only if it changes the grad mode.
            with torch.no_grad():
                gm = capture_regions(model, x)
            self.assertEqual(len(get_region_records(gm)), 1)
            torch.testing.assert_close(gm(x), model(x))

    @parameterized.expand([("autocast",), ("no_grad",)])
    def test_higher_order_operator_can_read_region_output(self, construct):
        class ReadAfter(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.relu(x)
                scope = (
                    torch.autocast("cpu", enabled=False)
                    if construct == "autocast"
                    else torch.no_grad()
                )
                # The block's body copies y's tags onto its input and cast guard.
                with scope:
                    z = y.float().softmax(-1)
                return z + 1

        model, x = ReadAfter().eval(), torch.randn(2, 8)
        with torch.enable_grad():
            gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

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

    @parameterized.expand(
        [("residual", False, False), ("direct", True, False), ("strided", False, True)]
    )
    def test_multiple_inputs_outputs_and_state(self, _, direct_output, strided):
        x, y = (
            torch.randn(8, 2).t() if strided else torch.randn(2, 8) for _ in range(2)
        )
        model = MultipleOutputs(direct_output=direct_output).eval()
        gm = capture_regions(model, x, y)
        records = get_region_records(gm)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].output_names), 2)
        self.assertEqual(len(records[0].input_names), 3)
        torch.testing.assert_close(gm(x, y), model(x, y))

    @parameterized.expand([("input",), ("interior",), ("output",), ("after",)])
    def test_dtype_casts_in_or_after_scope_are_captured(self, position):
        # Export guards each aten.to with an assert tagged from the cast's input.
        class Cast(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="cast"):
                    if position == "input":
                        y = torch.relu(x.double())
                    elif position == "interior":
                        y = torch.relu(x * 2).double() * 3
                    elif position == "output":
                        y = torch.relu(x * 2).to(torch.float64)
                    else:
                        y = torch.relu(x * 2)
                return y.double() + 1 if position == "after" else y + 1

        model, x = Cast().eval(), torch.randn(2, 8)
        gm = capture_regions(model, x)
        (child,) = region_children(gm)
        self.assertEqual(
            torch.ops.aten.to.dtype in operator_targets(child), position != "after"
        )
        torch.testing.assert_close(gm(x), model(x))

    @parameterized.expand(
        [
            ("eval_dropout", lambda x: F.dropout(x, 0.1, training=False)),
            ("no_op_casts", lambda x: x.float().to(torch.float32)),
        ]
    )
    def test_ops_returning_their_input_do_not_become_outputs(self, _, op):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)

            def forward(self, x):
                with region.execute_in_torch():
                    h = self.fc(op(x))
                # Export binds these reads of x to the op that returned it.
                return x + h, x

        model, x = Residual().eval(), torch.randn(2, 8)
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
                with region.execute_in_torch(name="dead_eval_dropout"):
                    unused = F.dropout(x * 2, 0.1, training=False)
                with region.execute_in_torch(name="returns_input"):
                    x = x.float()
                return x * 2

        x = torch.randn(2, 8)
        model = EmptyAndDead().eval()
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 0)
        torch.testing.assert_close(gm(x), model(x))

    def test_decomposition_may_drop_dead_regions_but_not_change_live_ones(self):
        with RegionCompilationSession(strict=True):
            ep = torch.export.export(
                ShapeOnlyRegion().eval(), (torch.randn(2, 8),), strict=True
            )
        ep = normalize_region_scopes(ep)
        captured = ep.graph_module
        decomposed = ep.run_decompositions(get_decompositions()).graph_module
        self.assertEqual(
            [r.id for r in region_hop_records(captured)], ["shape_only#0", "live#1"]
        )
        self.assertEqual([r.id for r in region_hop_records(decomposed)], ["live#1"])
        check_decomposed_regions(captured, decomposed)

        (hop,) = (node for node in decomposed.graph.nodes if is_region_hop(node))
        hop.kwargs = {"hints": {**hop.kwargs["hints"], "backend": "inductor"}}
        with self.assertRaisesRegex(RegionError, "changed"):
            check_decomposed_regions(captured, decomposed)

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

    @parameterized.expand([("live", True), ("dead", False)])
    def test_tensor_literal_as_region_output(self, _, live):
        class LiteralOutput(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.tensor([1.0, 2.0], device=x.device)
                return y if live else x * 2

        model, x = LiteralOutput().eval(), torch.randn(2)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), int(live))
        torch.testing.assert_close(gm(x), model(x))

    @parameterized.expand(
        [
            ("input", "alias region inputs"),
            ("adjacent_literal", "alias region inputs"),
            ("literal_outputs", "alias each other"),
            ("views", "alias each other"),
            ("chunk", "alias each other"),
        ]
    )
    def test_aliasing_is_rejected_before_decomposition(self, kind, message):
        class Aliased(torch.nn.Module):
            def forward(self, x):
                if kind == "adjacent_literal":
                    with region.execute_in_torch():
                        x = torch.tensor([1.0, 2.0])
                with region.execute_in_torch(name="aliased"):
                    if kind in ("input", "adjacent_literal"):
                        return x.view(-1)
                    y = torch.tensor([1.0, 2.0]) if kind == "literal_outputs" else x + 1
                    if kind == "chunk":
                        return y.chunk(2, dim=-1)
                    if kind == "views":
                        return y.view(2, 8), y.transpose(0, 1)
                    return y, y.view(-1)

        with RegionCompilationSession(strict=True):
            ep = torch.export.export(
                Aliased().eval(), (torch.randn(2, 8),), strict=True
            )
        with self.assertRaisesRegex(RegionCaptureError, f"aliased.*{message}.*clone"):
            normalize_region_scopes(ep)

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

    def test_failed_capture_does_not_leak_session(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            with RegionCompilationSession(strict=True):
                raise RuntimeError("deliberate failure")
        model, x = SharedBlock().eval(), torch.randn(2, 8)
        ep = torch.export.export(model, (x,), strict=True)
        self.assertNotIn(BEGIN, operator_targets(ep.graph_module))
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)

    @parameterized.expand(
        [
            ("plain", False, "Nested region scopes"),
            ("cond_after_inner", True, "'cond' inside a region"),
        ]
    )
    def test_nested_regions_are_rejected(self, _, cond_after_inner, message):
        class Nested(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch(name="outer"):
                    x = x + 1
                    with region.execute_in_torch(name="inner"):
                        x = torch.relu(x)
                    if cond_after_inner:
                        x = torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,))
                return x

        with self.assertRaisesRegex(RegionCaptureError, message):
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

    @parameterized.expand(
        [("chunk",), ("transpose",), ("unbind",), ("graph_inputs",), ("weights",)]
    )
    def test_inputs_sharing_storage_are_read_without_copies(self, kind):
        class SharedStorage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Views of one buffer, like cuDNN-flattened CUDA RNN weights.
                flat = torch.randn(32)
                self.w = torch.nn.Parameter(flat[:16].view(4, 4))
                self.v = torch.nn.Parameter(flat[16:].view(4, 4))

            def forward(self, x, y):
                h = x + 1
                if kind == "chunk":
                    a, b = h.chunk(2, dim=-1)
                elif kind == "transpose":
                    a, b = h, h.t()
                elif kind == "unbind":
                    a, b = h.view(2, 8).unbind(0)
                elif kind == "weights":
                    a, b = self.w, self.v
                else:
                    a, b = x, y
                with region.execute_in_torch():
                    out = torch.sigmoid(a) * b
                return out

        storage = torch.randn(24)
        x, y = storage[:16].view(4, 4), storage[8:].view(4, 4)
        model = SharedStorage().eval()
        gm = capture_regions(model, x, y)
        self.assertEqual(len(get_region_records(gm)), 1)
        # Decomposition needs copies, but the captured body reads the inputs.
        self.assertNotIn(torch.ops.aten.clone.default, operator_targets(gm))
        torch.testing.assert_close(gm(x, y), model(x, y))

    @parameterized.expand(
        [
            ("rand_like", lambda x: x + torch.rand_like(x)),
            ("train_dropout", lambda x: F.dropout(x * 2, 0.5, training=True)),
            (
                "attention_dropout",
                lambda x: F.scaled_dot_product_attention(x, x, x, dropout_p=0.1),
            ),
        ]
    )
    def test_rng_effects_are_rejected(self, _, body):
        class Random(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = body(x)
                return y

        with self.assertRaisesRegex(
            RegionCaptureError, "Random operations.* in 'region#0'"
        ):
            capture_regions(Random(), torch.randn(1, 4, 8))

    @parameterized.expand(
        [
            ("sdpa",),
            ("attention",),
            ("mlp",),
            ("lstm",),
            ("gru",),
            ("rnn_tanh",),
            ("rnn_relu",),
            ("native_dropout",),
        ]
    )
    def test_eval_mode_seeded_operations_are_supported(self, kind):
        model, x = SeededRegion(kind).eval(), torch.randn(1, 4, 8)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestExecuteInTorchIntegration(TestCase):
    @staticmethod
    def _compile(model, inputs, use_fast_partitioner=True, **overrides):
        options = dict(enabled_precisions={torch.float32}, disable_tf32=True)
        options.update(overrides)
        return compile_regions(
            model, inputs, use_fast_partitioner=use_fast_partitioner, **options
        )

    @parameterized.expand(
        [
            (f"{name}_{partition}", model, shape, count, fast)
            for name, model, shape, count in [
                ("shared", SharedBlock, (2, 8), 1),
                ("multiple_outputs", MultipleOutputs, (2, 8), 1),
                ("repeated", RepeatedScopes, (2, 8), 2),
                ("folded_state", BufferRegion, (2, 8), 1),
                ("constant_input", lambda: BufferRegion(constant_only=True), (2, 8), 1),
                ("tensor_literal", TensorLiteralRegion, (2,), 1),
            ]
            for partition, fast in [("fast", True), ("global", False)]
            if fast or name in {"shared", "multiple_outputs", "constant_input"}
        ]
    )
    def test_region_survives_compilation(self, _, factory, shape, count, fast):
        model = factory().eval().cuda()
        inputs = [torch.randn(shape, device="cuda")]
        if isinstance(model, MultipleOutputs):
            inputs.append(torch.randn_like(inputs[0]))
        expected = model(*inputs)
        compiled = self._compile(model, inputs, fast)
        engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
        self.assertTrue(engines)
        if isinstance(model, SharedBlock):
            self.assertEqual(len(engines), 2)
        self.assertEqual(len(get_region_records(compiled)), count)
        torch.testing.assert_close(compiled(*inputs), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_region_left_dead_by_decomposition_is_a_noop(self, _, fast):
        model = ShapeOnlyRegion().eval().cuda()
        x = torch.randn(2, 8, device="cuda")
        compiled = self._compile(model, (x,), fast)
        self.assertEqual([r.id for r in get_region_records(compiled)], ["live#1"])
        torch.testing.assert_close(compiled(x), model(x))

    def test_cpu_scalars_do_not_make_a_region_span_devices(self):
        x = torch.randn(2, 8, device="cuda")
        model = CpuScalarRegion().eval()  # Keep the 0-dim buffer on the CPU.
        compiled = self._compile(model, [x])
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x))
        # A CPU tensor with dimensions still makes the region span devices.
        with self.assertRaisesRegex(
            RegionCaptureError, "'branches#0' cannot span multiple devices: cpu, cuda"
        ):
            capture_regions(MultipleOutputs().cuda(), x, torch.randn(2, 8))

    @parameterized.expand([("eager",), ("inductor",)])
    def test_eval_attention_region_compiles(self, backend):
        if backend == "inductor" and not has_triton():
            self.skipTest("CUDA Inductor requires Triton support on the device")
        model = SeededRegion("attention", backend).eval().cuda()
        x = torch.randn(1, 4, 8, device="cuda")
        compiled = self._compile(model, (x,))
        self.assertTrue(
            any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
        )
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x), atol=5e-3, rtol=5e-3)

    @parameterized.expand([("eager",), ("inductor",)])
    def test_fp32_island_in_fp16_model_compiles(self, backend):
        if backend == "inductor" and not has_triton():
            self.skipTest("CUDA Inductor requires Triton support on the device")
        model = Fp32Island(backend).eval().cuda()
        x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
        compiled = self._compile(model, (x,))
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand([("lstm",), ("gru",)])
    def test_rnn_region_with_flattened_weights_compiles(self, kind):
        model = SeededRegion(kind).eval().cuda()
        weights = list(getattr(model, kind).parameters())
        if not all(torch._C._is_alias_of(weights[0], w) for w in weights[1:]):
            self.skipTest("cuDNN did not flatten the RNN weights into one buffer")
        x = torch.randn(1, 4, 8, device="cuda")
        compiled = self._compile(model, (x,))
        ((owner, call),) = (
            (module, node)
            for module in compiled.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if REGION_META_KEY in node.meta
        )
        operands = [
            operator.attrgetter(arg.target)(owner)
            for arg in call.args
            if arg.op == "get_attr"
        ]
        # Copied weights would make cuDNN repack them on every call.
        self.assertEqual(len(operands), len(weights))
        self.assertTrue(all(torch._C._is_alias_of(weights[0], w) for w in operands))
        torch.testing.assert_close(compiled(x), model(x), atol=5e-3, rtol=5e-3)

    @parameterized.expand([("legacy", False), ("decomposition", True)])
    def test_complex_regions_are_rejected_before_lowering(
        self, _, use_complex_decomposition
    ):
        # Both lowering modes must reject before partitioning.
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
            (f"{setting}_{dryrun=}", setting, dryrun)
            for setting in (
                "require_full_compilation",
                "offload_module_to_cpu",
                "enable_autocast",
                "enable_resource_partitioning",
            )
            for dryrun in (False, True)
        ]
    )
    def test_conflicting_settings_stop_before_unsafe_stages(self, _, setting, dryrun):
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
            if dryrun:
                # Given a path, a dry run also saves its report there.
                directory = stack.enter_context(tempfile.TemporaryDirectory())
                dryrun = str(Path(directory) / "report.txt")
                logs = stack.enter_context(self.assertLogs("torch_tensorrt", "WARNING"))
            else:
                stack.enter_context(
                    self.assertRaisesRegex((RuntimeError, ValueError), setting)
                )
            compiled = self._compile(model, (x,), dryrun=dryrun, **{setting: True})
            if dryrun:
                report = Path(dryrun).read_text()
        if dryrun:
            stops = [line for line in logs.output if "Region dry-run stopped" in line]
            self.assertEqual(len(stops), 1, logs.output)
            self.assertEqual(len(get_region_records(compiled)), 1)
            (conflict,) = compiled.meta["torch_tensorrt_region_conflicts"]
            self.assertIn(f"Dry run stopped before partitioning: {conflict}", report)
            self.assertFalse(
                any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
            )
            torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand(
        [("require_full_compilation",), ("enable_resource_partitioning",)]
    )
    def test_conflicting_dryrun_stops_when_decomposition_drops_the_region(
        self, setting
    ):
        model, x = DeadRegion().eval().cuda(), torch.randn(2, 8, device="cuda")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            compiled = self._compile(model, (x,), dryrun=str(path), **{setting: True})
            report = path.read_text()
        (conflict,) = compiled.meta["torch_tensorrt_region_conflicts"]
        self.assertIn(f"Dry run stopped before partitioning: {conflict}", report)
        torch.testing.assert_close(compiled(x), model(x))

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

    def test_deepcopy_of_compiled_model_saves_and_loads(self):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        expected = model(x)
        copied = copy.deepcopy(self._compile(model, (x,)))
        torch.testing.assert_close(copied(x), expected, atol=5e-3, rtol=5e-3)
        with tempfile.TemporaryDirectory() as directory:
            loaded = save_and_load(copied, Path(directory))
            torch.testing.assert_close(loaded(x), expected, atol=5e-3, rtol=5e-3)

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
            run_python("-c", script, artifact, data_file, expected_engines, timeout=90)


if __name__ == "__main__":
    run_tests()
