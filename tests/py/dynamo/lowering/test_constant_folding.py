# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import copy
import itertools
import threading
import types
import unittest
from unittest import mock

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes import (
    _aten_lowering_pass,
    complex_decomposition_adapter,
    constant_folding,
)
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold

# PyTorch before 2.9, as on JetPack, has only the older allow_tf32 flags.
_HAS_FP32_PRECISION = hasattr(torch.backends.cuda.matmul, "fp32_precision")


def _attention(x, mask):
    query, key = x[..., :8], x.transpose(-1, -2)[..., :8]
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, key, attn_mask=mask
    )


def _same_batch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != y.shape[0]:
        raise ValueError("same_batch_add needs equal batch sizes")
    return x + y


torch.library.custom_op(
    "constant_folding_test::same_batch_add", mutates_args=(), tags=torch.Tag.pointwise
)(_same_batch_add).register_fake(_same_batch_add)


class _BroadcastBias(torch.nn.Module):
    """An attention bias computed from small buffers and broadcast over the batch."""

    def __init__(self, batch, reader="attention", broadcast="expand"):
        super().__init__()
        self.batch, self.reader, self.broadcast = batch, reader, broadcast
        self.register_buffer("reference", torch.randn(4, 32))
        self.register_buffer("frequency", torch.randn(512))
        self.register_buffer("scale", torch.tensor(8.0))

    def forward(self, x):
        bias = torch.sin(self.reference[:, :, None] * self.frequency)[None]
        if self.broadcast == "repeat":
            bias = bias.repeat(self.batch, 1, 1, 1)
        elif self.broadcast == "cat":
            bias = torch.cat([bias] * self.batch)
        else:
            bias = bias.expand(self.batch, 4, 32, 512)
            if self.broadcast == "clone":
                bias = bias.clone()
        if self.reader == "add":
            return x + bias
        if self.reader.startswith("scores"):
            query, key = x[..., :8], x.transpose(-1, -2)[..., :8]
            scores = query @ key.transpose(-1, -2)
            if self.reader == "scores_sum":
                return (scores + bias).sum(-1)
            if self.reader == "scores_matmul":
                return (scores + bias) @ key
            if self.reader == "scores_view_matmul":
                return (scores.view(64, 32, 512).view(16, 4, 32, 512) + bias) @ key
            if self.reader == "scores_plus_bias":
                return torch.softmax(scores * 0.125 + (x + bias), -1) @ key
            if self.reader == "scores_plus_bias_sum":
                return (scores * 0.125 + (x + bias)).sum(-1)
            if self.reader == "scores_tensor_scale":
                return torch.softmax(scores / self.scale + bias, -1) @ key
            if self.reader == "scores_upcast":
                # What a cast before the softmax becomes after the decompositions.
                upcast = torch.ops.aten._to_copy.default(
                    scores * 0.125 + bias, dtype=torch.float64
                )
                return torch.softmax(upcast, -1).float() @ key
            return torch.softmax(scores * 0.125 + bias, -1) @ key
        if self.reader == "matmul_after":
            return (x + bias) @ x[0, 0, :8].t()
        if self.reader == "matmul_after_views":
            moved = (x + bias).transpose(0, 1)
            cast = torch.ops.aten._to_copy.default(moved, dtype=x.dtype)
            return cast @ x[0, 0, :8].t()
        if self.reader == "pointwise_after_views":
            return torch.relu((x + bias).transpose(0, 1))
        if self.reader == "permuted_matmul_after":
            moved = x.transpose(0, 1) + bias.permute(1, 0, 2, 3)
            return (x + bias) @ x[0, 0, :8].t(), moved @ x[0, 0, :8].t()
        if self.reader == "pointwise_before_matmul":
            return (torch.relu(x) + bias) @ x[0, 0, :8].t()
        if self.reader == "softmax":
            return torch.softmax(x + bias, -1)
        if self.reader == "chain":
            return torch.relu(x + bias) * 0.5
        if self.reader == "where_softmax":
            return torch.softmax(torch.where(x > 0, bias, x), -1)
        if self.reader == "where_chain":
            return torch.relu(torch.where(x > 0, bias, x))
        if self.reader == "where_view":
            return torch.softmax(torch.where(x > 0, bias, x).view(16, 4, 32 * 512), -1)
        if self.reader == "sum":
            return _attention(x, bias.sum(0))
        if self.reader == "cat":
            return torch.cat([x, bias], -1)[..., ::2]
        if self.reader == "slice":
            return _attention(x, bias), x * bias[3:4]
        if self.reader.startswith("permute"):
            permuted = bias.permute(1, 0, 2, 3)
            if self.reader == "permute_cat":
                return _attention(x, bias), torch.cat([x.transpose(0, 1), permuted], -1)
            out = _attention(x, bias)
            out = out + _attention(x.transpose(0, 1), permuted).transpose(0, 1)
            # A reduction of the permute is folded on its own.
            if self.reader == "permute_sum":
                return out + permuted.sum(-1, keepdim=True).transpose(0, 1)
            return out
        if self.reader == "efficient_attention":
            query, key = x[..., :8], x.transpose(-1, -2)[..., :8]
            return torch.ops.aten._scaled_dot_product_efficient_attention(
                query, key, key, bias, False
            )[0]
        return _attention(x, bias)


class _ExpandedTable(torch.nn.Module):
    """An attention mask expanded from a parameter or a buffer."""

    def __init__(self, table, as_parameter):
        super().__init__()
        if as_parameter:
            self.table = torch.nn.Parameter(table)
        else:
            self.register_buffer("table", table)

    def forward(self, x):
        return _attention(x, self.table.expand(16, 4, 32, 512))


class _Permute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 512, 513))

    def forward(self, x):
        return x + self.weight.permute(0, 2, 1)


class _Squeeze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 512, 513))

    def forward(self, x):
        return x + self.weight.squeeze()


class _AsStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(512, 513))

    def forward(self, x):
        return x + torch.as_strided(self.weight, size=(512, 513), stride=(513, 1))


class _Select(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(512, 513))

    def forward(self, x):
        return x + self.weight[0]


@torch.library.custom_op("constant_folding_test::quantize", mutates_args=())
def _quantize(x: torch.Tensor) -> torch.Tensor:
    raise AssertionError("constant folding ran the quantize op")


@_quantize.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class TestConstantFolding(TestCase):
    def _export_and_fold(self, model, example_input):
        exported_program = torch.export.export(model, (example_input,))
        return constant_fold(exported_program.module(), CompilationSettings())

    def _assert_installed(self, gm, op):
        self.assertNotIn(op, [node.target for node in gm.graph.nodes])
        self.assertTrue(
            any(name.startswith("_frozen_param") for name, _ in gm.named_parameters())
        )

    def test_folded_constants_are_deepcopy_safe_parameters(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x + self.weight.sin()

        example_input = (torch.randn(4, 4),)

        for offload_module_to_cpu in (False, True):
            with self.subTest(offload_module_to_cpu=offload_module_to_cpu):
                exported_program = torch.export.export(Model(), example_input)
                gm = exported_program.module()
                gm = constant_fold(
                    gm,
                    CompilationSettings(offload_module_to_cpu=offload_module_to_cpu),
                )

                frozen_parameters = [
                    parameter
                    for name, parameter in gm.named_parameters()
                    if name.startswith("_frozen_param")
                ]
                self.assertEqual(len(frozen_parameters), 1)

                frozen_parameter = frozen_parameters[0]
                self.assertIsInstance(frozen_parameter, torch.nn.Parameter)
                self.assertFalse(frozen_parameter.requires_grad)
                self.assertIsNone(frozen_parameter.grad_fn)

                copy.deepcopy(gm)

    def test_repeats_of_an_op_on_a_constant_fold_once(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                x = x + self.weight.sin()
                for _ in range(3):
                    x = x @ self.weight.sin().t()
                # Returned, so it must stay a tensor of its own.
                return x, self.weight.sin()

        model = Model()
        example_input = torch.randn(4, 4)
        for offload_module_to_cpu in (False, True):
            with self.subTest(offload_module_to_cpu=offload_module_to_cpu):
                gm = constant_fold(
                    torch.export.export(model, (example_input,)).module(),
                    CompilationSettings(offload_module_to_cpu=offload_module_to_cpu),
                )
                frozen = [
                    name
                    for name, _ in gm.named_parameters()
                    if name.startswith("_frozen_param")
                ]
                # The sine, its transpose, and the returned sine.
                self.assertEqual(len(frozen), 3)
                torch.testing.assert_close(gm(example_input), model(example_input))

    def _frozen_shapes(self, gm):
        return [
            tuple(getattr(gm, node.target).shape)
            for node in gm.graph.nodes
            if node.op == "get_attr" and node.target.startswith("_frozen_param")
        ]

    def _assert_frozen_storage_fits(self, gm):
        for name, value in gm.named_parameters():
            if name.startswith("_frozen_param"):
                self.assertEqual(
                    value.untyped_storage().nbytes(),
                    value.numel() * value.element_size(),
                )

    def test_folded_broadcast_installs_only_its_source(self):
        example_input = torch.randn(16, 4, 32, 512)
        for broadcast, offload_module_to_cpu in itertools.product(
            ("expand", "clone", "repeat", "cat"), (False, True)
        ):
            model = _BroadcastBias(16, broadcast=broadcast)
            with self.subTest(broadcast=broadcast, offload=offload_module_to_cpu):
                gm = constant_fold(
                    torch.export.export(model, (example_input,)).module(),
                    CompilationSettings(offload_module_to_cpu=offload_module_to_cpu),
                )
                self.assertEqual(self._frozen_shapes(gm), [(1, 4, 32, 512)])
                self._assert_frozen_storage_fits(gm)
                torch.testing.assert_close(gm(example_input), model(example_input))

    def test_folded_broadcast_slices_and_permutes_read_its_source(self):
        example_input = torch.randn(16, 4, 32, 512)
        source = (1, 4, 32, 512)
        for (reader, shapes), offload_module_to_cpu in itertools.product(
            (
                ("slice", [source]),
                ("permute", [source]),
                ("permute_sum", [source, (16, 4, 32, 1)]),
            ),
            (False, True),
        ):
            model = _BroadcastBias(16, reader, "repeat")
            with self.subTest(reader=reader, offload=offload_module_to_cpu):
                gm = constant_fold(
                    torch.export.export(model, (example_input,)).module(),
                    CompilationSettings(offload_module_to_cpu=offload_module_to_cpu),
                )
                self.assertEqual(self._frozen_shapes(gm), shapes)
                torch.testing.assert_close(gm(example_input), model(example_input))

    def test_folded_broadcast_read_by_a_pointwise_op_before_another_kernel(self):
        example_input = torch.randn(16, 4, 32, 512)
        for reader in (
            "scores_softmax",
            "scores_tensor_scale",
            "scores_upcast",
            "scores_plus_bias",
            "matmul_after",
            "matmul_after_views",
            "permuted_matmul_after",
            "where_softmax",
        ):
            model = _BroadcastBias(16, reader, "repeat")
            with self.subTest(reader=reader):
                gm = self._export_and_fold(model, example_input)
                self.assertEqual(self._frozen_shapes(gm), [(1, 4, 32, 512)])
                torch.testing.assert_close(gm(example_input), model(example_input))

    def test_folded_constant_installs_one_slice_of_each_repeated_dim(self):
        class Mask(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("keep", torch.rand(16, 512) > 0.5)

            def forward(self, x):
                mask = self.keep[:, None, None, :].expand(16, 4, 32, 512).clone()
                return _attention(x, torch.where(mask, 0.0, -1e4))

        class Zeros(torch.nn.Module):
            def __init__(self, negative):
                super().__init__()
                zeros = torch.zeros(16, 4, 32, 512)
                if negative:
                    zeros[1] = -0.0
                self.register_buffer("zeros", zeros)

            def forward(self, x):
                return _attention(x, self.zeros * 2)

        example_input = torch.randn(16, 4, 32, 512)
        # Bits are compared, so -0.0 does not count as a repeat of 0.0.
        for model, shape in (
            (Mask(), (16, 1, 1, 512)),
            (Zeros(negative=False), (1, 1, 1, 1)),
            (Zeros(negative=True), (16, 1, 1, 1)),
        ):
            with self.subTest(shape=shape):
                gm = self._export_and_fold(model, example_input)
                self.assertEqual(self._frozen_shapes(gm), [shape])
                torch.testing.assert_close(gm(example_input), model(example_input))

    def test_small_or_reduced_broadcast_folds_as_before(self):
        # Saves under 1 MiB, folds to a value that is not a broadcast, or has a reader other
        # than the attention mask, a pointwise op that TensorRT runs as fast on a broadcast, a
        # slice, or a permute read only by those.
        x = torch.randn(16, 4, 32, 512)
        for model, example_input, shape in (
            (_BroadcastBias(4), torch.randn(4, 4, 32, 512), (4, 4, 32, 512)),
            (_BroadcastBias(16, "sum"), x, (4, 32, 512)),
            (_BroadcastBias(16, "add"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "scores_sum"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "pointwise_before_matmul"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "pointwise_after_views"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "scores_matmul"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "scores_view_matmul"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "scores_plus_bias_sum"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "softmax"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "chain"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "where_chain"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "where_view"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "cat"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "efficient_attention"), x, (16, 4, 32, 512)),
            (_BroadcastBias(16, "permute_cat"), x, (16, 4, 32, 512)),
        ):
            with self.subTest(reader=model.reader, shape=shape):
                gm = self._export_and_fold(model, example_input)
                self.assertEqual(self._frozen_shapes(gm), [shape])

    def test_folded_broadcast_keeps_full_install_when_needed(self):
        class Table(torch.nn.Module):
            def __init__(self, table):
                super().__init__()
                self.register_buffer("table", table)

        class AttentionAndMatmul(Table):
            # Two copies of one op share a folded value. A matmul reads one, so neither takes the slice.
            def forward(self, x, y):
                return _attention(x, self.table.expand(16, 4, 32, 512)), (
                    self.table.expand(16, 4, 32, 512) @ y
                )

        class Returned(Table):
            def forward(self, x, y):
                bias = torch.sin(self.table).repeat(16, 1, 1, 1)
                return _attention(x, bias), bias[3:4]

        class LaterColumnDiffers(Table):
            def forward(self, x, y):
                return _attention(x, self.table * 2)

        class SliceOfManyCopies(Table):
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512)
                return _attention(x, bias), x[..., :256] * bias[..., :256]

        class PermuteReadByMul(Table):
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512)
                return _attention(x, bias), x.transpose(0, 1) * bias.permute(1, 0, 2, 3)

        class EqualShapesOnly(Table):
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512) * 1
                return torch.softmax(torch.ops.prims.maximum(bias, x), -1)

        class PermuteReadByEqualShapesOnly(Table):
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512) * 1
                larger = torch.ops.prims.maximum(
                    bias.permute(1, 0, 2, 3), x.transpose(0, 1)
                )
                return _attention(x, bias), torch.softmax(larger, -1)

        class WhereOfItsOwnPermute(Table):
            # Read directly and through the permute, both reads would take the slice.
            def forward(self, x, y):
                bias = self.table.expand(16, 16, 32, 512)
                chosen = torch.where(y.sum() > 0, bias, bias.permute(1, 0, 2, 3))
                return torch.softmax(chosen, -1)

        class RepeatsReadByOneOp(Table):
            # Each could take the slice, but the op would shrink if both did.
            def forward(self, x, y):
                first, second = (self.table.expand(16, 4, 32, 512) for _ in range(2))
                return torch.softmax(torch.where(y.sum() > 0, first, second), -1)

        class PermutesOfRepeatsReadByOneOp(Table):
            def forward(self, x, y):
                first, second = (self.table.expand(16, 4, 32, 512) for _ in range(2))
                chosen = torch.where(
                    y.sum() > 0, first.permute(1, 0, 2, 3), second.permute(1, 0, 2, 3)
                )
                return _attention(x, first), _attention(x, second), chosen.softmax(-1)

        class RepeatsAroundAnotherValue(Table):
            # The other value is decided between the two repeats, seeing only the first shrunk.
            def __init__(self, table):
                super().__init__(table)
                self.register_buffer("other", torch.randn(1, 4, 32, 512))

            def forward(self, x, y):
                first = self.table.expand(16, 4, 32, 512)
                out = _attention(x, first)
                other = self.other.expand(16, 4, 32, 512)
                second = self.table.expand(16, 4, 32, 512)
                out = out + _attention(x, other) + _attention(x, second)
                chosen = torch.where(
                    y.sum() > 0, second.permute(1, 0, 2, 3), other.permute(1, 0, 2, 3)
                )
                return out, chosen.softmax(-1)

        class PointwiseOpWhoseFakeKernelRaises(Table):
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512)
                added = torch.ops.constant_folding_test.same_batch_add(bias, x)
                return _attention(x, bias), added.softmax(-1)

        class SliceOfDataDependentLength(Table):
            def forward(self, x, y):
                length = (y[0, :2] > 0).sum().item() + 1
                torch._check(length <= 16)
                bias = self.table.expand(16, 4, 32, 512)
                return _attention(x, bias), bias[:length] * 2

        class PermuteOfAPermuteLeftInPlace(Table):
            # The second permute is a large view of the buffer, so it stays in the graph.
            def forward(self, x, y):
                bias = self.table.expand(16, 4, 32, 512)
                permuted = bias.permute(1, 0, 2, 3)
                out = _attention(x, bias)
                out = out + _attention(x.transpose(0, 1), permuted).transpose(0, 1)
                return out, torch.cat([x, permuted.permute(1, 0, 2, 3)], -1)

        x = torch.randn(16, 4, 32, 512)
        y = torch.randn(512, 8)
        full = (16, 4, 32, 512)
        # Repeats in the corner the first check reads, but not in column 100 of copy 5.
        table = torch.randn(1, 4, 32, 512).expand(full).clone()
        table[5, ..., 100] += 1
        for model, shapes in (
            (AttentionAndMatmul(torch.randn(1, 4, 32, 512)), [full]),
            (Returned(torch.randn(1, 4, 32, 512)), [full]),
            (LaterColumnDiffers(table), [full]),
            # The slice is folded on its own.
            (SliceOfManyCopies(torch.randn(1, 4, 32, 512)), [full, (16, 4, 32, 256)]),
            (PermuteReadByMul(torch.randn(1, 4, 32, 512)), [full]),
            (EqualShapesOnly(torch.randn(1, 4, 32, 512)), [full]),
            (PermuteReadByEqualShapesOnly(torch.randn(1, 4, 32, 512)), [full]),
            (WhereOfItsOwnPermute(torch.randn(1, 1, 32, 512)), [(16, 16, 32, 512)]),
            (RepeatsReadByOneOp(torch.randn(1, 4, 32, 512)), [full]),
            (PermutesOfRepeatsReadByOneOp(torch.randn(1, 4, 32, 512)), [full]),
            (
                RepeatsAroundAnotherValue(torch.randn(1, 4, 32, 512)),
                [(1, 4, 32, 512), full],
            ),
            (SliceOfDataDependentLength(torch.randn(1, 4, 32, 512)), [full]),
            (PointwiseOpWhoseFakeKernelRaises(torch.randn(1, 4, 32, 512)), [full]),
            (PermuteOfAPermuteLeftInPlace(torch.randn(1, 4, 32, 512)), [full]),
        ):
            with self.subTest(model=type(model).__name__):
                gm = constant_fold(
                    torch.export.export(model, (x, y)).module(), CompilationSettings()
                )
                self.assertEqual(sorted(self._frozen_shapes(gm)), sorted(shapes))
                torch.testing.assert_close(gm(x, y), model(x, y))

    def test_refittable_engine_keeps_folded_broadcasts_at_full_shape(self):
        x = torch.randn(16, 4, 32, 512)
        for as_parameter, table in itertools.product(
            (True, False),
            (torch.randn(4, 32, 512), torch.randn(32, 512).expand(4, 32, 512)),
        ):
            with self.subTest(as_parameter=as_parameter, strides=table.stride()):
                model = _ExpandedTable(table, as_parameter)
                gm = constant_fold(
                    torch.export.export(model, (x,)).module(),
                    CompilationSettings(immutable_weights=False),
                )
                self.assertEqual(self._frozen_shapes(gm), [(16, 4, 32, 512)])
                torch.testing.assert_close(gm(x), model(x))

    def test_folded_broadcast_skips_sparse_constants(self):
        class Sparse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                dense = torch.zeros(16, 4, 32, 512)
                dense[..., ::64] = 1.0
                self.register_buffer("mask", dense.to_sparse())

            def forward(self, x):
                return _attention(x, self.mask * 2)

        model = Sparse()
        x = torch.randn(16, 4, 32, 512)
        gm = constant_fold(
            torch.export.export(model, (x,)).module(), CompilationSettings()
        )
        torch.testing.assert_close(gm(x), model(x))

    def test_reader_walk_is_linear_in_graph_size(self):
        # A chain longer than the recursion limit, then a graph whose paths double each step.
        graph = torch.fx.Graph()
        start = node = graph.placeholder("x")
        for _ in range(1500):
            node = graph.call_function(torch.ops.aten.mul.Tensor, (node, 0.5))
        for _ in range(40):
            sine = graph.call_function(torch.ops.aten.sin.default, (node,))
            node = graph.call_function(torch.ops.aten.add.Tensor, (node, sine))
        graph.output(node)
        lookups = collections.Counter()

        class Replacements(dict):
            # Fails at once, as a walk that follows every path would take 2**40 steps.
            def __contains__(self, key):
                lookups[key] += 1
                if lookups[key] > 2:
                    raise AssertionError(f"{key} was looked up a third time")
                return super().__contains__(key)

        reader = next(iter(start.users))
        self.assertTrue(
            constant_folding._folded_away(reader, Replacements({node: None}))
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.refit,
        "CUDA and refit required",
    )
    def test_refit_of_a_folded_broadcast(self):
        dense = lambda: torch.randn(4, 32, 512, device="cuda")
        repeated = lambda: torch.randn(32, 512, device="cuda").expand(4, 32, 512)
        x = torch.randn(16, 4, 32, 512, device="cuda")
        for as_parameter, (build, refit) in itertools.product(
            (True, False), ((repeated, dense), (dense, repeated), (dense, dense))
        ):
            with self.subTest(
                as_parameter=as_parameter, build=build is dense, refit=refit is dense
            ):
                compiled = torch_tensorrt.dynamo.compile(
                    torch.export.export(_ExpandedTable(build(), as_parameter), (x,)),
                    inputs=[x],
                    min_block_size=1,
                    immutable_weights=False,
                )
                new_model = _ExpandedTable(refit(), as_parameter)
                refitted = torch_tensorrt.dynamo.refit_module_weights(
                    compiled_module=compiled,
                    new_weight_module=torch.export.export(new_model, (x,)),
                    arg_inputs=[x],
                )
                torch.testing.assert_close(
                    refitted(x), new_model(x), atol=2e-3, rtol=2e-3
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_folded_broadcast_compiles_end_to_end(self):
        cases = (
            ("attention", torch.float32, 2e-3),
            ("attention", torch.float16, 1e-2),
            ("slice", torch.float32, 2e-3),
            ("permute", torch.float16, 1e-2),
            ("scores_softmax", torch.float16, 1e-2),
            ("scores_plus_bias", torch.float16, 1e-2),
            # Its outputs reach 500, where TensorRT may round a few fp16 steps apart from eager.
            ("permuted_matmul_after", torch.float16, 1e-1),
        )
        installed = []

        def fold_and_record(gm, settings):
            gm = constant_fold(gm, settings)
            installed.append(self._frozen_shapes(gm))
            return gm

        passes = _aten_lowering_pass.ATEN_POST_LOWERING_PASSES.passes
        for reader, dtype, tolerance in cases:
            with (
                self.subTest(reader=reader, dtype=dtype),
                mock.patch.object(
                    _aten_lowering_pass.ATEN_POST_LOWERING_PASSES,
                    "passes",
                    [fold_and_record if p is constant_fold else p for p in passes],
                ),
            ):
                installed.clear()
                model = _BroadcastBias(16, reader, "repeat").to("cuda", dtype)
                example_input = torch.randn(16, 4, 32, 512, device="cuda", dtype=dtype)
                compiled_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, (example_input,)),
                    inputs=[example_input],
                    min_block_size=1,
                )
                self.assertEqual(installed, [[(1, 4, 32, 512)]])
                torch.testing.assert_close(
                    compiled_model(example_input),
                    model(example_input),
                    atol=tolerance,
                    rtol=tolerance,
                )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
        "TF32 needs an Ampere or newer GPU",
    )
    @unittest.skipUnless(_HAS_FP32_PRECISION, "needs the fp32_precision flags")
    def test_float32_folds_follow_disable_tf32(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("position", torch.randn(1, 64, 32, 32))
                self.proj = torch.nn.Conv2d(64, 96, 1)
                self.register_buffer("table", torch.randn(32, 64))
                self.register_buffer("mix", torch.randn(64, 32))

            def forward(self, x):
                return x + self.proj(self.position) + self.table @ self.mix

        torch.manual_seed(0)
        model = Model().cuda().eval()
        example_input = torch.randn(1, 96, 32, 32, device="cuda")
        expected_conv = torch.nn.functional.conv2d(
            model.position.double(),
            model.proj.weight.double(),
            model.proj.bias.double(),
        )
        expected_matmul = model.table.double() @ model.mix.double()
        conv, matmul = torch.backends.cudnn.conv, torch.backends.cuda.matmul
        saved = conv.fp32_precision, matmul.fp32_precision
        for disable_tf32 in (True, False):
            with self.subTest(disable_tf32=disable_tf32):
                conv.fp32_precision = matmul.fp32_precision = "tf32"
                try:
                    with mock.patch.object(
                        constant_folding,
                        "_ieee_float32",
                        wraps=constant_folding._ieee_float32,
                    ) as ieee_float32:
                        gm = constant_fold(
                            torch.export.export(model, (example_input,)).module(),
                            CompilationSettings(disable_tf32=disable_tf32),
                        )
                    # With TF32 allowed, folding keeps the caller's settings.
                    self.assertEqual(ieee_float32.called, disable_tf32)
                    self.assertEqual(
                        (conv.fp32_precision, matmul.fp32_precision), ("tf32", "tf32")
                    )
                finally:
                    conv.fp32_precision, matmul.fp32_precision = saved
                if not disable_tf32:
                    continue
                folded = {
                    tuple(p.shape): p
                    for n, p in gm.named_parameters()
                    if n.startswith("_frozen_param")
                }
                for expected in (expected_conv, expected_matmul):
                    error = (folded[tuple(expected.shape)].double() - expected).abs()
                    self.assertLess(error.max().item(), 1e-5)

    def test_float32_folding_works_without_fp32_precision(self):
        # As before PyTorch 2.9: no fp32_precision flags, and matmul.allow_tf32 reads True for
        # both "high" and "medium" and sets "high".
        state = {"matmul": "medium"}

        class LegacyMatmul:
            @property
            def allow_tf32(self):
                return state["matmul"] != "highest"

            @allow_tf32.setter
            def allow_tf32(self, value):
                state["matmul"] = "high" if value else "highest"

        cudnn = types.SimpleNamespace(allow_tf32=True)
        with (
            mock.patch.object(torch.backends, "cudnn", cudnn),
            mock.patch.object(torch.backends.cuda, "matmul", LegacyMatmul()),
            mock.patch.object(
                torch, "get_float32_matmul_precision", lambda: state["matmul"]
            ),
            mock.patch.object(
                torch, "set_float32_matmul_precision", lambda p: state.update(matmul=p)
            ),
        ):
            with constant_folding._ieee_float32():
                self.assertEqual(
                    (cudnn.allow_tf32, state["matmul"]), (False, "highest")
                )
        self.assertEqual((cudnn.allow_tf32, state["matmul"]), (True, "medium"))

    @unittest.skipUnless(_HAS_FP32_PRECISION, "needs the fp32_precision flags")
    def test_overlapping_float32_folds_wait_for_each_other(self):
        conv, matmul = torch.backends.cudnn.conv, torch.backends.cuda.matmul
        saved = conv.fp32_precision, matmul.fp32_precision
        inside, release = threading.Event(), threading.Event()

        def first():
            with constant_folding._ieee_float32():
                inside.set()
                release.wait(10)

        def second():
            with constant_folding._ieee_float32():
                pass

        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        conv.fp32_precision = matmul.fp32_precision = "tf32"
        try:
            threads[0].start()
            inside.wait(10)
            threads[1].start()
            threads[1].join(0.5)
            # The second fold must not start while the first one holds the flags.
            self.assertTrue(threads[1].is_alive())
            release.set()
            for t in threads:
                t.join(10)
            self.assertEqual(
                (conv.fp32_precision, matmul.fp32_precision), ("tf32", "tf32")
            )
        finally:
            release.set()
            conv.fp32_precision, matmul.fp32_precision = saved

    def test_only_converter_safe_aliased_ops_skip_install(self):
        permute_gm = self._export_and_fold(_Permute(), torch.randn(1, 513, 512))
        self.assertIn(
            torch.ops.aten.permute.default,
            [node.target for node in permute_gm.graph.nodes],
        )
        self.assertFalse(
            any(
                name.startswith("_frozen_param")
                for name, _ in permute_gm.named_parameters()
            )
        )

        for model, example_input, op in (
            (
                _Squeeze(),
                torch.randn(512, 513),
                torch.ops.aten.squeeze.default,
            ),
            (
                _AsStrided(),
                torch.randn(512, 513),
                torch.ops.aten.as_strided.default,
            ),
            (
                _Select(),
                torch.randn(513),
                torch.ops.aten.select.int,
            ),
        ):
            with self.subTest(op=op):
                self._assert_installed(
                    self._export_and_fold(model, example_input),
                    op,
                )

    def test_aliased_permute_into_dequantize_is_installed(self):
        try:
            import torchao  # noqa: F401

            dequantize_affine = torch.ops.torchao.dequantize_affine.default
        except (ImportError, AttributeError):
            self.skipTest("torchao required")

        class Fp8WeightOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(1025, 1024).to(torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.scale = torch.nn.Parameter(
                    torch.ones(1, 1025), requires_grad=False
                )

            def forward(self, x):
                weight = dequantize_affine(
                    self.weight.permute(1, 0),
                    [1024, 1],
                    self.scale,
                    None,
                    torch.float8_e4m3fn,
                    output_dtype=torch.float32,
                )
                return x @ weight

        gm = self._export_and_fold(Fp8WeightOnly(), torch.randn(2, 1024))
        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops.aten.permute.default, targets)
        self.assertIn(dequantize_affine, targets)

    def test_quantization_ops_on_constants_are_not_run(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ (_quantize(self.weight * 2) + 1)

        gm = torch.export.export(Model(), (torch.randn(4, 4),)).module()
        quantize = torch.ops.constant_folding_test.quantize.default
        folder = constant_folding._TorchTensorRTConstantFolder(
            gm, skip_constructors=False
        )
        folder.quantization_ops.add(quantize)
        folder.run()

        folded = [node.target for node in folder.node_replacements]
        self.assertIn(torch.ops.aten.mul.Tensor, folded)
        self.assertNotIn(quantize, folded)
        self.assertNotIn(torch.ops.aten.add.Tensor, folded)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_aliased_view_models_compile_end_to_end(self):
        for model, example_input in (
            (_Permute(), torch.randn(1, 513, 512)),
            (_Squeeze(), torch.randn(512, 513)),
            (_AsStrided(), torch.randn(512, 513)),
            (_Select(), torch.randn(513)),
        ):
            model = model.eval().cuda()
            example_input = example_input.cuda()

            with self.subTest(model=type(model).__name__):
                with torch.no_grad():
                    expected = model(example_input)
                    exported_program = torch.export.export(model, (example_input,))
                    compiled_model = torch_tensorrt.dynamo.compile(
                        exported_program,
                        inputs=[example_input],
                        min_block_size=1,
                        require_full_compilation=True,
                        offload_module_to_cpu=True,
                    )
                    actual = compiled_model(example_input)

                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(
        not torch.cuda.is_available()
        or not complex_decomposition_adapter.has_complex_decomposition(),
        "needs CUDA and the upstream complex decomposition",
    )
    def test_offloaded_constants_reach_the_complex_decomposition(self):
        # offload_module_to_cpu leaves folded constants on the CPU; the complex
        # decomposition retraces the graph and folds it again, and must neither
        # fall back over them nor mix them with the device tensors it adds.
        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                angles = torch.outer(torch.arange(16.0), torch.ones(4))
                self.register_buffer(
                    "freqs", torch.polar(torch.ones_like(angles), angles)
                )
                # Read only by constant folding, so the retraced module drops it.
                self.scale = torch.nn.Parameter(torch.ones(4))
                self.register_buffer("causal", torch.ones(8, 8).tril())

            def forward(self, x):
                freqs = self.freqs[: x.shape[0]] * self.scale.mean()
                z = torch.view_as_complex(x.reshape(x.shape[0], 4, 2))
                y = torch.view_as_real(z * freqs).flatten(1)
                q = y[None, None]
                # The retrace expands the attention, turning the folded mask into
                # a bias with scalars it creates on the device.
                attended = torch.nn.functional.scaled_dot_product_attention(
                    q, q, q, attn_mask=self.causal > 0
                )
                return y, attended

        model = Rotary().eval().cuda()
        example_input = torch.randn(8, 8, device="cuda")
        with (
            torch.no_grad(),
            mock.patch.object(
                complex_decomposition_adapter,
                "complex_graph_detection",
                side_effect=AssertionError("fell back to the legacy complex pass"),
            ),
        ):
            expected = model(example_input)
            exported_program = torch.export.export(model, (example_input,))
            compiled_model = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs=[example_input],
                min_block_size=1,
                offload_module_to_cpu=True,
            )
            actual = compiled_model(example_input)

        torch.testing.assert_close(actual[0], expected[0])
        # TensorRT may run the attention's matrix multiplies in TF32.
        torch.testing.assert_close(actual[1], expected[1], rtol=0.02, atol=0.02)
        self.assertEqual(model.scale.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_offloaded_constant_moves_to_its_device_only_for_an_op_that_fails(
        self,
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("c", torch.randn(4096))
                self.register_buffer("d", torch.randn(4096))

            def forward(self, x):
                # Fails with c on the CPU next to a scalar on the device.
                shifted = self.c + torch.zeros((), device=x.device)
                # Takes a source on another device and keeps the device of c.
                copied = torch.ops.aten.copy.default(self.c, self.d)
                return x * self.c.sin() + shifted + copied.sin()

        example_input = torch.randn(4096, device="cuda")
        gm = torch.export.export(Model().cuda(), (example_input,)).module()
        # What an earlier fold leaves with offload_module_to_cpu: the value on the
        # CPU, its metadata on the device.
        c = gm.c.cpu()
        gm.c = c
        d = gm.d.cpu()
        gm = constant_fold(gm, CompilationSettings(offload_module_to_cpu=True))

        folded = [
            getattr(gm, n.target).cpu() for n in gm.graph.nodes if n.op == "get_attr"
        ]
        self.assertTrue(any(torch.equal(t, c) for t in folded))
        # sin rounds differently on the GPU, so these show both still ran on the CPU.
        self.assertTrue(any(torch.equal(t, c.sin()) for t in folded))
        self.assertTrue(any(torch.equal(t, d.sin()) for t in folded))

    @unittest.skipIf(
        not torch.cuda.is_available()
        or not complex_decomposition_adapter.has_complex_decomposition(),
        "needs CUDA and the upstream complex decomposition",
    )
    def test_complex_fallback_does_not_hold_device_copies_of_offloaded_constants(
        self,
    ):
        from torch._functorch._aot_autograd import complex_decomposition

        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                angles = torch.outer(torch.arange(256.0), torch.ones(512))
                self.register_buffer(
                    "freqs", torch.polar(torch.ones_like(angles), angles)
                )
                self.scale = torch.nn.Parameter(torch.ones(512))

            def forward(self, x):
                z = torch.view_as_complex(x.reshape(256, 512, 2))
                return torch.view_as_real(z * (self.freqs * self.scale.mean()))

        settings = CompilationSettings(offload_module_to_cpu=True)
        example_input = torch.randn(256, 1024, device="cuda")
        exported_program = torch.export.export(Rotary().cuda(), (example_input,))
        # Folds freqs * scale into a 1 MiB constant kept on the CPU.
        gm = constant_fold(exported_program.module(), settings)

        def failing_retrace(gm, flat_args, decompositions):
            # Like a real retrace, its frame holds the constants it was handed.
            constants = [
                getattr(gm, n.target) for n in gm.graph.nodes if n.op == "get_attr"
            ]
            raise NotImplementedError(f"unsupported op, {len(constants)} constants")

        allocated = {}

        def legacy_pass(gm, settings):
            allocated["fallback"] = torch.cuda.memory_allocated()
            return gm

        with (
            mock.patch.object(
                complex_decomposition, "decompose_complex_in_graph", failing_retrace
            ),
            mock.patch.object(
                complex_decomposition_adapter, "complex_graph_detection", legacy_pass
            ),
        ):
            allocated["before"] = torch.cuda.memory_allocated()
            complex_decomposition_adapter.complex_decomposition_adapter(gm, settings)

        self.assertEqual(allocated["fallback"], allocated["before"])


if __name__ == "__main__":
    run_tests()
