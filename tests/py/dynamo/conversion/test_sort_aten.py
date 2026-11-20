# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import ENABLED_FEATURES, Input
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._meta_keys import NUM_SORT_KEYS
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    aten_ops_sort_stable,
    sort_stable_validator,
    sort_validator,
)
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
    _remove_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.annotate_sort_num_keys import (
    _num_index_values,
    annotate_sort_num_keys,
)

from .harness import DispatchTestCase

_NEEDS_STABLE_SORT_CONVERTER = unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx or not is_tensorrt_version_supported("10.8.0"),
    "the stable sort converter needs TensorRT 10.8 or newer, and not TensorRT-RTX",
)


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 0, True),
            ((2, 3, 4, 5), 1, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
            ((1, 2, 5, 3), -2, False),
            ((6, 2, 1, 3), -4, True),
            # ITopKLayer requires rank >= 2; converter broadcasts rank-1
            ((64,), 0, False),
            ((64,), 0, True),
            ((16,), -1, True),
        ]
    )
    def test_sort(self, input_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sort(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


class TestSortConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_descending",
                (2, 1, 4),
                (3, 2, 4),
                (3, 3, 4),
                2,
                True,
            ),
            (
                "4d_dynamic_ascending",
                (2, 2, 1, 4),
                (2, 2, 2, 4),
                (3, 3, 2, 4),
                3,
                False,
            ),
            (
                "4d_dynamic_descending_neg_dim",
                (1, 3, 1, 1),
                (2, 3, 2, 2),
                (3, 3, 2, 4),
                -3,
                True,
            ),
        ]
    )
    def test_sort_dynamic(self, _, min_shape, opt_shape, max_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sort(),
            input_specs,
            output_dtypes=[torch.float, torch.int64],
            use_dynamo_tracer=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSortValidatorDefaultDim(TestCase):
    """Regression test for https://github.com/pytorch/TensorRT/issues/3777.

    sort_validator is only consulted by the partitioner, which
    DispatchTestCase bypasses (it interprets the traced graph directly), so
    the cases above don't exercise it. torch.sort(x) with no explicit dim
    omits dim from the FX node's args entirely (relying on the schema
    default of -1), so this goes through the full compile() pipeline, where
    sort_validator used to crash with `IndexError: tuple index out of
    range` from indexing node.args[1] unconditionally.
    """

    def test_sort_default_dim_compiles(self):
        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.randn(4, 8).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_sort_scalar_input_falls_back(self):
        """A 0-D (scalar) input has no dim to normalize: get_positive_dim
        computes dim % rank, and rank=0 for a scalar raises ZeroDivisionError.
        sort_validator must reject scalar inputs before reaching that call so
        the node falls back to PyTorch instead of crashing the partitioner."""

        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.tensor(5.0).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 0)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


class TestSortValidator(unittest.TestCase):
    def _sort_node(self, fn):
        class Mod(nn.Module):
            def forward(self, x):
                return fn(x)

        gm = torch.export.export(Mod(), (torch.randn(4, 8),)).module()
        return next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.sort.default
        )

    def test_defaulted_dim_does_not_raise(self):
        # export drops dim=-1, so args is (x,) only; indexing args[1] used to IndexError
        node = self._sort_node(lambda x: torch.sort(x))
        self.assertEqual(len(node.args), 1)
        self.assertTrue(sort_validator(node))

    def test_explicit_dim_still_works(self):
        node = self._sort_node(lambda x: torch.sort(x, 0))
        self.assertGreaterEqual(len(node.args), 2)
        self.assertTrue(sort_validator(node))


@_NEEDS_STABLE_SORT_CONVERTER
class TestSortStableConverter(DispatchTestCase):
    """Stable sort of the indices that topk or argmax return, as mixture-of-experts routing does."""

    @parameterized.expand(
        [
            # 8000 keys, above the 3840 elements a TopK layer can sort, in 62.5 tiles
            ("partial_tile", (1000, 16), 8, None, False, False),
            ("whole_tiles_descending_int32", (512, 16), 8, None, True, True),
            ("dim_not_last", (300, 1, 32), 1, 0, False, False),
            # The largest sorts the validator accepts: 2**20 counts of 2**14 elements, and of
            # 2**13 key values.
            ("most_counts", (2048, 64), 8, None, False, False),
            ("most_key_values", (16, 8192), 8, None, False, False),
            ("one_key_value", (500, 1), 1, None, False, False),
        ]
    )
    def test_sort_stable(self, _, shape, k, dim, descending, int32):
        class Sort(nn.Module):
            def forward(self, x):
                ids = torch.topk(x, k, dim=-1)[1]
                if dim is None:
                    ids = ids.reshape(-1)
                if int32:
                    ids = ids.to(torch.int32)
                return torch.ops.aten.sort.stable(
                    ids,
                    stable=True,
                    dim=-1 if dim is None else dim,
                    descending=descending,
                )

        self.run_test(
            Sort(),
            [torch.randn(shape)],
            rtol=0,
            atol=0,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    def test_tiles_of_one_key(self):
        # Nearly every key is 0, so most tiles hold one key and its count reaches the tile size.
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.stable(x.argmax(-1), stable=True)

        self.run_test(
            Sort(),
            [torch.randn(3000, 4) + torch.tensor([8.0, 0.0, 0.0, 0.0])],
            rtol=0,
            atol=0,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


_CONVERTERS_LOGGER = "torch_tensorrt.dynamo.conversion.aten_ops_converters"


def _stable_sort_node(fn, x, dynamic_shapes=None):
    class Mod(nn.Module):
        def forward(self, x):
            return fn(x)

    # Partitioning sees the graph after decompositions and lowering passes.
    ep = torch.export.export(Mod(), (x,), dynamic_shapes=dynamic_shapes)
    gm = post_lowering(
        ep.run_decompositions(get_decompositions(False)).module(),
        CompilationSettings(),
    )
    return next(
        n
        for n in reversed(gm.graph.nodes)
        if n.op == "call_function" and n.target is torch.ops.aten.sort.stable
    )


def _topk_ids(x):
    return torch.topk(x, 2)[1].reshape(-1)


class TestSortStableAnyTensorRT(unittest.TestCase):
    """These tests need no converter, so they run on any TensorRT version."""

    def test_old_tensorrt(self):
        node = _stable_sort_node(
            lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
        )
        with (
            mock.patch(
                "torch_tensorrt.dynamo.conversion.aten_ops_converters.ENABLED_FEATURES",
                ENABLED_FEATURES._replace(tensorrt_rtx=False),
            ),
            mock.patch(
                "torch_tensorrt.dynamo.conversion.aten_ops_converters.is_tensorrt_version_supported",
                return_value=False,
            ),
            self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs,
        ):
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("needs TensorRT 10.8", logs.output[0])

    def test_tensorrt_rtx(self):
        node = _stable_sort_node(
            lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
        )
        with (
            mock.patch(
                "torch_tensorrt.dynamo.conversion.aten_ops_converters.ENABLED_FEATURES",
                ENABLED_FEATURES._replace(tensorrt_rtx=True),
            ),
            self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs,
        ):
            self.assertFalse(sort_stable_validator(node))
        self.assertIn("TensorRT-RTX", logs.output[0])

    def test_converter_refuses_int32_overflow(self):
        # A row of 8 keys over 2**28 values needs 2**35 counts, past the int32 positions.
        node = _stable_sort_node(
            lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
        )
        node.meta[NUM_SORT_KEYS] = 2**28
        ctx = mock.Mock(current_node=node)
        with self.assertRaisesRegex(RuntimeError, "int32"):
            aten_ops_sort_stable.__wrapped__(
                ctx, node.target, node.args, node.kwargs, node.name
            )

    def test_cache_key_has_key_range(self):
        node = _stable_sort_node(
            lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
        )

        def cache_key(num_keys):
            node.meta[NUM_SORT_KEYS] = num_keys
            return BaseEngineCache.get_hash(
                node.graph.owning_module, [Input(shape=(4, 8))], CompilationSettings()
            )

        self.assertNotEqual(cache_key(8), cache_key(128))

    def test_topk_dim_keyword(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 8)
        topk = graph.call_function(torch.ops.aten.topk.default, (x, 2), {"dim": 0})
        ids = graph.call_function(operator.getitem, (topk, 1))
        self.assertEqual(_num_index_values(ids), 4)

    def test_stale_range_cleared(self):
        node = _stable_sort_node(
            lambda x: torch.sort(x.int().reshape(-1), stable=True), torch.randn(4, 8)
        )
        node.meta[NUM_SORT_KEYS] = 8
        annotate_sort_num_keys(node.graph.owning_module, CompilationSettings())
        self.assertNotIn(NUM_SORT_KEYS, node.meta)

    def test_range_after_registered_passes(self):
        # A pass registered by a user runs after the built-in ones and can change the keys.
        @_aten_lowering_pass
        def shift_keys(gm, settings):
            sort = next(
                n for n in gm.graph.nodes if n.target is torch.ops.aten.sort.stable
            )
            with gm.graph.inserting_before(sort):
                shifted = gm.graph.call_function(
                    torch.ops.aten.add.Tensor, (sort.args[0], 8)
                )
            shifted.meta["val"] = sort.args[0].meta["val"] + 8
            sort.replace_input_with(sort.args[0], shifted)
            gm.recompile()
            return gm

        try:
            node = _stable_sort_node(
                lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
            )
        finally:
            _remove_lowering_pass(index=-1)
        self.assertNotIn(NUM_SORT_KEYS, node.meta)

    def test_range_after_fake_tensor_refresh(self):
        # A registered pass that widens the topk input but keeps its old shape in meta.
        @_aten_lowering_pass
        def widen_scores(gm, settings):
            topk = next(
                n for n in gm.graph.nodes if n.target is torch.ops.aten.topk.default
            )
            scores = topk.args[0]
            with gm.graph.inserting_before(topk):
                doubled = gm.graph.call_function(torch.ops.aten.mul.Tensor, (scores, 2))
                wide = gm.graph.call_function(
                    torch.ops.aten.cat.default, ([scores, doubled], 1)
                )
            doubled.meta["val"] = wide.meta["val"] = scores.meta["val"]
            topk.replace_input_with(scores, wide)
            gm.recompile()
            return gm

        try:
            node = _stable_sort_node(
                lambda x: torch.sort(_topk_ids(x), stable=True), torch.randn(4, 8)
            )
        finally:
            _remove_lowering_pass(index=-1)
        self.assertEqual(node.meta[NUM_SORT_KEYS], 16)


@_NEEDS_STABLE_SORT_CONVERTER
class TestSortStableValidator(unittest.TestCase):
    @parameterized.expand(
        [
            ("topk", _topk_ids, 8),
            ("argmax_int32", lambda x: x.argmax(-1).to(torch.int32), 8),
            ("argmax_flat", lambda x: x.argmax().reshape(1), 32),
            ("argmax_scalar", lambda x: x.sum().argmax(0).reshape(1), 1),
            ("argmin", lambda x: x.argmin(-1), 8),
            ("permute", lambda x: torch.topk(x, 2)[1].permute(1, 0).reshape(-1), 8),
            ("select", lambda x: torch.topk(x, 2)[1][1], 8),
        ]
    )
    def test_known_range(self, _, keys, num_keys):
        node = _stable_sort_node(
            lambda x: torch.sort(keys(x), stable=True), torch.randn(4, 8)
        )
        self.assertEqual(node.meta[NUM_SORT_KEYS], num_keys)
        self.assertTrue(sort_stable_validator(node))

    @parameterized.expand(
        [
            ("float_keys", "int32 or", lambda x: _topk_ids(x).float()),
            ("scalar", "int32 or", lambda x: x.argmax()),
            ("empty_row", "empty row", _topk_ids, (0, 8)),
            ("many_rows", "one row", lambda x: torch.topk(x, 8)[1], (4, 300)),
            ("unbacked_rows", "one row", lambda x: torch.topk(x[x[:, 0] > 0], 2)[1]),
            (
                "unbacked_length",
                "empty row",
                lambda x: torch.topk(x[x[:, 0] > 0], 2)[1].reshape(-1),
            ),
            # Their count would grow with the square of the length.
            ("sort_indices", "Cannot tell", lambda x: torch.sort(x)[1].reshape(-1)),
            ("shifted", "Cannot tell", lambda x: _topk_ids(x) + 1),
            # int8 wraps the 300 positions around.
            (
                "int8_cast",
                "Cannot tell",
                lambda x: _topk_ids(x).to(torch.int8).to(torch.int32),
                (4, 300),
            ),
            (
                "topk_values",
                "Cannot tell",
                lambda x: torch.topk(x, 2)[0].long().reshape(-1),
            ),
            (
                "dynamic_key_range",
                "Cannot tell",
                _topk_ids,
                (4, 8),
                {"x": {1: torch.export.Dim("keys", min=2, max=64)}},
            ),
            (
                "unbounded_length",
                "The length of",
                _topk_ids,
                (4, 8),
                {"x": {0: torch.export.Dim("tokens", min=2)}},
            ),
            (
                "large_dynamic_max",
                "counts, where",
                _topk_ids,
                (4, 8),
                {"x": {0: torch.export.Dim("tokens", min=2, max=2_200_000)}},
            ),
            # 1026 keys with 1024 values each, one tile past the cap.
            ("too_many_counts", "counts, where", _topk_ids, (513, 1024)),
            # 16,386 keys with 2 values each, just past the cap.
            ("too_long", "elements, where", _topk_ids, (8193, 2)),
        ]
    )
    def test_refused(self, _, message, keys, shape=(4, 8), dynamic_shapes=None):
        node = _stable_sort_node(
            lambda x: torch.sort(keys(x), stable=True),
            torch.randn(shape),
            dynamic_shapes,
        )
        with self.assertLogs(_CONVERTERS_LOGGER, level="DEBUG") as logs:
            self.assertFalse(sort_stable_validator(node))
        self.assertIn(message, logs.output[0])
        self.assertEqual(logs.records[0].levelname, "DEBUG")


@_NEEDS_STABLE_SORT_CONVERTER
@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSortStableCompile(TestCase):
    def _compile(self, mod, inputs, **kwargs):
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1, **kwargs
        )
        in_pytorch = [
            str(node.target)
            for name, sub in trt_mod.named_children()
            if "_run_on_gpu" in name
            for node in sub.graph.nodes
            if node.op == "call_function"
        ]
        return trt_mod, in_pytorch

    def test_keys_made_in_pytorch(self):
        # The engine sees only the placeholder for the topk result, the same for both expert
        # counts, so the range has to come from the whole graph and go into the cache key.
        class Routing(nn.Module):
            def forward(self, scores):
                experts = torch.topk(scores, 8, dim=-1)[1].reshape(-1)
                return experts.argsort(stable=True) + 1

        with tempfile.TemporaryDirectory() as cache_dir:
            # The last compile loads the engine built for 16 experts and refits it.
            for num_experts, cached in ((16, False), (128, False), (16, True)):
                mod = Routing().eval().cuda()
                inputs = [torch.randn(1000, num_experts).cuda()]
                with self.assertLogs("torch_tensorrt", level="INFO") as logs:
                    trt_mod, in_pytorch = self._compile(
                        mod,
                        inputs,
                        torch_executed_ops={"torch.ops.aten.topk.default"},
                        immutable_weights=False,
                        cache_built_engines=True,
                        reuse_cached_engines=True,
                        engine_cache_dir=cache_dir,
                    )
                hit = any("Found the cached engine" in line for line in logs.output)
                self.assertEqual(hit, cached)
                self.assertNotIn("aten.sort.stable", " ".join(in_pytorch))
                torch.testing.assert_close(
                    trt_mod(*inputs), mod(*inputs), rtol=0, atol=0
                )

    def test_keys_out_of_range(self):
        # Compiled for 16 experts and called with 128: the engine input keeps its shape, and
        # nothing checks the keys that topk makes in PyTorch.
        class Routing(nn.Module):
            def forward(self, scores):
                return torch.topk(scores, 8, dim=-1)[1].reshape(-1).argsort(stable=True)

        trt_mod, in_pytorch = self._compile(
            Routing().eval().cuda(),
            [torch.randn(1000, 16).cuda()],
            torch_executed_ops={"torch.ops.aten.topk.default"},
        )
        self.assertNotIn("aten.sort.stable", " ".join(in_pytorch))
        order = trt_mod(torch.randn(1000, 128).cuda())
        torch.testing.assert_close(
            order.sort().values, torch.arange(8000).cuda(), rtol=0, atol=0
        )

    def test_refused_sort_runs_in_pytorch(self):
        class Routing(nn.Module):
            def forward(self, scores):
                return torch.topk(scores, 8, dim=-1)[1].argsort(stable=True)

        mod = Routing().eval().cuda()
        scores = torch.randn(2, 300).cuda()
        trt_mod, in_pytorch = self._compile(mod, [scores])
        self.assertIn("aten.sort.stable", " ".join(in_pytorch))
        torch.testing.assert_close(trt_mod(scores), mod(scores), rtol=0, atol=0)

    def test_dynamic_length_at_every_size(self):
        class Routing(nn.Module):
            def forward(self, scores):
                experts = torch.topk(scores, 8, dim=-1)[1].reshape(-1)
                return experts.argsort(stable=True)

        mod = Routing().eval().cuda()
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 128), opt_shape=(500, 128), max_shape=(1000, 128)
            )
        ]
        trt_mod, in_pytorch = self._compile(mod, inputs)
        self.assertEqual(in_pytorch, [])
        # One partial tile, whole tiles (4096 keys) and the largest size.
        for tokens in (1, 512, 1000):
            scores = torch.randn(tokens, 128).cuda()
            torch.testing.assert_close(trt_mod(scores), mod(scores), rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
