# type: ignore
"""Converter tests for the fallback (non-KV) path of aten.slice_scatter.default.

The converter has two paths:

1. **KV-cache fast path** — emits ``IKVCacheUpdateLayer`` with aliased I/O.
   Aliasing requires the C++ runtime, so these cases are tested end-to-end
   in ``tests/py/dynamo/runtime/test_aliased_io.py`` (the Python-runtime
   converter harness can't bind aliased addresses).

2. **Scatter fallback** — equivalent to the historical Torch-TensorRT
   decomposition (``arange + scatter``). Used for any shape that doesn't
   meet KVCacheUpdate's invariants.

This file covers the fallback path. To force the fallback regardless of
shape we add a small no-op (``+ 0``) to the cache so it isn't a direct
network input — the converter's "input is a placeholder" check fails and
falls through to scatter.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.aten_ops_converters import slice_scatter_validator
from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
    slice_scatter as slice_scatter_impl,
)

from .harness import DispatchTestCase

# What torch.export writes in place of the dim size for an open-ended slice.
OPEN_END = 2**63 - 1


class _SliceScatterNotInputModule(torch.nn.Module):
    """Helper: forces the fallback path by making `cache` not a direct
    network input (the converter's KV fast path requires placeholder input).
    """

    def __init__(self, dim, start, end, step=1):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, cache_in, update):
        # `cache_in + 0` produces a non-placeholder ITensor, forcing the
        # converter to take the scatter fallback rather than KVCacheUpdate.
        cache = cache_in + 0
        return torch.ops.aten.slice_scatter.default(
            cache, update, self.dim, self.start, self.end, self.step
        )


class TestSliceScatterFallback(DispatchTestCase):
    @parameterized.expand(
        [
            # (name, cache_shape, update_shape, dim, start, end)
            # 3-D
            ("rank3_dim1", (4, 8, 16), (4, 2, 16), 1, 3, 5),
            # 4-D writing on dim != 2 (not eligible for KVCacheUpdate)
            ("rank4_dim1", (2, 8, 4, 16), (2, 2, 4, 16), 1, 2, 4),
            ("rank4_dim3", (2, 4, 16, 8), (2, 4, 16, 2), 3, 1, 3),
            # 2-D
            ("rank2_dim0", (8, 16), (3, 16), 0, 2, 5),
            # 5-D
            ("rank5_dim2", (2, 3, 8, 4, 16), (2, 3, 2, 4, 16), 2, 1, 3),
            # 4-D dim=2 — the eligible shape, but forced via non-placeholder
            # input. Tests that the fallback handles the same shape correctly.
            ("rank4_dim2_forced", (2, 4, 16, 8), (2, 4, 1, 8), 2, 3, 4),
        ]
    )
    def test_fallback(self, _, cache_shape, update_shape, dim, start, end):
        module = _SliceScatterNotInputModule(dim, start, end)
        cache = torch.randn(cache_shape)
        update = torch.randn(update_shape)
        self.run_test(module, [cache, update])

    def test_fallback_step_two(self):
        module = _SliceScatterNotInputModule(2, 0, 16, step=2)
        cache = torch.randn(2, 4, 16, 8)
        update = torch.randn(2, 4, 8, 8)
        self.run_test(module, [cache, update])

    def test_fallback_dynamic_shape(self):
        """Dim 2, the one written, is fixed at 64 on purpose: a dynamic sliced dim has
        no lowering and is validated away instead."""
        module = _SliceScatterNotInputModule(2, 1, 60, step=3)
        input_specs = [
            Input(
                min_shape=(2, 4, 64, 8),
                opt_shape=(3, 4, 64, 12),
                max_shape=(4, 4, 64, 16),
                dtype=torch.float32,
            ),
            Input(
                min_shape=(2, 4, 20, 8),
                opt_shape=(3, 4, 20, 12),
                max_shape=(4, 4, 20, 16),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(module, input_specs)

    def test_fallback_open_end_step_two(self):
        """``cache[:, :, ::2, :]``: the open end has to be clamped to the dim, since the
        index range is built with ``np.arange``."""
        module = _SliceScatterNotInputModule(2, 0, OPEN_END, step=2)
        cache = torch.randn(2, 4, 16, 8)
        update = torch.randn(2, 4, 8, 8)
        self.run_test(module, [cache, update])

    def test_fallback_open_end_interior_start(self):
        """``cache[:, :, 3:, :]``: a start that rules out the full-overwrite shortcut,
        so the clamp is what makes the write 13 slots wide and not INT64_MAX - 3."""
        module = _SliceScatterNotInputModule(2, 3, OPEN_END, step=1)
        cache = torch.randn(2, 4, 16, 8)
        update = torch.randn(2, 4, 13, 8)
        self.run_test(module, [cache, update])

    def test_full_overwrite_is_identity(self):
        """When start=0, end=dim_size, step=1, the converter short-circuits
        and returns ``src`` directly. Wrap the returned tensor in a small op
        so it isn't simultaneously a network input and a network output —
        which TRT rejects (handled by ``repair_input_as_output`` in
        production but bypassed in this lower-level harness)."""

        class M(torch.nn.Module):
            def forward(self, cache_in, update):
                cache = cache_in + 0  # force non-placeholder
                out = torch.ops.aten.slice_scatter.default(cache, update, 2, 0, 16)
                return out + 0  # avoid placeholder-as-output

        cache = torch.randn(2, 4, 16, 8)
        update = torch.randn(2, 4, 16, 8)
        self.run_test(M(), [cache, update])


class TestSliceScatterEarlyExits(unittest.TestCase):
    """The converter's three raising exits, driven through the converter itself.

    All are reached before the converter touches anything but ``input.shape``, so
    ``_call`` passes ``None`` for ``ctx``, ``target``, ``source_ir`` and ``src``, and
    an object carrying only a shape for the cache. Those five stand-ins are what
    breaks if any exit is ever moved below a line that reads one of them.

    ``run_test`` reaches none of them: a non-int bound has no concrete
    ``aten.slice_scatter`` to trace into, and a bad ``dim`` raises out of torch before
    any engine is built, pinning torch's message rather than the converter's.
    """

    _CACHE_SHAPE = (2, 4, 16, 8)

    def _call(self, dim, start, end, step, cache_shape=None):
        cache = SimpleNamespace(shape=cache_shape or self._CACHE_SHAPE)
        return slice_scatter_impl(
            None, None, None, "test_slice_scatter", cache, None, dim, start, end, step
        )

    def test_dynamic_bound_is_not_implemented(self):
        # A bound that is not a Python int is what the converter cannot lower; a
        # bare object stands in for the symbolic value that carries one in a real
        # dynamic-shape graph, since the resolver discriminates only on int-ness.
        symbolic_end = object()
        with self.assertRaisesRegex(
            NotImplementedError, "dynamic start/end/step is not yet supported"
        ):
            self._call(2, 0, symbolic_end, 1)

    def test_out_of_range_dim_is_an_index_error(self):
        """A Python int outside the rank is rejected on range, not on type, and the
        message has to say which: the type it prints is the required one, so the range
        is the only thing that explains the error."""
        with self.assertRaisesRegex(
            IndexError,
            r"^slice_scatter: 9 of type int is not a valid dim for a rank-4 input; "
            r"dim must be a Python int in \[-4, 4\)$",
        ):
            self._call(9, 0, 4, 1)

    def test_non_int_dim_is_an_index_error(self):
        """An in-range numpy.int64 dim is rejected on type, not on range, and the
        message has to say which: the value it prints is in range, so the type is the
        only thing that explains the error. Matching the type name rather than the
        rendered value keeps this independent of how numpy formats a scalar, which
        changed in numpy 2.0."""
        with self.assertRaisesRegex(
            IndexError,
            r"^slice_scatter: 2 of type int64 is not a valid dim for a rank-4 input; "
            r"dim must be a Python int in \[-4, 4\)$",
        ):
            self._call(np.int64(2), 0, 4, 1)

    def test_any_bound_on_a_dynamic_dim_is_not_implemented(self):
        """TensorRT's -1 is no size to resolve a bound against, concrete ones included.
        Matching on the dim separates this from the dynamic-bounds exit above."""
        for start, end in ((3, OPEN_END), (0, 40), (1, 5)):
            with self.assertRaises(NotImplementedError) as raised:
                self._call(2, start, end, 1, cache_shape=(2, 4, -1, 8))
            self.assertIn("dim 2 of the input is dynamic", str(raised.exception))


class TestSliceScatterValidator(TestCase):
    """What the validator keeps out of TensorRT, checked directly rather than through a
    compile that happens to succeed. Nodes are built by hand so their metadata is."""

    # Every shape a bound comes in, all needing the size of the dim written. `1:5` fits
    # the declared minimum and is rejected anyway: only the range says so, not the dim.
    _BOUNDS = (
        (3, OPEN_END),
        (3, None),
        (None, None),
        (-4, None),
        (-4, 12),
        (0, 40),
        (1, 5),
    )

    @staticmethod
    def _static_node(*slice_args):
        graph = torch.fx.Graph()
        cache = graph.placeholder("cache")
        cache.meta["val"] = torch.empty((2, 4, 16, 8), device="meta")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty((2, 4, 13, 8), device="meta")
        return graph.call_function(
            torch.ops.aten.slice_scatter.default, args=(cache, src, *slice_args)
        )

    @staticmethod
    def _dynamic_seq_node(*slice_args):
        """Spliced into an exported graph, so the cache carries a real ``SymInt``."""

        class Passthrough(torch.nn.Module):
            def forward(self, cache, update):
                return cache + 0

        seq = torch.export.Dim("seq", min=8, max=32)
        ep = torch.export.export(
            Passthrough(),
            (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 13, 8)),
            dynamic_shapes={"cache": {2: seq}, "update": None},
        )
        gm = ep.module()
        cache, update = [n for n in gm.graph.nodes if n.op == "placeholder"][:2]
        output = next(n for n in gm.graph.nodes if n.op == "output")
        with gm.graph.inserting_before(output):
            return gm.graph.call_function(
                torch.ops.aten.slice_scatter.default, args=(cache, update, *slice_args)
            )

    def test_every_bound_on_a_dynamic_dim_is_rejected(self):
        """Left in the engine, the open end reaches ``np.arange`` as a request for
        INT64_MAX entries, and ``end=40`` as one for 40 slots of an at-most-32 dim."""
        for start, end in self._BOUNDS:
            self.assertFalse(
                slice_scatter_validator(self._dynamic_seq_node(2, start, end))
            )

    def test_a_static_dim_resolves_every_bound(self):
        """The same bounds on a static dim all resolve -- clamped, or counted from the
        end -- so none of them is the validator's business."""
        for start, end in self._BOUNDS:
            self.assertTrue(slice_scatter_validator(self._static_node(2, start, end)))

    def test_a_node_without_shape_metadata_is_passed(self):
        """Rejecting is the more damaging guess: vetoing a write the KV-cache classifier
        read the same metadata for fails ``assert_predicted_kv_aliased``."""
        graph = torch.fx.Graph()
        cache = graph.placeholder("cache")
        src = graph.placeholder("src")
        node = graph.call_function(
            torch.ops.aten.slice_scatter.default, args=(cache, src, 2, 3, OPEN_END)
        )
        self.assertEqual(cache.meta, {})
        self.assertTrue(slice_scatter_validator(node))


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSliceScatterDynamicDimEndToEnd(TestCase):
    """A write on a dynamic dim has no lowering, and the point of the validator is that
    the model compiles anyway. The ``+ 1`` gives the engine something to take, so the
    write is partitioned out to PyTorch rather than the whole graph falling back."""

    class Write(torch.nn.Module):
        """``cache[:, :, start:end, :] = update``, as the op export emits."""

        def __init__(self, start, end):
            super().__init__()
            self.start = start
            self.end = end

        def forward(self, cache, update):
            written = torch.ops.aten.slice_scatter.default(
                cache, update, 2, self.start, self.end
            )
            return written + 1.0

    def _assert_matches_eager(self, mod, dynamic_shapes, *shape_pairs):
        """Compile at the first pair of shapes, check every pair."""
        mod = mod.eval().cuda()
        inputs = [
            [torch.randn(cache).cuda(), torch.randn(update).cuda()]
            for cache, update in shape_pairs
        ]
        ep = torch.export.export(mod, tuple(inputs[0]), dynamic_shapes=dynamic_shapes)
        trt_mod = torch_tensorrt.dynamo.compile(ep, inputs[0], min_block_size=1)
        for args in inputs:
            torch.testing.assert_close(trt_mod(*args), mod(*args))

    def test_open_end_on_a_dynamic_dim_matches_eager(self):
        """``cache[:, :, 3:, :] = update``, where the open end is INT64_MAX."""
        seq = torch.export.Dim("seq", min=8, max=32)
        self._assert_matches_eager(
            self.Write(3, OPEN_END),
            {"cache": {2: seq}, "update": {2: seq - 3}},
            ((2, 4, 16, 8), (2, 4, 13, 8)),
            ((2, 4, 24, 8), (2, 4, 21, 8)),
        )

    def test_an_end_past_the_dynamic_dim_matches_eager(self):
        """``cache[:, :, :40, :] = update`` on a dim declared at most 32: a plain 40
        looks resolvable until it is read against the dim it can never fit."""
        seq = torch.export.Dim("seq", min=8, max=32)
        self._assert_matches_eager(
            self.Write(0, 40),
            {"cache": {2: seq}, "update": {2: seq}},
            ((2, 4, 16, 8), (2, 4, 16, 8)),
            ((2, 4, 24, 8), (2, 4, 24, 8)),
        )


if __name__ == "__main__":
    run_tests()
