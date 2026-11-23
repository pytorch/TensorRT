# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""End-to-end tests for ``aten.index_copy`` KV-cache aliasing.

Two converters are registered for ``aten.index_copy.default``:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires for
  the narrow KV-eligible case (4-D static cache, dim=2, source seq dim
  size 1, batch=1) and emits ``IKVCacheUpdateLayer`` with aliased I/O.

* ``aten_ops_index_copy_fallback`` — STANDARD priority. Fires for
  everything else; produces correct results via the scatter path. No
  graph break.

These tests verify both paths end-to-end via the C++ runtime: the
fast path mutates in place, the fallback produces correct numerical
results without aliasing.
"""

import contextlib
import random
import tempfile
from unittest import mock

import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests


class _CudaGraphsOff(TestCase):
    """Another test in the same process can leave CUDA graphs on, and these tests
    expect them off unless they turn them on themselves."""

    def setUp(self):
        super().setUp()
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        super().tearDown()


def _compile(model, args, **kwargs):
    ep = export(model, tuple(args))
    return torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=list(args),
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
        **kwargs,
    )


def _aliased_io(compiled):
    for _name, mod in compiled.named_modules():
        if hasattr(mod, "aliased_io") and mod.aliased_io:
            return dict(mod.aliased_io)
    return {}


class TestIndexCopyKVFastPath(_CudaGraphsOff):
    """KV-eligible: 4-D static cache, dim=2, batch=1, single-position
    write. The fast path emits ``IKVCacheUpdateLayer`` with aliased output
    only for a cache the model writes."""

    def test_functional_write_of_an_input_leaves_it_alone(self):
        """The caller's cache is an input the model does not write, so the engine
        must not update it in place, for index_copy and for slice_scatter."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(1, 4, 1, 8, device="cuda") * 7.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager_out = torch.ops.aten.index_copy.default(cache, 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager_out))
        self.assertTrue(torch.equal(cache_run, cache))

        class S(torch.nn.Module):
            def forward(self, cache, update):
                return torch.ops.aten.slice_scatter.default(cache, update, 2, 3, 4)

        compiled = _compile(S().cuda(), (cache.clone(), update.clone()))
        self.assertEqual(_aliased_io(compiled), {})
        cache_run = cache.clone()
        out = compiled(cache_run, update)
        out_val = out[0] if isinstance(out, tuple) else out
        self.assertTrue(torch.allclose(out_val, S()(cache, update)))
        self.assertTrue(torch.equal(cache_run, cache))


class TestIndexCopyFallback(TestCase):
    """Cases where the validator denies the KV fast path. The fallback
    converter must produce correct results without aliasing."""

    def test_rank_3_input_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_copy.default(x, 1, index, update)

        x = torch.zeros(2, 8, 16, device="cuda")
        index = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")
        update = torch.randn(2, 3, 16, device="cuda")

        compiled = _compile(M().cuda(), (x.clone(), index, update.clone()))

        # No aliasing (validator rejected the KV path).
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(x.clone(), index, update)
        eager = torch.ops.aten.index_copy.default(x.clone(), 1, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_int8_and_float64_writes_stay_in_pytorch(self):
        """TensorRT treats an int8 scatter as quantized and fails the engine build,
        and without ``truncate_double`` a float64 engine expects float32."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        for dtype in (torch.int8, torch.float64):
            with self.subTest(dtype=dtype):
                cache = torch.zeros(1, 4, 16, 8, dtype=dtype, device="cuda")
                index = torch.tensor([9, 2], device="cuda")
                update = torch.full((1, 4, 2, 8), 3, dtype=dtype, device="cuda")
                compiled = _compile(M().cuda(), (cache, index, update))
                torch.testing.assert_close(
                    compiled(cache.clone(), index, update),
                    M()(cache.clone(), index, update),
                )

    def test_dim_other_than_two_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 1, index, update)

        cache = torch.zeros(1, 16, 4, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(1, 1, 4, 8, device="cuda") * 5.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        eager = torch.ops.aten.index_copy.default(cache.clone(), 1, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_batch_gt_one_uses_fallback(self):
        """Batch > 1 currently routes to fallback (broadcasting writeIndices
        is a Phase-2 extension)."""

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(4, 4, 16, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(4, 4, 1, 8, device="cuda") * 7.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
        self.assertTrue(torch.allclose(out, eager))

    def test_0d_input_uses_fallback(self):
        # x[0] * 2.0 and y[0] * 3.0 are 0-d, and index_copy puts one into the other.
        class M(torch.nn.Module):
            def forward(self, x, index, y):
                return torch.ops.aten.index_copy.default(
                    x[0] * 2.0, 0, index, y[0] * 3.0
                )

        x = torch.tensor([2.0, 5.0], device="cuda")
        index = torch.tensor([0], dtype=torch.int64, device="cuda")
        y = torch.tensor([7.0, 1.0], device="cuda")

        compiled = _compile(M().cuda(), (x, index, y))
        torch.testing.assert_close(compiled(x, index, y), M()(x, index, y))

    def test_0d_source_into_1d_input_uses_fallback(self):
        # y[0] * 3.0 is 0-d, and index_copy puts it into one element of x.
        class M(torch.nn.Module):
            def forward(self, x, index, y):
                return torch.ops.aten.index_copy.default(x * 2.0, 0, index, y[0] * 3.0)

        x = torch.tensor([2.0, 5.0, 4.0], device="cuda")
        index = torch.tensor([1], dtype=torch.int64, device="cuda")
        y = torch.tensor([7.0, 1.0], device="cuda")

        compiled = _compile(M().cuda(), (x, index, y))
        torch.testing.assert_close(compiled(x, index, y), M()(x, index, y))

    def test_negative_dim_uses_fallback(self):
        class M(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_copy.default(x, -1, index, update)

        x = torch.zeros(2, 3, device="cuda")
        index = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
        update = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")

        compiled = _compile(M().cuda(), (x.clone(), index, update))
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(x.clone(), index, update)
        eager = torch.ops.aten.index_copy.default(x.clone(), -1, index, update)
        torch.testing.assert_close(out, eager)

    def test_caches_the_kv_layer_cannot_take_use_the_fallback(self):
        """The KV layer takes float32, float16 and bfloat16 caches only, so a
        one-token write into an integer cache, a buffer or the caller's, takes the
        fallback and keeps its write-back. So does a float64 buffer: with
        ``truncate_double`` the layer would update a float32 copy. A float64 caller
        cache already keeps its write-back without this."""

        class M(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, dtype=dtype))
                self.register_buffer("other", torch.zeros(1, 4, 16, 8, dtype=dtype))

            def forward(self, given, x, input_pos):
                given.index_copy_(2, input_pos, x)
                self.cache.index_copy_(2, input_pos, x)
                self.other[:, :, 5:6, :] = x
                return self.cache.sum(2) + self.other.sum(2) + given.sum(2)

        for dtype, options in (
            (torch.int32, {}),
            (torch.int64, {}),
            (torch.float64, {"truncate_double": True}),
        ):
            with self.subTest(dtype=dtype):

                def step(i):
                    x = torch.full((1, 4, 1, 8), i + 1, device="cuda").to(dtype)
                    return x, torch.tensor([2 + i], device="cuda")

                given = torch.zeros(1, 4, 16, 8, dtype=dtype, device="cuda")
                expected, eager = given.clone(), M(dtype).cuda()
                compiled = _compile(
                    M(dtype).cuda(), (given.clone(),) + step(0), **options
                )
                for i in range(3):
                    out = compiled(given, *step(i))
                    out = out[0] if isinstance(out, tuple) else out
                    want = eager(expected, *step(i))
                    torch.testing.assert_close(out, want, check_dtype=False)
                    torch.testing.assert_close(given, expected)
                    torch.testing.assert_close(compiled.cache, eager.cache)
                    torch.testing.assert_close(compiled.other, eager.other)


class TestCallerTensorsKeepWhatEagerDoes(_CudaGraphsOff):
    """A compiled model changes a caller's tensor exactly when eager PyTorch does."""

    def test_other_writes_into_a_caller_tensor_are_kept(self):
        """An add_ or copy_ into a caller's tensor is no KV cache write, so its
        write-back copy_ stays, after any read of the old contents."""

        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8)

            def forward(self, x, cache):
                cache.add_(x)
                return self.lin(x) * 2

        class Copy(Add):
            def forward(self, x, cache):
                cache.copy_(x * 3)
                return self.lin(x) * 2

        class ReadThenAdd(Add):
            def forward(self, x, cache):
                old = cache * 2
                cache.add_(x)
                return self.lin(x) + old

        x = torch.randn(4, 8, device="cuda")
        for model in (Add(), Copy(), ReadThenAdd()):
            with self.subTest(model=type(model).__name__), torch.no_grad():
                model = model.cuda().eval()
                compiled = _compile(model, (x, torch.zeros(4, 8, device="cuda")))
                given = torch.ones(4, 8, device="cuda")
                expected = given.clone()
                for _ in range(2):
                    out = compiled(x, given)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    # TensorRT may run the linear layer in TF32.
                    torch.testing.assert_close(
                        out, model(x, expected), atol=1e-3, rtol=1e-3
                    )
                    torch.testing.assert_close(given, expected)

    def test_a_float64_caller_cache_keeps_its_write(self):
        """With ``truncate_double`` the engine gets a float32 copy of a float64
        cache, so the write-back ``copy_`` runs in PyTorch. Moving the engine's
        inputs and outputs to float32 must not drop the part that holds it."""

        class M(torch.nn.Module):
            def forward(self, cache, pos, k):
                cache.index_copy_(2, pos, k)
                return cache * 2

        args = (
            torch.zeros(1, 2, 8, 4, device="cuda", dtype=torch.float64),
            torch.tensor([3], device="cuda"),
            torch.ones(1, 2, 1, 4, device="cuda", dtype=torch.float64),
        )
        compiled = _compile(M(), [a.clone() for a in args], truncate_double=True)
        given = [a.clone() for a in args]
        expected = [a.clone() for a in args]
        out = compiled(*given)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, M()(*expected), check_dtype=False)
        torch.testing.assert_close(given[0], expected[0])

    class _WeightedWriter(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.full((1, 4, 1, 8), float(weight), device="cuda")
            )

        def forward(self, cache, x, pos):
            old = torch.index_select(cache, 2, pos)
            cache.index_copy_(2, pos, old + x * self.weight)
            return cache.sum(2)

    @staticmethod
    def _example():
        # Strided, so the engine gets a copy and the write-back stores the write.
        return (
            torch.zeros(1, 16, 4, 8, device="cuda").transpose(1, 2),
            torch.ones(1, 4, 1, 8, device="cuda"),
            torch.tensor([3], device="cuda"),
        )

    def _weighted_call(self, model, compiled):
        given, expected = self._example(), self._example()
        with torch.no_grad():
            for _ in range(2):
                out = compiled(*given)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, model(*expected))
                torch.testing.assert_close(given[0], expected[0])

    def test_a_cache_written_from_a_parameter_compiles_with_gradients_on(self):
        """The written value needs a gradient, so export records the caller's cache
        as needing one too, and compiling with gradients on still works."""
        model = self._WeightedWriter(2).eval()
        compiled = _compile(model, self._example())
        self._weighted_call(model, compiled)

    def test_refit_keeps_the_write_back_of_a_caller_cache(self):
        """The write-back runs in a part left to PyTorch whose result nothing uses.
        Refit keeps it, so the refitted module still stores the write."""
        from torch_tensorrt.dynamo import refit_module_weights

        example = self._example()
        with torch.no_grad():
            compiled = _compile(
                self._WeightedWriter(2), example, immutable_weights=False
            )
        new = torch.export.export(self._WeightedWriter(5), example, strict=False)
        refitted = refit_module_weights(
            compiled, new, arg_inputs=list(example), in_place=True
        )
        self.assertIs(refitted, compiled)
        self._weighted_call(self._WeightedWriter(5), refitted)

    def test_a_graph_too_small_for_an_engine_still_writes_the_caller_cache(self):
        """At the default min_block_size this model gets no engine and runs in
        PyTorch, so its write, predicted in place, is stored by its write-back."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        make = lambda: torch.zeros(1, 4, 16, 8, device="cuda")
        x, pos = torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([3]).cuda()
        compiled = torch_tensorrt.compile(
            export(M(), (make(), x, pos)),
            ir="dynamo",
            inputs=[make(), x, pos],
            use_python_runtime=False,
        )
        given, expected = make(), make()
        for step in range(2):
            pos = torch.tensor([step + 3]).cuda()
            out = compiled(given, x, pos)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, M()(expected, x, pos))
            torch.testing.assert_close(given, expected)

    def test_a_write_left_in_pytorch_next_to_an_engine_still_writes_the_caller_cache(
        self,
    ):
        """At the default min_block_size the write, predicted in place, lands in a
        part left to PyTorch next to an engine, which stores it by its write-back."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                for _ in range(8):
                    x = x * 1.1 + 0.1
                cache.index_copy_(2, pos, torch.cumsum(x, 3))
                return cache.sum(2)

        make = lambda: torch.zeros(1, 4, 16, 8, device="cuda")
        x = torch.ones(1, 4, 1, 8, device="cuda")
        compiled = torch_tensorrt.compile(
            export(M(), (make(), x, torch.tensor([3]).cuda())),
            ir="dynamo",
            inputs=[make(), x, torch.tensor([3]).cuda()],
            torch_executed_ops={"torch.ops.aten.cumsum.default"},
            use_python_runtime=False,
        )
        given, expected = make(), make()
        for step in range(2):
            pos = torch.tensor([step + 3]).cuda()
            out = compiled(given, x, pos)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, M()(expected, x, pos))
            torch.testing.assert_close(given, expected)

    def test_a_float64_cache_buffer_keeps_its_write(self):
        """With ``truncate_double`` the engine gets a float32 copy of a float64
        buffer, so the buffer is copied back rather than updated in place."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "cache",
                    torch.zeros(1, 4, 16, 8, device="cuda", dtype=torch.float64),
                )

            def forward(self, x, pos):
                self.cache.index_copy_(2, pos, x)
                return self.cache.sum(2)

        x = torch.ones(1, 4, 1, 8, device="cuda", dtype=torch.float64)
        for full in (False, True):
            with self.subTest(require_full_compilation=full):
                eager = M()
                compiled = _compile(
                    M(),
                    (x, torch.tensor([3]).cuda()),
                    truncate_double=True,
                    require_full_compilation=full,
                )
                for step in range(3):
                    pos = torch.tensor([step + 3]).cuda()
                    out = compiled(x, pos)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    torch.testing.assert_close(out, eager(x, pos), check_dtype=False)
                    torch.testing.assert_close(
                        dict(compiled.named_buffers())["cache"], eager.cache
                    )

    def test_a_write_back_comes_after_the_readers_of_the_old_contents(self):
        """The write-back runs in PyTorch, where the partitioner orders it only
        after the value it stores. The block that reads the old contents of the
        cache comes later here, so it reads a copy taken before the write."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                b = torch.cumsum(x.relu(), 2)
                r = (cache * b.sum()).sum()
                cache.index_copy_(2, pos, x)
                return r

        make = lambda: torch.arange(512.0, device="cuda").reshape(1, 4, 16, 8) / 100
        x, pos = torch.full((1, 4, 1, 8), 5.0, device="cuda"), torch.tensor([3]).cuda()
        compiled = _compile(
            M(), (make(), x, pos), torch_executed_ops={"torch.ops.aten.cumsum.default"}
        )
        given, expected = make(), make()
        out = compiled(given, x, pos)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, M()(expected, x, pos))
        torch.testing.assert_close(given, expected)

    def test_a_read_of_the_old_contents_of_a_cache_sees_them(self):
        """A buffer stored from a caller's cache, and a clone or sum of a cache
        buffer, read the contents from before the write, as in eager."""

        class StoredFromCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, cache, x, pos):
                self.buf.copy_(cache)
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        class ReadBeforeWrite(torch.nn.Module):
            def __init__(self, read):
                super().__init__()
                self.read = read
                self.register_buffer(
                    "cache", torch.arange(512.0, device="cuda").reshape(1, 4, 16, 8)
                )

            def forward(self, x, pos):
                old = self.read(self.cache)
                self.cache.index_copy_(2, pos, x)
                return old.sum() + self.cache.sum()

        x = torch.full((1, 4, 1, 8), 1000.0, device="cuda")
        make = lambda: torch.ones(1, 4, 16, 8, device="cuda")
        eager, compiled = StoredFromCache(), _compile(
            StoredFromCache(), (make(), x, torch.tensor([3]).cuda())
        )
        given, expected = make(), make()
        for step in range(2):
            pos = torch.tensor([step + 3]).cuda()
            compiled(given, x, pos)
            eager(expected, x, pos)
            torch.testing.assert_close(given, expected)
            torch.testing.assert_close(dict(compiled.named_buffers())["buf"], eager.buf)
        for name, read in (("clone", torch.clone), ("sum", torch.sum)):
            with self.subTest(read=name):
                eager = ReadBeforeWrite(read)
                compiled = _compile(
                    ReadBeforeWrite(read), (x, torch.tensor([3]).cuda())
                )
                for step in range(2):
                    pos = torch.tensor([step + 3]).cuda()
                    out = compiled(x, pos)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    torch.testing.assert_close(out, eager(x, pos))

    def test_a_buffer_write_wrapped_in_a_copy_is_done_in_place(self):
        """The check looks through copy(cache, write) to the write, so the mark goes
        on the write the converter reads."""

        class CopyWrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8, device="cuda"))

            def forward(self, x, pos):
                write = torch.index_copy(self.cache, 2, pos, x)
                self.cache.copy_(torch.ops.aten.copy(self.cache, write))
                return self.cache.sum(2)

        x = torch.ones(1, 4, 1, 8, device="cuda")
        eager = CopyWrapped()
        compiled = _compile(CopyWrapped(), (x, torch.tensor([3]).cuda()))
        self.assertEqual(len(_aliased_io(compiled)), 1)
        for step in range(2):
            pos = torch.tensor([step + 3]).cuda()
            out = compiled(x * (step + 1), pos)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, eager(x * (step + 1), pos))

    def test_a_write_back_copies_the_cache_only_for_readers_of_its_old_contents(self):
        """The copy is of the whole cache, so it is taken only when something reads
        the old contents: not for export's guards, nor for the copy a functional
        write into the cache becomes."""
        from torch_tensorrt.dynamo import _compiler

        class Functional(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.copy_(torch.index_copy(cache, 2, pos, x))
                return cache.sum(2)

        class OldReader(torch.nn.Module):
            def forward(self, cache, x, pos):
                r = cache.sum()
                cache.index_copy_(2, pos, x)
                return cache.sum(2) + r

        real = _compiler.lift_mutated_buffers

        def lift(gm, *args, **kwargs):
            gm, lifted = real(gm, *args, **kwargs)
            copies.append(
                sum(
                    n.target is torch.ops.aten.clone.default
                    and n.args[0].op == "placeholder"
                    for n in gm.graph.nodes
                )
            )
            return gm, lifted

        make = lambda: torch.zeros(1, 16, 4, 8, device="cuda").transpose(1, 2)
        x, pos = torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([3]).cuda()
        for model, expected in ((Functional(), [0]), (OldReader(), [1])):
            with self.subTest(model=type(model).__name__):
                copies = []
                with mock.patch.object(_compiler, "lift_mutated_buffers", lift):
                    compiled = _compile(model, (make(), x, pos))
                self.assertEqual(copies, expected)
                given, want = make(), make()
                out = compiled(given, x, pos)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, model(want, x, pos))
                torch.testing.assert_close(given, want)

    def test_the_engine_cache_key_marks_a_write_not_done_in_place(self):
        """An engine cached before the mark existed may update its input in place,
        so a write not done in place gets its own part of the key."""
        import logging

        from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
        from torch_tensorrt.dynamo._settings import CompilationSettings

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                return torch.index_copy(cache, 2, pos, x).sum(2)

        args = (
            torch.zeros(1, 4, 16, 8, device="cuda"),
            torch.ones(1, 4, 1, 8, device="cuda"),
            torch.tensor([3]).cuda(),
        )
        gm = export(M(), args).module()
        specs = [torch_tensorrt.Input.from_tensor(a) for a in args]
        with self.assertLogs(
            "torch_tensorrt.dynamo._engine_cache", level=logging.DEBUG
        ) as logs:
            unmarked = BaseEngineCache.get_hash(gm, specs, CompilationSettings())
        graph = next(line for line in logs.output if "graph_str" in line)
        self.assertIn("index_copy.default", graph)
        self.assertIn("_trt_no_kv_alias", graph)
        (write,) = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.index_copy.default
        )
        write.meta["_trt_kv_alias"] = True
        marked = BaseEngineCache.get_hash(gm, specs, CompilationSettings())
        self.assertNotEqual(marked, unmarked)

    def test_a_cached_engine_is_not_reused_for_a_renamed_cache(self):
        """The engine's aliased_io names the input it updates, so a model whose cache
        argument has another name builds its own engine."""

        class Cache(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return cache.sum(2)

        class Kv(torch.nn.Module):
            def forward(self, kv, x, pos):
                kv.index_copy_(2, pos, x)
                return kv.sum(2)

        make = lambda: torch.zeros(1, 4, 16, 8, device="cuda")
        x, pos = torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([3]).cuda()
        with tempfile.TemporaryDirectory() as cache_dir:
            for model in (Cache(), Kv()):
                compiled = _compile(
                    model,
                    (make(), x, pos),
                    immutable_weights=False,
                    cache_built_engines=True,
                    reuse_cached_engines=True,
                    engine_cache_dir=cache_dir,
                )
                given, expected = make(), make()
                out = compiled(given, x, pos)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, model(expected, x, pos))
                torch.testing.assert_close(given, expected)

    def test_the_engine_cache_keeps_in_place_and_write_back_engines_apart(self):
        """The same graph writes a contiguous cache in place and a strided one
        through its write-back, so the two engines need different cache keys."""

        class M(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(2, pos, x)
                return cache.sum(2), cache

        caches = (
            lambda: torch.zeros(1, 16, 4, 8, device="cuda").transpose(1, 2),
            lambda: torch.zeros(1, 4, 16, 8, device="cuda"),
        )
        x, pos = torch.ones(1, 4, 1, 8, device="cuda"), torch.tensor([3]).cuda()
        with tempfile.TemporaryDirectory() as cache_dir:
            for make in caches:
                compiled = _compile(
                    M(),
                    (make(), x, pos),
                    immutable_weights=False,
                    cache_built_engines=True,
                    reuse_cached_engines=True,
                    engine_cache_dir=cache_dir,
                )
                given, expected = make(), make()
                out = compiled(given, x, pos)
                torch.testing.assert_close(tuple(out), M()(expected, x, pos))
                torch.testing.assert_close(given, expected)


class TestEntryPointsWithoutLiftingKeepTheWrite(_CudaGraphsOff):
    """The torch.compile backend, the serialized engine and the Windows cross compile
    do not lift buffers, and they still store an in-place write of a caller's
    tensor: in place when the engine can, otherwise through its write-back."""

    class _Writer(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache.sum(2)

    @staticmethod
    def _step(n):
        g = torch.Generator().manual_seed(n)
        return torch.randn(1, 4, 1, 8, generator=g).cuda(), torch.tensor(
            [2 + n], device="cuda"
        )

    def _check_steps(self, compiled, eager, cache, expected):
        for n in range(3):
            x, pos = self._step(n)
            torch.testing.assert_close(compiled(cache, x, pos), eager(expected, x, pos))
            torch.testing.assert_close(cache, expected)

    def test_torch_compile_stores_the_write_of_a_caller_cache(self):
        caches = {
            "contiguous": (
                lambda: torch.zeros(1, 4, 16, 8, device="cuda"),
                {"arg0_1": "kv_cache_update"},
            ),
            "other strides": (
                lambda: torch.zeros(1, 16, 4, 8, device="cuda").transpose(1, 2),
                {},
            ),
        }
        from torch_tensorrt.dynamo.backend.backends import torch_tensorrt_backend

        for name, (make, aliased) in caches.items():
            with self.subTest(cache=name):
                torch._dynamo.reset()
                built = []

                def backend(gm, sample_inputs):
                    built.append(
                        torch_tensorrt_backend(
                            gm,
                            sample_inputs,
                            options={
                                "min_block_size": 1,
                                "pass_through_build_failures": True,
                            },
                        )
                    )
                    return built[-1]

                compiled = torch.compile(
                    self._Writer(), backend=backend, fullgraph=True
                )
                self._check_steps(compiled, self._Writer(), make(), make())
                self.assertEqual(dict(_aliased_io(built[0]).values()), aliased)
        torch._dynamo.reset()

    def test_torch_compile_stores_the_write_of_a_graph_too_small_for_an_engine(self):
        """At the default min_block_size the backend builds no engine, so the write
        predicted in place is stored by its write-back, as in compile()."""
        torch._dynamo.reset()
        compiled = torch.compile(
            self._Writer(),
            backend="torch_tensorrt",
            options={"pass_through_build_failures": True},
        )
        cache, expected = self._example()[0], self._example()[0]
        self._check_steps(compiled, self._Writer(), cache, expected)
        torch._dynamo.reset()

    def test_a_write_without_gradients_into_a_tensor_that_needs_one(self):
        """A model may write a tensor that needs a gradient inside its own
        torch.no_grad(). The write-back left to PyTorch runs without autograd too."""

        class Input(torch.nn.Module):
            def forward(self, cache, x, pos):
                with torch.no_grad():
                    cache.index_copy_(2, pos, x)
                return cache.sum(2)

        class ParameterCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache = torch.nn.Parameter(torch.zeros(1, 4, 16, 8).cuda())

            def forward(self, x, pos):
                with torch.no_grad():
                    self.cache.index_copy_(2, pos, x)
                return self.cache.sum(2)

        make = lambda: torch.zeros(1, 4, 16, 8, device="cuda", requires_grad=True)
        x, pos = self._step(0)
        options = {"min_block_size": 99, "pass_through_build_failures": True}
        with self.subTest(entry="torch.compile", cache="input"):
            torch._dynamo.reset()
            compiled = torch.compile(Input(), backend="torch_tensorrt", options=options)
            given, expected = make(), make()
            torch.testing.assert_close(
                compiled(given, x, pos), Input()(expected, x, pos)
            )
            torch.testing.assert_close(given, expected)
        with self.subTest(entry="torch.compile", cache="parameter"):
            torch._dynamo.reset()
            model, eager = ParameterCache(), ParameterCache()
            compiled = torch.compile(model, backend="torch_tensorrt", options=options)
            torch.testing.assert_close(compiled(x, pos), eager(x, pos))
            torch.testing.assert_close(model.cache, eager.cache)
        torch._dynamo.reset()
        for traced in (make().detach(), make()):
            with self.subTest(entry="compile", traced=traced.requires_grad):
                compiled = torch_tensorrt.dynamo.compile(
                    export(Input(), (traced, x, pos)),
                    arg_inputs=[traced, x, pos],
                    min_block_size=99,
                )
                given, expected = make(), make()
                torch.testing.assert_close(
                    compiled(given, x, pos), Input()(expected, x, pos)
                )
                torch.testing.assert_close(given, expected)

    class _ViewReturner(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache.reshape(1, 4, 16, 8)

    def test_torch_compile_stores_the_write_of_an_input_it_returns_a_view_of(self):
        """AOT refuses an output that is a view of another output, as the written
        input's final value and the returned reshape of it would be. The fallback
        runs the graph as Dynamo gave it: lowering changes the graph in place, and in
        the changed graph the write lands in a clone of the input."""
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                torch._dynamo.reset()
                build = (
                    mock.patch(
                        "torch_tensorrt.dynamo.backend.backends.compile_module",
                        side_effect=RuntimeError("build failed"),
                    )
                    if fallback
                    else contextlib.nullcontext()
                )
                with build:
                    compiled = torch.compile(
                        self._ViewReturner(),
                        backend="torch_tensorrt",
                        options={
                            "min_block_size": 1,
                            "pass_through_build_failures": not fallback,
                        },
                    )
                    cache, expected = (
                        torch.zeros(1, 4, 16, 8).cuda() for _ in range(2)
                    )
                    self._check_steps(compiled, self._ViewReturner(), cache, expected)
        torch._dynamo.reset()

    def _example(self):
        return (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)

    def test_full_compilation_updates_in_place_or_names_the_input(self):
        """With full support compile returns the engine module itself, whose
        ``aliased_io`` the check must read too. A write that is not done in place
        needs its write-back in PyTorch: the KV layer takes only a rank-4 cache."""
        example = self._example()
        compiled = torch_tensorrt.dynamo.compile(
            export(self._Writer(), example),
            arg_inputs=list(example),
            min_block_size=1,
            require_full_compilation=True,
        )
        cache, expected = example[0].clone(), example[0].clone()
        self._check_steps(compiled, self._Writer(), cache, expected)

        class Rank3(torch.nn.Module):
            def forward(self, cache, x, pos):
                cache.index_copy_(1, pos, x)
                return cache.sum(1)

        example = (
            torch.zeros(4, 16, 8, device="cuda"),
            torch.ones(4, 1, 8, device="cuda"),
            torch.tensor([3], device="cuda"),
        )
        with self.assertRaisesRegex(
            AssertionError,
            r"cannot update input\(s\) \['cache'\].*contiguous tensor on the GPU",
        ):
            torch_tensorrt.dynamo.compile(
                export(Rank3(), example),
                arg_inputs=list(example),
                min_block_size=1,
                require_full_compilation=True,
            )

    def test_the_serialized_engine_updates_in_place_or_refuses(self):
        """The caller binds the engine's memory, so a host example still gives a
        cache updated in place. A write the engine cannot do in place is refused,
        since nothing would store it."""
        import tensorrt as trt

        example = tuple(t.cpu() for t in self._example())
        engine = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(self._Writer(), example),
                arg_inputs=list(example),
                min_block_size=1,
            )
        )
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
            bytes(engine)
        )
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.assertIn("cache", [engine.get_aliased_input_tensor(n) for n in names])

        class Strided(torch.nn.Module):
            def forward(self, cache, x):
                cache[:, :, 0:8:2, :] = x
                return cache.sum(2)

        example = (
            torch.zeros(1, 4, 16, 8, device="cuda"),
            torch.ones(1, 4, 4, 8, device="cuda"),
        )
        with self.assertRaisesRegex(
            RuntimeError, r"cannot update input\(s\) \['cache'\]"
        ):
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                export(Strided(), example), arg_inputs=list(example), min_block_size=1
            )

    def test_a_saved_module_keeps_the_write_back_or_refuses(self):
        """The part left to PyTorch that stores the cache is unused, so saving has
        to keep it: a program retraced with dynamic shapes stores the write, and the
        legacy exporter refuses the copy_ instead of dropping it."""
        if not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("the Python runtime cannot retrace with dynamic shapes")

        class M(torch.nn.Module):
            def forward(self, cache, x, pos, y):
                cache.index_copy_(1, pos, x)
                return cache.sum(1) + y.sum()

        make = lambda: torch.zeros(4, 16, 8, device="cuda")
        x, pos = torch.ones(4, 1, 8, device="cuda"), torch.tensor([2], device="cuda")
        y = torch.ones(3, device="cuda")
        example = (make(), x, pos, y)
        dynamic = {
            "cache": None,
            "x": None,
            "pos": None,
            "y": {0: torch.export.Dim("n", min=2, max=8)},
        }
        inputs = [torch_tensorrt.Input.from_tensor(t) for t in example[:3]] + [
            torch_tensorrt.Input(min_shape=(2,), opt_shape=(3,), max_shape=(8,))
        ]
        compiled = torch_tensorrt.dynamo.compile(
            export(M(), example, dynamic_shapes=dynamic),
            arg_inputs=inputs,
            min_block_size=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/m.ep"
            torch_tensorrt.save(
                compiled,
                path,
                output_format="exported_program",
                arg_inputs=list(example),
                dynamic_shapes=dynamic,
                retrace=True,
            )
            loaded = torch.export.load(path).module()
            given, expected = make(), make()
            torch.testing.assert_close(
                loaded(given, x, pos, y), M()(expected, x, pos, y)
            )
            torch.testing.assert_close(given, expected)
            with self.assertRaisesRegex(Exception, "copy_"):
                torch_tensorrt.save(
                    compiled,
                    path,
                    output_format="exported_program",
                    arg_inputs=list(example),
                    retrace=False,
                )

    def test_the_windows_cross_compile_marks_the_write(self):
        """Building for Windows needs a Windows-enabled build, so this checks the
        graph the build would be handed."""
        from torch_tensorrt import _features
        from torch_tensorrt.dynamo import _compiler

        handed = []

        def compile_module(gm, *args, **kwargs):
            handed.append(gm)
            return gm

        example = self._example()
        enabled = _features.ENABLED_FEATURES._replace(windows_cross_compile=True)
        with (
            mock.patch.object(_features, "ENABLED_FEATURES", enabled),
            mock.patch.object(_compiler, "compile_module", compile_module),
        ):
            torch_tensorrt.dynamo.cross_compile_for_windows(
                export(self._Writer(), example),
                arg_inputs=list(example),
                min_block_size=1,
                dryrun=True,
            )
        (write,) = [
            n
            for n in handed[0].graph.nodes
            if n.target is torch.ops.aten.index_copy.default
        ]
        self.assertTrue(write.meta.get("_trt_kv_alias"))


def _python_runtime():
    """The Python runtime runs the engine when the C++ runtime library is not
    installed, which is the only way to select it."""
    from torch_tensorrt.dynamo.runtime import _TorchTensorRTModule as runtime

    without_cpp = runtime.ENABLED_FEATURES._replace(torch_tensorrt_runtime=False)
    return mock.patch.object(runtime, "ENABLED_FEATURES", without_cpp)


def _python_engine_op():
    """The C++ op is loaded here, so the Python engine is called as its own op
    calls it, which returns a copy of an output in an input's memory."""

    def python_op(input_tensors, engine):
        outputs = engine.execute(input_tensors)
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
        inputs = {tensor.untyped_storage()._cdata for tensor in input_tensors}
        return [
            output.clone() if output.untyped_storage()._cdata in inputs else output
            for output in outputs
        ]

    return mock.patch.object(torch.ops.tensorrt, "execute_engine", python_op)


class TestCallerCacheTheEngineGetsACopyOf(_CudaGraphsOff):
    """A caller's cache with other strides, or on the host, reaches the engine as a
    copy. The runtime stores the copy's update, in both engine runtimes."""

    @staticmethod
    def _step(n):
        g = torch.Generator().manual_seed(n)
        x = torch.randn(1, 4, 1, 8, generator=g).cuda()
        return x, torch.tensor([2 + n], device="cuda")

    class _SumWriter(torch.nn.Module):
        def forward(self, cache, x, pos):
            cache.index_copy_(2, pos, x)
            return cache.sum(2)

    _COPIED = {
        "other strides": lambda: torch.randn(1, 16, 4, 8, device="cuda").transpose(
            1, 2
        ),
        "host": lambda: torch.randn(1, 4, 16, 8),
        "negative view": lambda: torch._neg_view(
            torch.randn(1, 4, 16, 8, device="cuda")
        ),
    }

    @staticmethod
    def _both_runtimes(model, example):
        modules = {"cpp": (_compile(model, example), contextlib.nullcontext)}
        with _python_runtime():
            modules["python"] = (_compile(model, example), _python_engine_op)
        return modules

    def test_torch_compile_traces_a_module_with_an_updated_input(self):
        """Dynamo traces the module's forward in one graph, as it did before the
        runtime checked the caller's caches."""
        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        module = _compile(self._SumWriter(), example)
        torch._dynamo.reset()
        traced = torch.compile(module, backend="eager", fullgraph=True)
        cache, expected = torch.zeros(1, 4, 16, 8, device="cuda"), example[0].clone()
        for n in range(2):
            x, pos = self._step(n)
            out = traced(cache, x, pos)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.testing.assert_close(out, self._SumWriter()(expected, x, pos))
            torch.testing.assert_close(cache, expected)

    def test_a_cache_the_model_does_not_return(self):
        """No write-back follows the engine, so the runtime stores the copy. A recorded
        CUDA graph writes a cache at the address it was recorded with, so it records
        again for a new cache and refuses a copy, which is gone after the call."""
        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        modules = self._both_runtimes(self._SumWriter(), example)
        for kind, (module, engine_op) in modules.items():
            self.assertEqual(len(_aliased_io(module)), 1)
            with engine_op():
                for name, make in self._COPIED.items():
                    with self.subTest(runtime=kind, cache=name):
                        cache = make()
                        expected = cache.clone()
                        for n in range(3):
                            x, pos = self._step(n)
                            out = module(cache, x, pos)
                            want = self._SumWriter()(
                                expected, x.to(expected.device), pos.to(expected.device)
                            )
                            torch.testing.assert_close(out.cpu(), want.cpu())
                            torch.testing.assert_close(cache, expected)
                with torch_tensorrt.runtime.enable_cudagraphs(module) as graphed:
                    with self.subTest(runtime=kind, cudagraphs="two caches"):
                        caches = [
                            torch.randn(1, 4, 16, 8, device="cuda") for _ in range(2)
                        ]
                        expected = [c.clone() for c in caches]
                        for n in range(4):
                            x, pos = self._step(n)
                            out = graphed(caches[n % 2], x, pos)
                            want = self._SumWriter()(expected[n % 2], x, pos)
                            torch.testing.assert_close(out, want)
                            for got, exp in zip(caches, expected):
                                torch.testing.assert_close(got, exp)
                    for name, make in self._COPIED.items():
                        with self.subTest(runtime=kind, cudagraphs=name):
                            with self.assertRaisesRegex(RuntimeError, "cannot update"):
                                graphed(make(), *self._step(0))
                    with self.subTest(runtime=kind, cudagraphs="after a failed call"):
                        # The next call records again, with the cache recorded last
                        # or with the one the failed call brought.
                        cache = torch.randn(1, 4, 16, 8, device="cuda")
                        targets = [(caches[1], expected[1]), (cache, cache.clone())]
                        with self.assertRaises((AssertionError, RuntimeError)):
                            x, pos = self._step(0)
                            graphed(cache, x.half(), pos)
                        for n in range(4):
                            got, want_cache = targets[n % 2]
                            x, pos = self._step(n)
                            out = graphed(got, x, pos)
                            want = self._SumWriter()(want_cache, x, pos)
                            torch.testing.assert_close(out, want)
                            torch.testing.assert_close(got, want_cache)

    def test_a_copied_cache_that_needs_a_gradient(self):
        """The runtime stores the copy's update without autograd, as the model's own
        write under no_grad does, so a cache that needs a gradient still gets it."""
        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        modules = self._both_runtimes(self._SumWriter(), example)
        caches = {
            "host": lambda: torch.zeros(1, 4, 16, 8, requires_grad=True),
            "other strides": lambda: torch.zeros(
                1, 16, 4, 8, device="cuda", requires_grad=True
            ).transpose(1, 2),
        }
        for kind, (module, engine_op) in modules.items():
            for name, make in caches.items():
                with self.subTest(runtime=kind, cache=name), engine_op():
                    cache = make()
                    expected = cache.detach().clone()
                    x, pos = self._step(0)
                    module(cache, x, pos)
                    self._SumWriter()(
                        expected, x.to(expected.device), pos.to(expected.device)
                    )
                    torch.testing.assert_close(cache.detach(), expected)

    def test_a_module_made_without_an_alias_map(self):
        """The alias map is optional when a module is made from a plan, so the module
        reads the alias the engine takes from its KV cache layer."""
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        for kind, runtime, engine_op in (
            ("cpp", contextlib.nullcontext, contextlib.nullcontext),
            ("python", _python_runtime, _python_engine_op),
        ):
            with self.subTest(runtime=kind), runtime():
                compiled = _compile(self._SumWriter(), example)
                built = next(
                    m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)
                )
                module = TorchTensorRTModule(
                    bytes(built.serialized_engine),
                    input_binding_names=list(built.input_binding_names),
                    output_binding_names=list(built.output_binding_names),
                    settings=built.settings,
                    num_user_outputs=built.num_user_outputs,
                )
                self.assertEqual(module.aliased_io, {})
                cache = torch.randn(1, 4, 16, 8)
                expected = cache.clone()
                x, pos = self._step(0)
                by_name = {"cache": cache, "x": x, "pos": pos}
                with engine_op():
                    out = module(*(by_name[n] for n in module.input_binding_names))
                want = self._SumWriter()(expected, x.cpu(), pos.cpu())
                torch.testing.assert_close(out.cpu(), want)
                torch.testing.assert_close(cache, expected)

    class _TwoCaches(torch.nn.Module):
        def forward(self, k, v, x, pos):
            k.index_copy_(2, pos, x)
            v.index_copy_(2, pos + 5, x * 2)
            return k.sum(2) + v.sum(2)

    def test_two_caches(self):
        """Copies of two caches that share an element would both write it back, so the
        module and the engine operator a saved program calls refuse such a call."""
        zeros = torch.zeros(1, 4, 16, 8, device="cuda")
        example = (zeros, zeros.clone()) + self._step(0)
        x, pos = self._step(0)

        def interleaved():
            kv = torch.randn(1, 4, 16, 8, 2, device="cuda")
            return kv[..., 0], kv[..., 1]

        def packed():
            kv = torch.randn(1, 16, 2, 4, 8, device="cuda")
            return kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        def interleaved_with_other_strides_on_a_size_1_dim():
            k, v = interleaved()
            return k, v.as_strided(v.shape, (1,) + v.stride()[1:], v.storage_offset())

        modules = self._both_runtimes(self._TwoCaches(), example)
        for kind, (module, engine_op) in modules.items():
            self.assertEqual(len(_aliased_io(module)), 2)
            engine = next(m for m in module.modules() if getattr(m, "aliased_io", None))
            with engine_op():
                for name in ("other strides", "host"):
                    cache = self._COPIED[name]()
                    by_name = {"k": cache, "v": cache, "x": x, "pos": pos}
                    with self.subTest(runtime=kind, caches=name):
                        with self.assertRaisesRegex(RuntimeError, "share memory"):
                            module(cache, cache, x, pos)
                        with self.assertRaisesRegex(RuntimeError, "share memory"):
                            torch.ops.tensorrt.execute_engine(
                                [by_name[n] for n in engine.input_binding_names],
                                engine.engine,
                            )
                for make in (
                    interleaved,
                    packed,
                    interleaved_with_other_strides_on_a_size_1_dim,
                ):
                    with self.subTest(runtime=kind, caches=make.__name__):
                        caches = make()
                        expected = [c.clone() for c in caches]
                        out = module(*caches, x, pos)
                        torch.testing.assert_close(
                            out, self._TwoCaches()(*expected, x, pos)
                        )
                        for got, exp in zip(caches, expected):
                            torch.testing.assert_close(got, exp)

    def test_the_check_costs_little_when_no_cache_is_copied(self):
        """With contiguous CUDA caches the module adds no check, and the engine
        compares only inputs whose bytes overlap, with no memory, reading each
        input's layout once."""
        from torch_tensorrt.dynamo.runtime import _TRTEngine

        def check(named):
            _TRTEngine._refuse_inputs_that_share_memory(
                [(name, t, True, True) for name, t in named]
            )

        with (
            mock.patch.object(
                _TRTEngine, "_layouts_share", wraps=_TRTEngine._layouts_share
            ) as compares,
            mock.patch.object(
                _TRTEngine, "_layout", wraps=_TRTEngine._layout
            ) as layouts,
        ):
            check([(f"c{i}", torch.zeros(4, device="cuda")) for i in range(56)])
            self.assertEqual(compares.call_count, 0)
            # 28 layers, each a large buffer holding its key and value interleaved.
            layers = [torch.zeros(1, 8, 256, 64, 2, device="cuda") for _ in range(28)]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.max_memory_allocated()
            check(
                [
                    (f"{n}{i}", kv[..., j])
                    for i, kv in enumerate(layers)
                    for j, n in enumerate("kv")
                ]
            )
            self.assertEqual(torch.cuda.max_memory_allocated(), before)
            self.assertEqual(compares.call_count, 28)
            # 8 caches interleaved in one buffer: every pair overlaps.
            one = torch.zeros(1, 8, 256, 64, 8, device="cuda")
            layouts.reset_mock()
            check([(f"c{i}", one[..., i]) for i in range(8)])
            self.assertEqual(layouts.call_count, 8)

        # The Python engine runs the check itself on every call, so only the C++
        # runtime shows what the module adds.
        if not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            return
        example = (torch.zeros(1, 4, 16, 8, device="cuda"),) + self._step(0)
        compiled = _compile(self._SumWriter(), example)
        with mock.patch.object(
            _TRTEngine,
            "_refuse_inputs_that_share_memory",
            wraps=_TRTEngine._refuse_inputs_that_share_memory,
        ) as module_checks:
            compiled(torch.zeros(1, 4, 16, 8, device="cuda"), *self._step(1))
            self.assertEqual(module_checks.call_count, 0)
            compiled(torch.zeros(1, 4, 16, 8), *self._step(1))
            self.assertEqual(module_checks.call_count, 1)


class TestInputsInTheMemoryOfAnUpdatedCache(_CudaGraphsOff):
    """The engine reads its other inputs before or after it updates a KV cache, so an
    input that shares an element with the cache is refused. The C++ check runs
    through its engine operator."""

    class _LateReader(torch.nn.Module):
        def forward(self, cache, reader, x, pos):
            cache.index_copy_(2, pos, x)
            return reader * cache.sum()

    def _check(self, calls=contextlib.nullcontext):
        # Each call makes the same values, for the compiled and the eager model.
        def values(*shape, seed=0):
            g = torch.Generator().manual_seed(seed)
            return torch.randn(*shape, generator=g).cuda()

        def separate():
            return values(1, 4, 16, 8), values(1, 4, 16, 8, seed=1)

        def even_and_odd_elements():
            # Only the strides' common divisor tells these apart, since the stride 16
            # is within the span of the smaller one.
            both = values(1136)
            return tuple(
                both.as_strided((1, 4, 16, 8), (1136, 284, 16, 6), offset)
                for offset in (0, 1)
            )

        model = self._LateReader()
        x, pos = torch.ones(1, 4, 1, 8).cuda(), torch.tensor([3], device="cuda")
        compiled = _compile(model, [*separate(), x, pos])
        self.assertEqual(len(_aliased_io(compiled)), 1)
        with calls():
            for make in (separate, even_and_odd_elements):
                with self.subTest(make.__name__):
                    given, expected = make(), make()
                    out = compiled(*given, x, pos)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    torch.testing.assert_close(out, model(*expected, x, pos))
                    torch.testing.assert_close(given[0], expected[0])
            # A refused call runs nothing, so the cases can share one cache.
            cache, b = values(1, 4, 16, 8), values(1, 4, 17, 9)
            for name, inputs in (
                ("a view", (cache, cache.view_as(cache))),
                ("a strided view", (cache, cache.view(1, 4, 8, 16).transpose(2, 3))),
                ("a negative view", (cache, torch._neg_view(cache))),
                ("a signed stride step away", (b[:, :, :16, 1:], b[:, :, 1:, :8])),
            ):
                with self.subTest(name):
                    with self.assertRaisesRegex(RuntimeError, "share memory"):
                        compiled(*inputs, x, pos)

    def test_the_python_check_never_misses_a_shared_element(self):
        """Views of ``arange`` hold their own offsets, so they show which elements two
        views share."""
        from torch_tensorrt.dynamo.runtime._TRTEngine import _layout, _layouts_share

        storage = torch.arange(700, dtype=torch.float32)

        def shared(a, b):
            return bool(set(a.flatten().tolist()) & set(b.flatten().tolist()))

        rng = random.Random(0)

        def layout():
            sizes = [rng.randint(1, 4) for _ in range(rng.randint(1, 4))]
            return sizes, [rng.randint(1, 25) for _ in sizes]

        wrong = []
        for _ in range(5000):
            a_sizes, a_strides = layout()
            if rng.random() < 0.6:
                order = rng.sample(range(len(a_sizes)), len(a_sizes))
                b_sizes = [a_sizes[d] for d in order]
                b_strides = [
                    a_strides[d] if a_sizes[d] > 1 else rng.randint(1, 30)
                    for d in order
                ]
            else:
                b_sizes, b_strides = layout()
            a_offset, b_offset = rng.randint(0, 40), rng.randint(0, 40)
            a = storage.as_strided(a_sizes, a_strides, a_offset)
            b = storage.as_strided(b_sizes, b_strides, b_offset)
            if shared(a, b) and not _layouts_share(_layout(a), _layout(b)):
                wrong.append(
                    (a_sizes, a_strides, a_offset, b_sizes, b_strides, b_offset)
                )
        self.assertEqual(wrong[:5], [])

    def test_the_cpp_runtime(self):
        self._check()

    def test_the_python_runtime(self):
        with _python_runtime():
            self._check(calls=_python_engine_op)

    @staticmethod
    def _cpp_and_python(builder, network, name, inputs, outputs, aliased_io):
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

        plan = bytes(
            builder.build_serialized_network(network, builder.create_builder_config())
        )
        module = TorchTensorRTModule(
            plan,
            inputs,
            outputs,
            name=name,
            settings=CompilationSettings(device=torch_tensorrt.Device("cuda:0")),
            aliased_io=aliased_io,
            num_user_outputs=1,
        )
        engine = TRTEngine(module._pack_engine_info())
        return (("cpp", module), ("python", lambda *args: engine.execute(args)))

    def test_a_plugin_input_may_share_another_input_but_not_a_cache(self):
        """A plugin's alias is not a KV cache: ``x.add_(y)`` with the same tensor for
        ``x`` and ``y`` reads both before it writes. Inputs that overlap only in part
        are refused, as eager refuses them: the engine writes ``x`` while it still
        reads ``y``. Sharing a cache is refused: the engine updates both in place, so
        a private copy of the plugin's input would take its update."""
        import tensorrt as trt

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(0)
        added = network.add_elementwise(
            network.add_input("x", trt.float32, (16,)),
            network.add_input("y", trt.float32, (16,)),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        added.name = "updated_x"
        network.mark_output(added)
        runs = self._cpp_and_python(
            builder,
            network,
            "user_alias_add",
            ["x", "y"],
            ["updated_x"],
            {"updated_x": ("x", "user")},
        )
        for name, run in runs:
            with self.subTest(name):
                x = torch.arange(1.0, 17.0, device="cuda")
                out = run(x, x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                torch.testing.assert_close(out, torch.arange(1.0, 17.0).cuda() * 2)
                both = torch.arange(1.0, 21.0, device="cuda")
                with self.assertRaisesRegex(RuntimeError, "share memory"):
                    run(both[:16], both[4:])

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(0)

        def total_of(tensor):
            return network.add_reduce(
                tensor, trt.ReduceOperation.SUM, 15, True
            ).get_output(0)

        cache = network.add_input("cache", trt.float32, (1, 4, 16, 8))
        reader = network.add_input("reader", trt.float32, (1, 4, 16, 8))
        update = network.add_input("update", trt.float32, (1, 4, 1, 8))
        pos = network.add_input("pos", trt.int32, (1,))
        written = network.add_kv_cache_update(
            cache, update, pos, trt.KVCacheMode.LINEAR
        ).get_output(0)
        written.name = "cache_out"
        doubled = network.add_elementwise(
            reader, reader, trt.ElementWiseOperation.SUM
        ).get_output(0)
        doubled.name = "reader_out"
        total = network.add_elementwise(
            total_of(written), total_of(doubled), trt.ElementWiseOperation.SUM
        ).get_output(0)
        total.name = "result"
        for tensor in (total, written, doubled):
            network.mark_output(tensor)
        runs = self._cpp_and_python(
            builder,
            network,
            "kv_and_user_alias",
            ["cache", "reader", "update", "pos"],
            ["result", "cache_out", "reader_out"],
            {
                "cache_out": ("cache", "kv_cache_update"),
                "reader_out": ("reader", "user"),
            },
        )
        for name, run in runs:
            for shared in (False, True):
                with self.subTest(name, shared=shared):
                    c = torch.full((1, 4, 16, 8), 3.0, device="cuda")
                    r = c if shared else c.clone()
                    args = (
                        c,
                        r,
                        torch.full((1, 4, 1, 8), 7.0, device="cuda"),
                        torch.tensor([2], dtype=torch.int32, device="cuda"),
                    )
                    if shared:
                        with self.assertRaisesRegex(RuntimeError, "share memory"):
                            run(*args)
                        continue
                    out = run(*args)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    torch.testing.assert_close(out, torch.full_like(out, 4736.0))
                    torch.testing.assert_close(r, torch.full_like(r, 6.0))


if __name__ == "__main__":
    run_tests()
