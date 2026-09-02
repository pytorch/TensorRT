# type: ignore
"""End-to-end tests for ``aten.index_copy`` KV-cache aliasing.

Two converters are registered for ``aten.index_copy.default``:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires for
  the narrow KV-eligible case (4-D static cache, dim=2, batch=1, and a
  contiguous run of write positions whose length may be static or only known
  at runtime) and emits ``IKVCacheUpdateLayer`` with aliased I/O.

* ``aten_ops_index_copy_fallback`` — STANDARD priority. Fires for
  everything else; produces correct results via the scatter path. No
  graph break.

These tests verify both paths end-to-end via the C++ runtime: the
fast path mutates in place, the fallback produces correct numerical
results without aliasing.
"""

from unittest import mock

import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests


def _compile(model, args):
    ep = export(model, tuple(args))
    return torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=list(args),
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
    )


def _compile_dynamic_write(model, cache, start, update, min_len=2, max_len=8):
    """Compile ``model(cache, start, update)`` with the write length dynamic.

    ``Dim(min=1)`` is recorded as min=2 (PyTorch 0/1 specialization), so the
    profile floor is 2 rather than 1.
    """
    batch, heads, _, head_dim = tuple(update.shape)
    seq = torch.export.Dim("seq", min=min_len, max=max_len)
    ep = export(
        model,
        (cache, start, update),
        dynamic_shapes={"cache": None, "start": None, "update": {2: seq}},
    )
    return torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(shape=tuple(cache.shape), dtype=cache.dtype),
            torch_tensorrt.Input(shape=tuple(start.shape), dtype=start.dtype),
            torch_tensorrt.Input(
                min_shape=(batch, heads, min_len, head_dim),
                opt_shape=(batch, heads, (min_len + max_len) // 2, head_dim),
                max_shape=(batch, heads, max_len, head_dim),
                dtype=update.dtype,
            ),
        ],
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
    )


def _aliased_io(compiled):
    for _name, mod in compiled.named_modules():
        if hasattr(mod, "aliased_io") and mod.aliased_io:
            return dict(mod.aliased_io)
    return {}


class TestIndexCopyKVFastPath(TestCase):
    """KV-eligible: 4-D static cache, dim=2, batch=1, single-position
    write. The validator passes and the fast path emits
    ``IKVCacheUpdateLayer`` with aliased output."""

    def test_single_position_write_aliased(self):
        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor([3], dtype=torch.int64, device="cuda")
        update = torch.ones(1, 4, 1, 8, device="cuda") * 7.0

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))

        # Fast path fired — aliasing recorded.
        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        # Numerical match against eager.
        cache_run = cache.clone()
        out = compiled(cache_run, index, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager = cache.clone()
        eager_out = torch.ops.aten.index_copy.default(eager, 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager_out))

    def test_dynamic_multi_position_write_aliased(self):
        """A prompt-sized write, of a length not known until runtime.

        This is prefill: one engine writes as many cache positions as the call
        brings, so the update's seq extent is symbolic at conversion time and
        the layer takes the run length from the tensor it is handed. Run at two
        lengths to show the engine is not specialised to either.
        """

        class M(torch.nn.Module):
            def forward(self, cache, start, update):
                index = torch.arange(update.shape[2], device=cache.device) + start
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        start = torch.tensor(3, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 4, 8, device="cuda")

        compiled = _compile_dynamic_write(M().cuda(), cache, start, update)

        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        # 2 is the profile floor, where an off-by-one in writeIndices shows.
        for run_len in (2, 4, 8):
            update = torch.randn(1, 4, run_len, 8, device="cuda")
            index = torch.arange(run_len, device="cuda") + start
            out = compiled(cache.clone(), start, update)
            out_val = out[0] if isinstance(out, tuple) else out
            eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
            self.assertTrue(
                torch.allclose(out_val, eager), f"mismatch writing {run_len} positions"
            )

    def test_static_multi_position_write_on_lifted_buffer_aliased(self):
        """A prefill-shaped write whose length is known at compile time.

        Regression: ``constant_fold``, inside ``post_lowering``, replaces a
        static-length ``arange`` with a frozen parameter. A validator that only
        pattern-matches the ``arange`` node therefore answers True for
        ``_buffer_lifting`` -- which runs *before* lowering and drops the
        write's ``copy_`` on that promise -- and False for the partitioner
        afterwards, so nothing aliased the buffer and
        ``assert_predicted_kv_aliased`` raised.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, start, update):
                index = torch.arange(4, device=update.device) + start
                self.cache.index_copy_(2, index, update)
                return self.cache.sum()

        start = torch.tensor(3, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 4, 8, device="cuda")

        compiled = _compile(M().cuda(), (start, update))

        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        compiled(start, update)

        # The engine wrote through the aliased binding, so the module's own
        # buffer holds the result -- at positions 3..6, and nowhere else.
        cache = dict(compiled.named_buffers())["cache"]
        self.assertTrue(torch.allclose(cache[:, :, 3:7, :], update))
        self.assertEqual(float(cache[:, :, :3, :].abs().sum()), 0.0)
        self.assertEqual(float(cache[:, :, 7:, :].abs().sum()), 0.0)

    def test_shifted_arange_operand_order_aliased(self):
        """``start + arange(n)`` -- the range on the right of the add.

        Addition commutes, so either operand can hold the range; choosing it by
        position would drop the fast path for half the graphs that qualify.
        """

        class M(torch.nn.Module):
            def forward(self, cache, start, update):
                index = start + torch.arange(update.shape[2], device=cache.device)
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        start = torch.tensor(3, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 4, 8, device="cuda")

        compiled = _compile_dynamic_write(M().cuda(), cache, start, update)

        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        index = torch.arange(4, device="cuda") + start
        out = compiled(cache.clone(), start, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager))


class TestIndexCopyFallback(TestCase):
    """Cases where the validator denies the KV fast path. The fallback
    converter must produce correct results without aliasing."""

    def test_scattered_index_uses_fallback(self):
        """Positions that do not ascend by one cannot be a KV cache write.

        ``IKVCacheUpdateLayer`` writes a contiguous block from a single start,
        so taking this path would write positions 5, 6, 7 where index_copy
        means 5, 1, 9 -- silently, and with the cache aliased. Everything else
        here is KV-eligible, so the index is the only thing keeping the fast
        path off it.
        """

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor([5, 1, 9], dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 3, 8, device="cuda")

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(cache.clone(), index, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager))

    def test_strided_index_uses_fallback(self):
        """A stride-2 range, of a length only known at runtime.

        The near-miss that matters: the index really is an ``arange`` and
        everything else is KV-eligible, so the step is the only thing keeping
        the fast path off it. Taken, the layer would write 1,2,3,4 where
        index_copy means 1,3,5,7 -- silently, with the cache aliased.
        """

        class M(torch.nn.Module):
            def forward(self, cache, start, update):
                index = (
                    torch.arange(0, 2 * update.shape[2], 2, device=cache.device) + start
                )
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        start = torch.tensor(1, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 4, 8, device="cuda")

        compiled = _compile_dynamic_write(
            M().cuda(), cache, start, update, min_len=2, max_len=4
        )
        self.assertEqual(_aliased_io(compiled), {})

        for run_len in (2, 4):
            u = torch.randn(1, 4, run_len, 8, device="cuda")
            index = torch.arange(0, 2 * run_len, 2, device="cuda") + start
            out = compiled(cache.clone(), start, u)
            out_val = out[0] if isinstance(out, tuple) else out
            eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, u)
            self.assertTrue(torch.allclose(out_val, eager), f"len {run_len}")

    def test_zero_dim_index_uses_fallback(self):
        """``aten.index_copy`` accepts a 0-d index; ``writeIndices`` wants rank 1.

        Left on the fast path this surfaces as a bare ``nbDims == 1`` parameter
        check from ``add_kv_cache_update``, naming nothing the user wrote.
        """

        class M(torch.nn.Module):
            def forward(self, cache, index, update):
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor(3, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 1, 8, device="cuda")

        compiled = _compile(M().cuda(), (cache.clone(), index, update.clone()))
        self.assertEqual(_aliased_io(compiled), {})

        out = compiled(cache.clone(), index, update)
        out_val = out[0] if isinstance(out, tuple) else out
        eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, update)
        self.assertTrue(torch.allclose(out_val, eager))

    def test_dynamic_write_falls_back_when_layer_declines(self):
        """The layer can turn a write down at conversion time, after the
        validator has already claimed the node.

        ``emit_kv_cache_update_layer`` returns None when the *network's* shapes
        disagree with the cache -- an extent static in FX metadata can still
        arrive as -1. The node belongs to this converter by then, so the scatter
        it falls back to has to handle a dynamic write length; raising there
        would abort a compile that previously succeeded via a Torch fallback.
        """

        class M(torch.nn.Module):
            def forward(self, cache, start, update):
                index = torch.arange(update.shape[2], device=cache.device) + start
                return torch.ops.aten.index_copy.default(cache, 2, index, update)

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        start = torch.tensor(3, dtype=torch.int64, device="cuda")
        update = torch.randn(1, 4, 4, 8, device="cuda")

        with mock.patch(
            "torch_tensorrt.dynamo.conversion.impl.index_copy."
            "emit_kv_cache_update_layer",
            return_value=None,
        ):
            compiled = _compile_dynamic_write(M().cuda(), cache, start, update)

        self.assertEqual(_aliased_io(compiled), {})

        for run_len in (2, 4, 8):
            u = torch.randn(1, 4, run_len, 8, device="cuda")
            index = torch.arange(run_len, device="cuda") + start
            out = compiled(cache.clone(), start, u)
            out_val = out[0] if isinstance(out, tuple) else out
            eager = torch.ops.aten.index_copy.default(cache.clone(), 2, index, u)
            self.assertTrue(torch.allclose(out_val, eager), f"len {run_len}")

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


if __name__ == "__main__":
    run_tests()
