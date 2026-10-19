# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""End-to-end tests for ``tensorrt::kv_cache_update``, the KV-cache write with a
position per batch row.

* A cache passed to the compiled module: the converter emits an
  ``IKVCacheUpdateLayer`` whose ``writeIndices`` are the per-row positions, and
  the engine writes the caller's cache in place.
* A cache computed inside the graph: no network input to alias, so the converter
  falls back to a scatter.
* A cache held as a module buffer: the write is classified engine-aliased, so the
  buffer is updated in place on every call, with no copy back.
"""

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


def _aliased_io(compiled):
    for _name, mod in compiled.named_modules():
        if hasattr(mod, "aliased_io") and mod.aliased_io:
            return dict(mod.aliased_io)
    return {}


def _left_in_pytorch(compiled):
    """Calls of the op that run outside a TensorRT engine."""
    op = torch.ops.tensorrt.kv_cache_update.default
    return [
        node
        for _name, mod in compiled.named_modules()
        if isinstance(mod, torch.fx.GraphModule)
        for node in mod.graph.nodes
        if node.op == "call_function" and node.target is op
    ]


def _reference(cache, update, write_indices):
    out = cache.clone()
    for b, p in enumerate(write_indices.tolist()):
        out[b, :, p : p + update.shape[2]] = update[b]
    return out


class Write(torch.nn.Module):
    def forward(self, cache, update, write_indices):
        return torch.ops.tensorrt.kv_cache_update(cache, update, write_indices)


class TestEager(TestCase):
    def test_rows_written_at_their_own_positions(self):
        cache = torch.randn(3, 2, 16, 4)
        update = torch.randn(3, 2, 4, 4)
        write_indices = torch.tensor([0, 5, 12], dtype=torch.int32)
        out = torch.ops.tensorrt.kv_cache_update(cache.clone(), update, write_indices)
        self.assertEqual(out, _reference(cache, update, write_indices))

    def test_functional(self):
        cache = torch.zeros(2, 2, 8, 4)
        torch.ops.tensorrt.kv_cache_update(
            cache, torch.ones(2, 2, 1, 4), torch.tensor([1, 2])
        )
        self.assertEqual(cache, torch.zeros(2, 2, 8, 4))

    def test_one_position_per_row_required(self):
        with self.assertRaisesRegex(RuntimeError, "write_indices"):
            torch.ops.tensorrt.kv_cache_update(
                torch.zeros(3, 2, 8, 4), torch.zeros(3, 2, 1, 4), torch.tensor([1])
            )

    def test_update_must_match_the_cache_layout(self):
        with self.assertRaisesRegex(RuntimeError, "update must be"):
            torch.ops.tensorrt.kv_cache_update(
                torch.zeros(2, 2, 8, 4), torch.zeros(2, 3, 1, 4), torch.tensor([1, 2])
            )


class TestCallerOwnedCache(TestCase):
    """The cache is an engine input: written in place through aliased I/O."""

    def _check_in_place(self, s, index_dtype):
        cache = torch.randn(3, 4, 16, 8, device="cuda")
        update = torch.randn(3, 4, s, 8, device="cuda")
        write_indices = torch.tensor([0, 7, 16 - s], dtype=index_dtype, device="cuda")
        compiled = _compile(Write().cuda(), (cache.clone(), update, write_indices))

        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        expected = _reference(cache, update, write_indices)
        cache_run = cache.clone()
        ptr = cache_run.data_ptr()
        out = compiled(cache_run, update, write_indices)
        out = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out, expected)
        self.assertEqual(cache_run, expected)
        self.assertEqual(cache_run.data_ptr(), ptr)

    def test_one_position_per_row(self):
        self._check_in_place(1, torch.int32)

    def test_several_positions_per_row(self):
        self._check_in_place(4, torch.int32)

    def test_int64_write_indices(self):
        self._check_in_place(1, torch.int64)

    def test_updated_cache_feeds_the_rest_of_the_graph(self):
        """Batched decode reads the cache it just wrote (attention)."""

        class M(torch.nn.Module):
            def forward(self, cache, update, write_indices):
                cache = torch.ops.tensorrt.kv_cache_update(cache, update, write_indices)
                return cache.sum(dim=(1, 3))

        cache = torch.randn(2, 4, 16, 8, device="cuda")
        update = torch.randn(2, 4, 1, 8, device="cuda")
        write_indices = torch.tensor([3, 9], dtype=torch.int32, device="cuda")
        compiled = _compile(M().cuda(), (cache.clone(), update, write_indices))
        self.assertEqual(len(_aliased_io(compiled)), 1)

        expected = _reference(cache, update, write_indices)
        cache_run = cache.clone()
        out = compiled(cache_run, update, write_indices)
        out = out[0] if isinstance(out, tuple) else out
        torch.testing.assert_close(out, expected.sum(dim=(1, 3)))
        self.assertEqual(cache_run, expected)

    def test_rows_advance_independently_across_calls(self):
        compiled = _compile(
            Write().cuda(),
            (
                torch.zeros(2, 4, 16, 8, device="cuda"),
                torch.zeros(2, 4, 1, 8, device="cuda"),
                torch.zeros(2, dtype=torch.int32, device="cuda"),
            ),
        )
        cache = torch.zeros(2, 4, 16, 8, device="cuda")
        expected = cache.clone()
        positions = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
        for step in range(4):
            update = torch.full((2, 4, 1, 8), step + 1.0, device="cuda")
            compiled(cache, update, positions)
            expected = _reference(expected, update, positions)
            positions = positions + 1
        self.assertEqual(cache, expected)


class TestFallback(TestCase):
    def test_cache_computed_in_graph_is_scattered(self):
        class M(torch.nn.Module):
            def forward(self, base, update, write_indices):
                return torch.ops.tensorrt.kv_cache_update(
                    base * 2.0, update, write_indices
                )

        base = torch.randn(3, 4, 16, 8, device="cuda")
        update = torch.randn(3, 4, 2, 8, device="cuda")
        write_indices = torch.tensor([1, 14, 6], dtype=torch.int32, device="cuda")
        compiled = _compile(M().cuda(), (base.clone(), update, write_indices))
        # Converted (to a scatter), not left to PyTorch, and nothing aliased.
        self.assertEqual(_left_in_pytorch(compiled), [])
        self.assertEqual(_aliased_io(compiled), {})

        base_run = base.clone()
        out = compiled(base_run, update, write_indices)
        out = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out, _reference(base * 2.0, update, write_indices))
        self.assertEqual(base_run, base)


class TestBufferBackedCache(TestCase):
    """A module-owned cache: lifted to an engine input and written in place."""

    def test_buffer_updated_in_place_across_calls(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, update, write_indices):
                self.cache.copy_(
                    torch.ops.tensorrt.kv_cache_update(
                        self.cache, update, write_indices
                    )
                )
                return self.cache.sum(dim=(1, 3))

        update = torch.ones(2, 4, 1, 8, device="cuda")
        write_indices = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        compiled = _compile(M().cuda(), (update, write_indices))
        aliased = _aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        eager = M().cuda()
        for step in range(3):
            update = torch.randn(2, 4, 1, 8, device="cuda")
            write_indices = torch.tensor(
                [step, 3 + 2 * step], dtype=torch.int32, device="cuda"
            )
            out = compiled(update, write_indices)
            out = out[0] if isinstance(out, tuple) else out
            torch.testing.assert_close(out, eager(update, write_indices))
            self.assertEqual(compiled.cache, eager.cache)


if __name__ == "__main__":
    run_tests()
