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

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


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


if __name__ == "__main__":
    run_tests()
