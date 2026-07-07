# type: ignore
"""Serialization tests for aliased I/O.

Verifies that compiled modules with aliased I/O survive a round-trip
through ``torch_tensorrt.save`` / ``torch_tensorrt.load``:

* **User-input style** — the engine's ``aliased_io`` map is part of the
  C++ engine's serialized form (``ALIASED_IO_IDX`` in the wire format).
  After load, ``execute_engine`` reconstructs aliasing from those bytes
  and the runtime binds outputs to input ``data_ptr`` as before.

* **Buffer-backed style** — additionally requires that the lifted
  buffers (registered as ``nn.Module`` state on the compiled GraphModule
  and read via ``get_attr`` nodes in the fx graph) survive
  ``torch.export``. The ``inline_lifted_buffers_into_gm`` post-compile
  transform replaces what used to be an external ``BufferThreadingModule``
  wrapper — making the result a plain ``fx.GraphModule`` that exports
  naturally without a custom wrapper class.
"""

import tempfile

import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests


def _compile_and_roundtrip(model, args):
    """Compile, save, load, return (compiled, loaded_gm)."""
    ep = export(model, tuple(args))
    compiled = torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=list(args),
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".ep", delete=False) as f:
        path = f.name
    torch_tensorrt.save(compiled, path, arg_inputs=list(args))
    loaded_ep = torch_tensorrt.load(path)
    loaded = loaded_ep.module() if hasattr(loaded_ep, "module") else loaded_ep
    return compiled, loaded


class TestUserInputAliasingSurvivesSaveLoad(TestCase):
    """User passes the cache each call. The engine's aliased_io map is
    serialized in the engine bytes; after load, runtime aliasing still
    works."""

    def test_kv_cache_user_input_save_load(self):
        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        cache_sample = torch.zeros(1, 4, 16, 8, device="cuda")
        update_sample = torch.ones(1, 4, 1, 8, device="cuda") * 7.0
        compiled, loaded = _compile_and_roundtrip(
            M().cuda(), (cache_sample.clone(), update_sample.clone())
        )

        # Run loaded module; cache should be mutated in place via aliasing.
        cache_run = torch.zeros(1, 4, 16, 8, device="cuda")
        cache_id, cache_ptr = id(cache_run), cache_run.data_ptr()
        ret = loaded(cache_run, update_sample)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        eager = torch.zeros(1, 4, 16, 8, device="cuda")
        eager[:, :, 3:4, :] = update_sample
        self.assertTrue(torch.allclose(cache_run, eager))
        self.assertTrue(torch.allclose(ret_val, eager.sum()))
        # Aliased pointer identity is preserved post-load too.
        self.assertEqual(id(cache_run), cache_id)
        self.assertEqual(cache_run.data_ptr(), cache_ptr)


class TestBufferAliasingSurvivesSaveLoad(TestCase):
    """Module-held buffer (BUFFER + BUFFER_MUTATION). The post-compile
    transform registers the buffer on the compiled GraphModule and
    rewrites the lifted-buffer placeholder to a ``get_attr`` read. That
    structure exports cleanly; after load the buffer is still a
    ``nn.Module`` buffer on the loaded gm and the engine still aliases
    it in place."""

    def test_buffer_kv_save_load(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x = torch.ones(1, 4, 1, 8, device="cuda") * 3.0
        compiled, loaded = _compile_and_roundtrip(m, (x.clone(),))

        # The compiled module already had the buffer; the loaded one
        # should still have it (registered as nn.Module state, saved
        # natively through torch.export).
        self.assertTrue(hasattr(compiled, "cache"))
        self.assertTrue(hasattr(loaded, "cache"))
        self.assertEqual(tuple(loaded.cache.shape), tuple(compiled.cache.shape))

        # Reset to zero so the comparison is clean.
        loaded.cache.zero_()
        ret = loaded(x)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        eager = M().cuda()
        eager_ret = eager(x.clone())
        self.assertTrue(torch.allclose(ret_val, eager_ret))
        self.assertTrue(torch.allclose(loaded.cache, eager.cache))

    def test_buffer_kv_save_load_streaming(self):
        """Repeated calls on the LOADED module accumulate state on the
        loaded module's buffer."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x = torch.ones(1, 4, 1, 8, device="cuda")
        _, loaded = _compile_and_roundtrip(m, (x.clone(),))

        loaded.cache.zero_()
        # Each step overwrites position 3 (32 elements).
        loaded(torch.ones(1, 4, 1, 8, device="cuda") * 1.0)
        self.assertAlmostEqual(loaded.cache.sum().item(), 32.0, places=3)
        loaded(torch.ones(1, 4, 1, 8, device="cuda") * 5.0)
        self.assertAlmostEqual(loaded.cache.sum().item(), 160.0, places=3)
        loaded(torch.zeros(1, 4, 1, 8, device="cuda"))
        self.assertAlmostEqual(loaded.cache.sum().item(), 0.0, places=3)


if __name__ == "__main__":
    run_tests()
