# type: ignore
"""End-to-end tests for aliased I/O (KV-cache fast path + buffer lifting).

The KV-cache fast path emits ``IKVCacheUpdateLayer`` and binds the layer's
output to the cache input via aliased I/O. The C++ runtime honors the
aliasing by binding both bindings to the same device pointer; the engine
writes through the binding into the user's input storage. The Python
runtime does NOT support aliasing, so all tests here force the C++ runtime
(``use_python_runtime=False``).

These tests exercise the full pipeline:

* Converter emits the layer with aliased output.
* ``TRTInterpreter`` carries the aliased_io map (with kind tag) through.
* ``SerializedInterpreterResult`` plumbs it to ``TorchTensorRTModule``.
* C++ ``TRTEngine`` reconciles against ``getAliasedInputTensor`` at load.
* ``execute_engine`` binds the aliased output to the input ``data_ptr``,
  skipping fresh allocation.
* ``TorchTensorRTModule.forward`` filters aliased outputs from the user
  return tuple.
* For buffer-style models, ``lift_mutated_buffers`` rewrites the EP and
  ``BufferThreadingModule`` threads buffers through each call.

The Python runtime (:class:`~torch_tensorrt.dynamo.runtime._TRTEngine.TRTEngine`)
also honors aliasing by binding the aliased output to the source input's
storage; :class:`TestPythonRuntimeAliasedIO` exercises that path directly.
"""

import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine


def _compile_cpp(model, args):
    """Convenience: torch.export + torch_tensorrt.compile with C++ runtime."""
    ep = export(model, tuple(args))
    return torch_tensorrt.compile(
        ep,
        ir="dynamo",
        inputs=list(args),
        enabled_precisions={torch.float32},
        min_block_size=1,
        use_python_runtime=False,
    )


def _find_aliased_io(compiled):
    """Return the aliased_io map from the first inner TRT module, or {}."""
    for _name, mod in compiled.named_modules():
        if hasattr(mod, "aliased_io") and mod.aliased_io:
            return dict(mod.aliased_io)
    return {}


class TestUserInputKVCache(TestCase):
    """User passes the cache tensor on every call; engine mutates in place."""

    def test_single_slice_write_in_place(self):
        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        cache = torch.zeros(2, 4, 16, 8, device="cuda")
        update = torch.ones(2, 4, 1, 8, device="cuda") * 7.0
        compiled = _compile_cpp(M().cuda(), (cache.clone(), update.clone()))

        # Aliasing is recorded with kv_cache_update kind.
        aliased = _find_aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        cache_run = cache.clone()
        cache_id, cache_ptr = id(cache_run), cache_run.data_ptr()
        ret = compiled(cache_run, update)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        # Numerical match against eager.
        eager = cache.clone()
        eager[:, :, 3:4, :] = update
        self.assertTrue(torch.allclose(cache_run, eager))
        self.assertTrue(torch.allclose(ret_val, eager.sum()))

        # Identity preserved.
        self.assertEqual(id(cache_run), cache_id)
        self.assertEqual(cache_run.data_ptr(), cache_ptr)

    def test_paired_kv_caches(self):
        class M(torch.nn.Module):
            def forward(self, ck, cv, k, v):
                ck[:, :, 3:4, :] = k
                cv[:, :, 5:6, :] = v
                return ck.sum() + cv.sum()

        ck = torch.zeros(2, 4, 16, 8, device="cuda")
        cv = torch.zeros(2, 4, 16, 8, device="cuda")
        k = torch.ones(2, 4, 1, 8, device="cuda") * 3.0
        v = torch.ones(2, 4, 1, 8, device="cuda") * 5.0
        compiled = _compile_cpp(
            M().cuda(), (ck.clone(), cv.clone(), k.clone(), v.clone())
        )

        # Both K and V should be aliased.
        aliased = _find_aliased_io(compiled)
        self.assertEqual(len(aliased), 2)

        ck_run, cv_run = ck.clone(), cv.clone()
        ret = compiled(ck_run, cv_run, k, v)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        ck_eager, cv_eager = ck.clone(), cv.clone()
        ck_eager[:, :, 3:4, :] = k
        cv_eager[:, :, 5:6, :] = v
        self.assertTrue(torch.allclose(ck_run, ck_eager))
        self.assertTrue(torch.allclose(cv_run, cv_eager))
        self.assertTrue(torch.allclose(ret_val, ck_eager.sum() + cv_eager.sum()))

    def test_streaming_state_accumulates(self):
        """Repeated calls on the same cache tensor should observe the
        previous call's mutation."""

        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        proto = torch.zeros(2, 4, 16, 8, device="cuda")
        upd_proto = torch.ones(2, 4, 1, 8, device="cuda")
        compiled = _compile_cpp(M().cuda(), (proto.clone(), upd_proto.clone()))

        cache = torch.zeros(2, 4, 16, 8, device="cuda")
        # Step 1: write 1s -> 64 elements of 1 at position 3
        compiled(cache, torch.ones(2, 4, 1, 8, device="cuda") * 1.0)
        self.assertAlmostEqual(cache.sum().item(), 64.0, places=3)
        # Step 2: overwrite with 5s -> 64 * 5 = 320
        compiled(cache, torch.ones(2, 4, 1, 8, device="cuda") * 5.0)
        self.assertAlmostEqual(cache.sum().item(), 320.0, places=3)
        # Step 3: write 0s
        compiled(cache, torch.zeros(2, 4, 1, 8, device="cuda"))
        self.assertAlmostEqual(cache.sum().item(), 0.0, places=3)


class TestBufferBackedKVCache(TestCase):
    """Buffers held by the module via ``register_buffer``. The compile flow
    lifts the buffers to engine inputs and wraps the compiled module to
    thread them in automatically."""

    def test_buffer_mutation_in_place(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x = torch.ones(2, 4, 1, 8, device="cuda") * 7.0
        compiled = _compile_cpp(m, (x.clone(),))

        # The lifted buffer should be aliased.
        aliased = _find_aliased_io(compiled)
        self.assertEqual(len(aliased), 1)
        _, kind = next(iter(aliased.values()))
        self.assertEqual(kind, "kv_cache_update")

        # The compiled module owns the buffer (BufferThreadingModule).
        self.assertTrue(hasattr(compiled, "cache"))
        self.assertAlmostEqual(compiled.cache.sum().item(), 0.0)

        # Call the compiled module the same way the user wrote it: model(x).
        ret = compiled(x)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        # Buffer should be mutated; sum matches eager.
        eager_m = M().cuda()
        eager_ret = eager_m(x)
        self.assertTrue(torch.allclose(compiled.cache, eager_m.cache))
        self.assertTrue(torch.allclose(ret_val, eager_ret))

    def test_paired_buffer_caches(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache_k", torch.zeros(2, 4, 16, 8))
                self.register_buffer("cache_v", torch.zeros(2, 4, 16, 8))

            def forward(self, x_k, x_v):
                self.cache_k[:, :, 3:4, :] = x_k
                self.cache_v[:, :, 3:4, :] = x_v
                return self.cache_k.sum() + self.cache_v.sum()

        m = M().cuda()
        x_k = torch.ones(2, 4, 1, 8, device="cuda") * 3.0
        x_v = torch.ones(2, 4, 1, 8, device="cuda") * 5.0
        compiled = _compile_cpp(m, (x_k.clone(), x_v.clone()))

        aliased = _find_aliased_io(compiled)
        self.assertEqual(len(aliased), 2)

        ret = compiled(x_k, x_v)
        ret_val = ret[0] if isinstance(ret, tuple) else ret

        eager_m = M().cuda()
        eager_ret = eager_m(x_k, x_v)
        self.assertTrue(torch.allclose(compiled.cache_k, eager_m.cache_k))
        self.assertTrue(torch.allclose(compiled.cache_v, eager_m.cache_v))
        self.assertTrue(torch.allclose(ret_val, eager_ret))

    def test_buffer_streaming_persists_across_calls(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x_proto = torch.ones(2, 4, 1, 8, device="cuda")
        compiled = _compile_cpp(m, (x_proto.clone(),))

        compiled(torch.ones(2, 4, 1, 8, device="cuda") * 1.0)
        self.assertAlmostEqual(compiled.cache.sum().item(), 64.0, places=3)
        compiled(torch.ones(2, 4, 1, 8, device="cuda") * 5.0)
        self.assertAlmostEqual(compiled.cache.sum().item(), 320.0, places=3)
        compiled(torch.zeros(2, 4, 1, 8, device="cuda"))
        self.assertAlmostEqual(compiled.cache.sum().item(), 0.0, places=3)


class TestAliasedIORegressions(TestCase):
    """Models without aliased I/O should be unaffected by these changes."""

    def test_no_aliasing_path_untouched(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) * 2.0

        x = torch.randn(4, 8, device="cuda")
        y = torch.randn(4, 8, device="cuda")
        compiled = _compile_cpp(M().cuda(), (x, y))

        aliased = _find_aliased_io(compiled)
        self.assertEqual(aliased, {})

        got = compiled(x, y)
        expected = M().cuda()(x, y)
        self.assertTrue(torch.allclose(got, expected, atol=1e-4))

    def test_slice_scatter_fallback_path(self):
        """A slice_scatter that doesn't qualify for KVCacheUpdate should
        still produce correct results via the scatter fallback."""

        class M(torch.nn.Module):
            def forward(self, x, y):
                z = x.clone()
                z[:, 2:4, :] = y  # 3-D — wrong rank
                return z.sum()

        x = torch.ones(2, 8, 4, device="cuda")
        y = torch.zeros(2, 2, 4, device="cuda")
        compiled = _compile_cpp(M().cuda(), (x, y))

        # No aliasing since the cache isn't 4-D.
        self.assertEqual(_find_aliased_io(compiled), {})

        got = compiled(x, y).item()
        expected = M().cuda()(x, y).item()
        self.assertAlmostEqual(got, expected, places=3)


class TestPythonRuntimeAliasedIO(TestCase):
    """The Python ``TRTEngine`` honors aliasing (in-place write-through).

    In a build with the C++ runtime available, ``TorchTensorRTModule`` always
    backs itself with the C++ engine, so we drive the Python ``TRTEngine``
    directly from the compiled module's serialized info (the same tuple both
    runtimes consume) to exercise its aliasing path.
    """

    def _python_engine_for(self, model, args):
        compiled = _compile_cpp(model, args)
        for _name, mod in compiled.named_modules():
            if hasattr(mod, "aliased_io") and mod.aliased_io:
                return TRTEngine(mod._pack_engine_info())
        raise AssertionError("no aliased inner module found")

    def _ordered_inputs(self, engine, aliased_tensor, other_tensor):
        aliased_in = next(iter(engine.aliased_io.values()))[0]
        return [
            aliased_tensor if name == aliased_in else other_tensor
            for name in engine.in_binding_names
        ]

    def test_in_place_write_through(self):
        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        cache = torch.zeros(2, 4, 16, 8, device="cuda")
        update = torch.ones(2, 4, 1, 8, device="cuda") * 7.0
        engine = self._python_engine_for(M().cuda(), (cache.clone(), update.clone()))

        # The alias source input is flagged for the fast path.
        self.assertEqual(engine.aliased_input_binding_names, {"cache"})

        run_cache = cache.clone()
        ptr_before = run_cache.data_ptr()
        engine.execute(self._ordered_inputs(engine, run_cache, update))

        eager = cache.clone()
        eager[:, :, 3:4, :] = update
        # Written in place, same storage, matches eager.
        self.assertEqual(run_cache.data_ptr(), ptr_before)
        self.assertTrue(torch.allclose(run_cache, eager))

    def test_streaming_state_accumulates(self):
        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        proto = torch.zeros(2, 4, 16, 8, device="cuda")
        upd = torch.ones(2, 4, 1, 8, device="cuda")
        engine = self._python_engine_for(M().cuda(), (proto.clone(), upd.clone()))

        cache = torch.zeros(2, 4, 16, 8, device="cuda")
        engine.execute(self._ordered_inputs(engine, cache, torch.ones_like(upd) * 1.0))
        self.assertAlmostEqual(cache.sum().item(), 64.0, places=3)
        engine.execute(self._ordered_inputs(engine, cache, torch.ones_like(upd) * 5.0))
        self.assertAlmostEqual(cache.sum().item(), 320.0, places=3)


if __name__ == "__main__":
    run_tests()
