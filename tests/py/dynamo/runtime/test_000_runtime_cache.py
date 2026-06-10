import gc
import io
import os
import shutil
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from torch_tensorrt.runtime import RuntimeSettings, runtime_cache


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def _fresh_conv_model_and_inputs(seed=0):
    torch.manual_seed(seed)
    return ConvModel().eval().cuda(), [torch.randn(2, 3, 16, 16).cuda()]


def _apply_runtime_settings(compiled, rs):
    """Apply ``RuntimeSettings`` to every ``TorchTensorRTModule`` under ``compiled``.

    Mirrors what user code would do: ``mod.runtime_settings = rs`` after
    compile. The compile-time hint (``torchtrt.compile(runtime_settings=...)``)
    was dropped now that lazy ``IExecutionContext`` creation absorbs the
    one-create benefit it used to provide.
    """
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule):
            mod.runtime_settings = rs


def _compile(model, inputs, *, runtime_cache_path=None):
    """Compile ``model`` through whichever runtime the build selects.

    When ``runtime_cache_path`` is supplied, the cache path is applied via
    ``mod.runtime_settings = RuntimeSettings(runtime_cache=path)`` after
    compile -- matches the post-refactor user flow.
    """
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        min_block_size=1,
    )
    torch._dynamo.reset()
    if runtime_cache_path is not None:
        _apply_runtime_settings(
            compiled, RuntimeSettings(runtime_cache=runtime_cache_path)
        )
    return compiled


def _compile_simple(runtime_cache_path=None):
    """Compile SimpleModel through the build-selected runtime."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    return (
        _compile(model, inputs, runtime_cache_path=runtime_cache_path),
        inputs,
    )


def _find_python_trt_engine(compiled):
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod.engine
    return None


def _find_python_trt_module(compiled):
    """The parent ``TorchTensorRTModule`` is the canonical owner of the
    implicit cache handle (was on ``TRTRuntimeConfig`` before the unification)."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "Whitebox introspection requires the Python TRTEngine path",
)
class TestRuntimeCacheSetup(TestCase):
    """Tests that runtime config and per-engine cache are correctly created for RTX."""

    def test_runtime_config_created(self):
        compiled, _ = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.runtime_config)

    def test_context_is_lazy_until_forward(self):
        """Engine context is materialized lazily on first forward, not at setup."""
        compiled, inputs = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertFalse(
            engine.has_context(),
            "Expected lazy context: engine.has_context() must be False until first forward",
        )
        _ = compiled(*inputs)
        self.assertTrue(engine.has_context())

    def test_default_uses_temp_path_implicit_handle(self):
        """Default RuntimeSettings points runtime_cache at the per-user temp file
        (see _defaults.RUNTIME_CACHE_PATH); the module owns the implicit handle."""
        from torch_tensorrt.dynamo._defaults import RUNTIME_CACHE_PATH

        compiled, _ = _compile_simple()
        module = _find_python_trt_module(compiled)
        self.assertIsNotNone(module._implicit_cache_handle)
        self.assertEqual(module._implicit_cache_handle.path, RUNTIME_CACHE_PATH)

    def test_implicit_cache_handle_for_path_hint(self):
        """Passing a path string in RuntimeSettings.runtime_cache creates an implicit handle on the module."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            compiled, _ = _compile_simple(runtime_cache_path=path)
            module = _find_python_trt_module(compiled)
            self.assertIsNotNone(module._implicit_cache_handle)
            self.assertEqual(module._implicit_cache_handle.path, path)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache persistence is RTX-only",
)
class TestRuntimeCachePersistence(TestCase):
    """End-to-end: compile with a cache path, infer, destroy, reload, infer again."""

    def test_cache_saved_on_del(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            model, inputs = _fresh_conv_model_and_inputs(seed=42)
            compiled = _compile(model, inputs, runtime_cache_path=path)
            _ = compiled(*inputs)
            del compiled
            gc.collect()
            self.assertTrue(
                os.path.exists(path),
                f"Implicit cache handle should have saved to {path} on engine __del__",
            )

    def test_set_runtime_settings_saves_prior_cache_on_swap(self):
        """Re-pointing ``runtime_cache`` via the property setter saves the
        prior implicit cache before swapping. The work lives on
        :py:meth:`TorchTensorRTModule._materialize_implicit_handle`.
        """
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "cache_A.bin")
            path_b = os.path.join(tmp, "cache_B.bin")
            model, inputs = _fresh_conv_model_and_inputs(seed=42)
            compiled = _compile(model, inputs, runtime_cache_path=path_a)
            _ = compiled(*inputs)
            # Sanity: no file written yet (nothing has saved).
            self.assertFalse(os.path.exists(path_a))

            # Walk to the inner TorchTensorRTModule(s) and swap the cache path
            # directly -- the outer GraphModule doesn't carry the
            # ``runtime_settings`` property, and we want a *permanent* swap
            # (the runtime_config CM restores on exit, which would mask the
            # save-on-swap signal we're after). The walk is wrapped in a
            # helper so the loop variable doesn't outlive the call and keep
            # the inner module alive past ``del compiled``.
            def _swap_all(target: torch.nn.Module, new_rs: RuntimeSettings) -> int:
                count = 0
                for _, mod in target.named_modules():
                    if isinstance(mod, TorchTensorRTModule):
                        mod.runtime_settings = new_rs
                        count += 1
                return count

            swapped = _swap_all(compiled, RuntimeSettings(runtime_cache=path_b))
            self.assertGreater(swapped, 0, "Expected at least one TorchTensorRTModule")
            self.assertTrue(
                os.path.exists(path_a),
                f"Prior implicit cache should have been saved to {path_a} "
                "synchronously on set_runtime_settings swap",
            )
            _ = compiled(*inputs)
            del compiled
            gc.collect()
            self.assertTrue(
                os.path.exists(path_b),
                f"New implicit cache should have saved to {path_b} on engine "
                "__del__ after the swap",
            )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "runtime_cache CM is RTX-only",
)
class TestRuntimeCacheContextManager(TestCase):
    """Tests for the runtime_cache(target, path) shared-cache CM."""

    def test_with_cache_loads_and_saves(self):
        compiled, inputs = _compile_simple()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache(compiled, path) as rc:
                self.assertIsNotNone(rc)
                self.assertEqual(rc.path, path)
                _ = compiled(*inputs)
            # autosave on exit
            self.assertTrue(os.path.exists(path))

    def test_with_cache_in_memory_only(self):
        """path='' means in-memory only; no disk artifact after exit."""
        compiled, inputs = _compile_simple()
        with tempfile.TemporaryDirectory() as tmp:
            with runtime_cache(compiled, "") as rc:
                self.assertEqual(rc.path, "")
                _ = compiled(*inputs)
            self.assertFalse(os.listdir(tmp), "No files should be created for path=''")

    @unittest.skipIf(
        ENABLED_FEATURES.torch_tensorrt_runtime,
        "Identity assertion requires the Python TRTEngine path (whitebox)",
    )
    def test_shared_cache_pointer_across_modules(self):
        """Two modules sharing one runtime_cache handle reference the same IRuntimeCache."""
        compiled_a, inputs_a = _compile_simple()
        compiled_b, inputs_b = _compile_simple()
        eng_a = _find_python_trt_engine(compiled_a)
        eng_b = _find_python_trt_engine(compiled_b)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache([compiled_a, compiled_b], path) as rc:
                self.assertIs(eng_a.runtime_settings.runtime_cache, rc)
                self.assertIs(eng_b.runtime_settings.runtime_cache, rc)
                _ = compiled_a(*inputs_a)
                _ = compiled_b(*inputs_b)
            self.assertTrue(os.path.exists(path))

    def test_runtime_cache_on_empty_target_raises(self):
        """A target with no TRT submodules raises a clear error on enter."""
        empty = torch.nn.Linear(3, 3).cuda()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            with self.assertRaises(RuntimeError):
                with runtime_cache(empty, path):
                    pass

    def test_cm_does_not_double_save_on_rc_gc(self):
        """CM yields handle with autosave_on_del=False; only one save happens.

        Regression: if the CM-yielded handle had autosave_on_del=True, the
        handle's __del__ would re-save after the CM's __exit__ already wrote
        the file. We disable autosave_on_del on CM-created handles to avoid
        that double-write.
        """
        compiled, inputs = _compile_simple()
        save_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache(compiled, path) as rc:
                # CM-created handle must not autosave on del (CM saves explicitly).
                self.assertFalse(rc.autosave_on_del)
                original_save = rc.save

                def _tracking_save(p=None):
                    save_calls.append(p)
                    return original_save(p)

                rc.save = _tracking_save  # type: ignore[method-assign]
                _ = compiled(*inputs)
            # CM.__exit__ saves exactly once; rc going out of scope triggers
            # __del__ but autosave_on_del is False, so no second save.
            del rc
            gc.collect()
            self.assertEqual(len(save_calls), 1, f"Expected one save, got {save_calls}")
            self.assertTrue(os.path.exists(path))


class TestRuntimeCacheHandleAutosave(TestCase):
    """Whitebox tests for RuntimeCacheHandle.autosave_on_del semantics."""

    def test_user_built_handle_no_autosave_by_default(self):
        """Hand-built handle defaults to autosave_on_del=False; nothing on GC."""
        from torch_tensorrt.runtime._runtime_cache import RuntimeCacheHandle

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            handle = RuntimeCacheHandle(path=path)
            self.assertFalse(handle.autosave_on_del)
            del handle
            gc.collect()
            self.assertFalse(
                os.path.exists(path),
                "User-built handle with autosave_on_del=False should not save on GC",
            )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "runtime_cache stream-mode is RTX-only",
)
class TestRuntimeCacheStreamSupport(TestCase):
    """Stream-backed flavor of the runtime_cache CM.

    Mirrors the path-mode tests but routes through file-like objects:
    opened file handles (``"wb+"``) and ``io.BytesIO`` buffers. The handle's
    ``load_from_stream`` / ``save_to_stream`` primitives are exercised
    directly in the round-trip test.
    """

    @unittest.skipIf(
        ENABLED_FEATURES.torch_tensorrt_runtime,
        "Round-trip asserts load before first inference; cpp rt materializes "
        "the IRuntimeCache lazily on context creation",
    )
    def test_stream_bytesio_round_trip(self):
        """Write cache bytes through BytesIO once, read them back into a fresh module."""
        compiled_a, inputs_a = _compile_simple()
        buf = io.BytesIO()
        with runtime_cache(compiled_a, buf) as rc_a:
            self.assertEqual(rc_a.path, "", "stream-mode keeps path empty")
            _ = compiled_a(*inputs_a)
        written = buf.getvalue()
        self.assertGreater(
            len(written),
            0,
            "CM exit should have written cache bytes into BytesIO",
        )

        # Round-trip: fresh module + fresh BytesIO seeded with the bytes.
        compiled_b, inputs_b = _compile_simple()
        replay = io.BytesIO(written)
        with runtime_cache(compiled_b, replay) as rc_b:
            # Load fired on enter; the cache should now serialize back to bytes.
            sanity = io.BytesIO()
            wrote_back = rc_b.save_to_stream(sanity)
            self.assertGreater(
                wrote_back,
                0,
                "load_from_stream should have populated the cache",
            )
            _ = compiled_b(*inputs_b)

    def test_stream_open_file_handle(self):
        """Pass an opened binary file handle directly to the CM."""
        compiled, inputs = _compile_simple()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            with open(path, "wb+") as f:
                with runtime_cache(compiled, f) as rc:
                    self.assertEqual(rc.path, "")
                    _ = compiled(*inputs)
                # On CM exit the handle wrote into f; closing the with-open
                # flushes it. Verify after closure to avoid asserting on a
                # half-flushed stream.
            self.assertGreater(
                os.path.getsize(path),
                0,
                "CM exit should have written cache bytes into the file handle",
            )

    @unittest.skipIf(
        ENABLED_FEATURES.torch_tensorrt_runtime,
        "Calls load_from_stream before first inference; cpp rt materializes "
        "the IRuntimeCache lazily on context creation",
    )
    def test_handle_stream_methods_direct(self):
        """Exercise RuntimeCacheHandle.{load,save}_from_stream on a CM-yielded handle."""
        compiled_a, inputs_a = _compile_simple()
        with runtime_cache(compiled_a, "") as rc_a:
            _ = compiled_a(*inputs_a)
            buf = io.BytesIO()
            n = rc_a.save_to_stream(buf)
        if n == 0:
            self.skipTest(
                "cache had nothing to serialize after the warmup -- nothing to "
                "round-trip"
            )

        compiled_b, _ = _compile_simple()
        with runtime_cache(compiled_b, "") as rc_b:
            buf.seek(0)
            m = rc_b.load_from_stream(buf)
            self.assertEqual(m, n, "round-trip should consume the same byte count")

    def test_rejects_unsupported_io_type(self):
        """An int / random object is neither a path nor a stream -> TypeError."""
        compiled, _ = _compile_simple()
        with self.assertRaises(TypeError):
            runtime_cache(compiled, 42)  # type: ignore[arg-type]


if __name__ == "__main__":
    run_tests()
