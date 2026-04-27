import gc
import logging
import os
import shutil
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import RUNTIME_CACHE_PATH, TIMING_CACHE_PATH
from torch_tensorrt.dynamo._settings import CompilationSettings


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class TwoLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.linear(x))


def _compile_simple(runtime_cache_path=None):
    """Helper: compile SimpleModel with Python runtime, return (compiled_module, inputs)."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "enabled_precisions": {torch.float32},
        "use_python_runtime": True,
        "min_block_size": 1,
    }
    if runtime_cache_path is not None:
        kwargs["runtime_cache_path"] = runtime_cache_path
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled, inputs


def _find_python_trt_module(compiled):
    """Walk the compiled graph module to find PythonTorchTensorRTModule instances."""
    from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
        PythonTorchTensorRTModule,
    )

    for name, mod in compiled.named_modules():
        if isinstance(mod, PythonTorchTensorRTModule):
            return mod
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheSetup(TestCase):
    """Tests that runtime config and cache are correctly created for RTX."""

    def test_runtime_config_created(self):
        compiled, _ = _compile_simple()
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(
            mod, "No PythonTorchTensorRTModule found in compiled model"
        )
        self.assertIsNotNone(mod.runtime_config, "runtime_config should be set for RTX")
        self.assertIsNotNone(mod.runtime_cache, "runtime_cache should be set for RTX")

    def test_context_created_successfully(self):
        compiled, inputs = _compile_simple()
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod.context, "execution context should be created")
        # Verify inference works
        output = compiled(*[inp.clone() for inp in inputs])
        self.assertEqual(output.shape, inputs[0].shape)

    def test_runtime_cache_path_default(self):
        compiled, _ = _compile_simple()
        mod = _find_python_trt_module(compiled)
        self.assertEqual(mod.runtime_cache_path, RUNTIME_CACHE_PATH)

    def test_runtime_cache_path_custom(self):
        cache_dir = tempfile.mkdtemp()
        try:
            custom_path = os.path.join(cache_dir, "my_cache.bin")
            compiled, _ = _compile_simple(runtime_cache_path=custom_path)
            mod = _find_python_trt_module(compiled)
            self.assertEqual(mod.runtime_cache_path, custom_path)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCachePersistence(TestCase):
    """Tests that runtime cache is correctly saved to and loaded from disk."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_cache_saved_on_del(self):
        compiled, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        # Run inference to populate the cache
        _ = compiled(*[inp.clone() for inp in inputs])
        self.assertFalse(
            os.path.isfile(self.cache_path),
            "Cache should not exist before module cleanup",
        )
        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(self.cache_path),
            "Cache file should be created after module cleanup",
        )

    def test_cache_file_nonempty(self):
        compiled, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertGreater(
            os.path.getsize(self.cache_path),
            0,
            "Cache file should have nonzero size",
        )

    def test_cache_roundtrip(self):
        """Compile, infer, save. Then compile again with same cache path and verify correctness."""
        model = SimpleModel().eval().cuda()
        inputs = [torch.randn(2, 3).cuda()]
        ref_output = model(*inputs)

        # First compilation — populates and saves cache
        compiled1, _ = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled1(*[inp.clone() for inp in inputs])
        del compiled1
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))

        # Second compilation — should load cached data
        compiled2, _ = _compile_simple(runtime_cache_path=self.cache_path)
        output = compiled2(*[inp.clone() for inp in inputs])
        max_diff = float(torch.max(torch.abs(ref_output - output)))
        self.assertAlmostEqual(
            max_diff, 0, places=3, msg="Output mismatch after cache roundtrip"
        )

    def test_save_creates_directory(self):
        nested_path = os.path.join(self.cache_dir, "a", "b", "c", "runtime_cache.bin")
        compiled, inputs = _compile_simple(runtime_cache_path=nested_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(nested_path),
            "Save should create intermediate directories",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheConcurrency(TestCase):
    """Tests that file locking works for concurrent access."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_filelock_works(self):
        """Verify that filelock can be acquired on the cache path after save."""
        compiled, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))
        # Verify we can acquire a lock on the same path (no deadlock)
        from filelock import FileLock

        lock = FileLock(self.cache_path + ".lock")
        with lock.acquire(timeout=5):
            data = open(self.cache_path, "rb").read()
        self.assertGreater(len(data), 0)

    def test_sequential_save_load(self):
        """Two modules saving and loading from the same path should not corrupt data."""
        # First module saves
        compiled1, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled1(*[inp.clone() for inp in inputs])
        del compiled1
        gc.collect()
        size1 = os.path.getsize(self.cache_path)

        # Second module saves (overwrites)
        compiled2, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled2(*[inp.clone() for inp in inputs])
        del compiled2
        gc.collect()
        size2 = os.path.getsize(self.cache_path)

        self.assertGreater(size1, 0)
        self.assertGreater(size2, 0)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Timing cache skip is only relevant for TensorRT-RTX",
)
class TestTimingCacheSkipped(TestCase):
    """Tests that timing cache is correctly skipped for RTX builds."""

    def setUp(self):
        # Clean up any pre-existing timing cache
        if os.path.isfile(TIMING_CACHE_PATH):
            os.remove(TIMING_CACHE_PATH)

    def test_no_timing_cache_file(self):
        compiled, inputs = _compile_simple()
        _ = compiled(*[inp.clone() for inp in inputs])
        self.assertFalse(
            os.path.isfile(TIMING_CACHE_PATH),
            "Timing cache should NOT be created for RTX builds",
        )

    def test_timing_cache_skip_logged(self):
        with self.assertLogs(
            "torch_tensorrt.dynamo.conversion._TRTInterpreter", level="INFO"
        ) as cm:
            compiled, inputs = _compile_simple()
            _ = compiled(*[inp.clone() for inp in inputs])
        self.assertTrue(
            any("Skipping timing cache" in msg for msg in cm.output),
            f"Expected 'Skipping timing cache' log message, got: {cm.output}",
        )


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "This test verifies standard TRT behavior (non-RTX)",
)
class TestNonRTXUnchanged(TestCase):
    """Tests that standard TRT behavior is unaffected by the runtime cache changes."""

    def test_no_runtime_config_for_standard_trt(self):
        compiled, _ = _compile_simple()
        mod = _find_python_trt_module(compiled)
        if mod is not None:
            self.assertIsNone(
                mod.runtime_config,
                "runtime_config should be None for standard TRT",
            )
            self.assertIsNone(
                mod.runtime_cache,
                "runtime_cache should be None for standard TRT",
            )

    def test_timing_cache_still_created(self):
        # Clean up any pre-existing timing cache
        if os.path.isfile(TIMING_CACHE_PATH):
            os.remove(TIMING_CACHE_PATH)
        compiled, inputs = _compile_simple()
        _ = compiled(*[inp.clone() for inp in inputs])
        self.assertTrue(
            os.path.isfile(TIMING_CACHE_PATH),
            "Timing cache should still be created for standard TRT",
        )


if __name__ == "__main__":
    run_tests()
