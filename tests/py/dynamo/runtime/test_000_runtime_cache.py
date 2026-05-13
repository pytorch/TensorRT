import gc
import os
import shutil
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import RUNTIME_CACHE_PATH, TIMING_CACHE_PATH
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


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
    """Deterministic ConvModel + input pair for end-to-end cache tests on either runtime."""
    torch.manual_seed(seed)
    return ConvModel().eval().cuda(), [torch.randn(2, 3, 16, 16).cuda()]


def _compile(model, inputs, *, use_python_runtime, runtime_cache_path=None):
    """Compile `model` through either runtime. Returns the compiled module."""
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "use_python_runtime": use_python_runtime,
        "min_block_size": 1,
    }
    if runtime_cache_path is not None:
        kwargs["runtime_cache_path"] = runtime_cache_path
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled


def _compile_simple(runtime_cache_path=None):
    """Compile the SimpleModel on the Python runtime (used by Python-only setup tests)."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    return (
        _compile(
            model,
            inputs,
            use_python_runtime=True,
            runtime_cache_path=runtime_cache_path,
        ),
        inputs,
    )


def _find_python_trt_module(compiled):
    """Walk the compiled graph module to find PythonTorchTensorRTModule instances."""
    from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
        PythonTorchTensorRTModule,
    )

    for _name, mod in compiled.named_modules():
        if isinstance(mod, PythonTorchTensorRTModule):
            return mod
    return None


# Parameterize end-to-end cache persistence tests over both runtime paths. The C++
# variant is skipped inside the test body when the C++ runtime is not available.
_RUNTIMES = [("python", True), ("cpp", False)]


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheSetup(TestCase):
    """Python-runtime-only setup checks: the compiled module exposes a live runtime cache."""

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
    """Load-on-setup / save-on-destructor contract, exercised on both runtimes."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _skip_if_cpp_unavailable(self, use_python_runtime):
        if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("C++ runtime is not available")

    @parameterized.expand(_RUNTIMES)
    def test_cache_saved_on_del(self, _name, use_python_runtime):
        self._skip_if_cpp_unavailable(use_python_runtime)
        model, inputs = _fresh_conv_model_and_inputs()
        compiled = _compile(
            model,
            inputs,
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )
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

    @parameterized.expand(_RUNTIMES)
    def test_cache_file_nonempty(self, _name, use_python_runtime):
        self._skip_if_cpp_unavailable(use_python_runtime)
        model, inputs = _fresh_conv_model_and_inputs()
        compiled = _compile(
            model,
            inputs,
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertGreater(
            os.path.getsize(self.cache_path),
            0,
            "Cache file should have nonzero size",
        )

    @parameterized.expand(_RUNTIMES)
    def test_cache_roundtrip(self, _name, use_python_runtime):
        """Populate + save, then recompile and confirm correctness against eager output."""
        self._skip_if_cpp_unavailable(use_python_runtime)
        model, inputs = _fresh_conv_model_and_inputs()
        with torch.no_grad():
            ref_output = model(*inputs)

        compiled1 = _compile(
            model,
            inputs,
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )
        out1 = compiled1(*[inp.clone() for inp in inputs])
        self.assertGreater(
            cosine_similarity(ref_output, out1),
            COSINE_THRESHOLD,
            "First compiled output should match eager",
        )
        del compiled1
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))

        compiled2 = _compile(
            model,
            inputs,
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )
        out2 = compiled2(*[inp.clone() for inp in inputs])
        self.assertGreater(
            cosine_similarity(ref_output, out2),
            COSINE_THRESHOLD,
            "Second compiled output (warm cache) should still match eager",
        )

    @parameterized.expand(_RUNTIMES)
    def test_save_creates_directory(self, _name, use_python_runtime):
        self._skip_if_cpp_unavailable(use_python_runtime)
        nested_path = os.path.join(self.cache_dir, "a", "b", "c", "runtime_cache.bin")
        model, inputs = _fresh_conv_model_and_inputs()
        compiled = _compile(
            model,
            inputs,
            use_python_runtime=use_python_runtime,
            runtime_cache_path=nested_path,
        )
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
    """Tests that file locking works for concurrent access (Python runtime only)."""

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
        from filelock import FileLock

        lock = FileLock(self.cache_path + ".lock")
        with lock.acquire(timeout=5):
            data = open(self.cache_path, "rb").read()
        self.assertGreater(len(data), 0)

    def test_sequential_save_load(self):
        """Two modules saving and loading from the same path should not corrupt data."""
        compiled1, inputs = _compile_simple(runtime_cache_path=self.cache_path)
        _ = compiled1(*[inp.clone() for inp in inputs])
        del compiled1
        gc.collect()
        size1 = os.path.getsize(self.cache_path)

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
        if os.path.isfile(TIMING_CACHE_PATH):
            os.remove(TIMING_CACHE_PATH)
        compiled, inputs = _compile_simple()
        _ = compiled(*[inp.clone() for inp in inputs])
        self.assertTrue(
            os.path.isfile(TIMING_CACHE_PATH),
            "Timing cache should still be created for standard TRT",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "New serialization indices are registered only on TensorRT-RTX builds",
)
class TestCppSerializationIndices(TestCase):
    """Verify the new RTX-only C++ serialization indices are registered by the runtime."""

    def test_new_indices_registered(self):
        self.assertEqual(int(torch.ops.tensorrt.ABI_VERSION()), 9)
        self.assertEqual(int(torch.ops.tensorrt.SERIALIZATION_LEN()), 15)
        self.assertEqual(int(torch.ops.tensorrt.RUNTIME_CACHE_PATH_IDX()), 12)
        self.assertEqual(
            int(torch.ops.tensorrt.DYNAMIC_SHAPES_KERNEL_STRATEGY_IDX()), 13
        )
        self.assertEqual(int(torch.ops.tensorrt.CUDA_GRAPH_STRATEGY_IDX()), 14)


if __name__ == "__main__":
    run_tests()
