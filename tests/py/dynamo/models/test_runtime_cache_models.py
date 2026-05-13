import gc
import importlib
import os
import shutil
import tempfile
import time
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

# Parameterize end-to-end cache tests over both runtime paths. The C++ variant is
# skipped inside the test body when the C++ runtime is not available.
_RUNTIMES = [("python", True), ("cpp", False)]


def _compile(model, inputs, *, use_python_runtime, runtime_cache_path):
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "enabled_precisions": {torch.float32},
        "use_python_runtime": use_python_runtime,
        "min_block_size": 1,
        "runtime_cache_path": runtime_cache_path,
    }
    return torchtrt.compile(model, **kwargs)


def _skip_if_cpp_unavailable(testcase, use_python_runtime):
    if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
        testcase.skipTest("C++ runtime is not available")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
class TestRuntimeCacheModels(TestCase):
    """End-to-end model tests with runtime cache enabled — both runtimes."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    @parameterized.expand(_RUNTIMES)
    def test_resnet18_with_runtime_cache(self, _name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()

        compiled = _compile(
            model,
            [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )

        ref_output = model(input_tensor)
        trt_output = compiled(input_tensor)

        cos_sim = cosine_similarity(ref_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"ResNet18 cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD}",
        )

        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(self.cache_path),
            "Runtime cache should be saved after ResNet18 inference",
        )

    @parameterized.expand(_RUNTIMES)
    def test_resnet18_cache_reuse(self, _name, use_python_runtime):
        """Compile + infer twice with same cache path. Second run loads cached data."""
        _skip_if_cpp_unavailable(self, use_python_runtime)
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()
        ref_output = model(input_tensor)

        compile_kwargs = {
            "inputs": [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            "use_python_runtime": use_python_runtime,
            "runtime_cache_path": self.cache_path,
        }

        # First compilation — cold cache
        compiled1 = _compile(model, **compile_kwargs)
        _ = compiled1(input_tensor)
        del compiled1
        gc.collect()
        torch._dynamo.reset()
        self.assertTrue(os.path.isfile(self.cache_path))
        cache_size_1 = os.path.getsize(self.cache_path)

        # Second compilation — warm cache
        compiled2 = _compile(model, **compile_kwargs)
        output2 = compiled2(input_tensor)

        cos_sim = cosine_similarity(ref_output, output2)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"ResNet18 (cached) cosine similarity {cos_sim} below threshold",
        )

        del compiled2
        gc.collect()
        cache_size_2 = os.path.getsize(self.cache_path)
        self.assertGreater(cache_size_1, 0)
        self.assertGreater(cache_size_2, 0)

    @parameterized.expand(_RUNTIMES)
    def test_mobilenet_v2_with_runtime_cache(self, _name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        import torchvision.models as models

        model = models.mobilenet_v2(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()

        compiled = _compile(
            model,
            [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )

        ref_output = model(input_tensor)
        trt_output = compiled(input_tensor)

        cos_sim = cosine_similarity(ref_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"MobileNetV2 cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD}",
        )

        del compiled
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheDynamicShapes(TestCase):
    """Tests runtime cache with dynamic input shapes, exercised on both runtimes."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    @parameterized.expand(_RUNTIMES)
    def test_dynamic_batch_with_cache(self, _name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = ConvModel().eval().cuda()

        compiled = _compile(
            model,
            [
                torchtrt.Input(
                    min_shape=(1, 3, 32, 32),
                    opt_shape=(4, 3, 32, 32),
                    max_shape=(8, 3, 32, 32),
                    dtype=torch.float32,
                )
            ],
            use_python_runtime=use_python_runtime,
            runtime_cache_path=self.cache_path,
        )

        for batch_size in (1, 4):
            input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()
            ref_output = model(input_tensor)
            out = compiled(input_tensor)
            cos_sim = cosine_similarity(ref_output, out)
            self.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                f"BS={batch_size} cosine similarity {cos_sim} below threshold",
            )

        del compiled
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))

    @parameterized.expand(_RUNTIMES)
    def test_cache_valid_across_shapes(self, _name, use_python_runtime):
        """Save cache from one shape, load and verify it works with another shape in range."""
        _skip_if_cpp_unavailable(self, use_python_runtime)

        class SimpleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleConv().eval().cuda()

        compile_kwargs = {
            "inputs": [
                torchtrt.Input(
                    min_shape=(1, 3, 16, 16),
                    opt_shape=(2, 3, 16, 16),
                    max_shape=(4, 3, 16, 16),
                    dtype=torch.float32,
                )
            ],
            "use_python_runtime": use_python_runtime,
            "runtime_cache_path": self.cache_path,
        }

        # First run with batch=2 — saves cache
        compiled1 = _compile(model, **compile_kwargs)
        input_bs2 = torch.randn(2, 3, 16, 16).cuda()
        _ = compiled1(input_bs2)
        del compiled1
        gc.collect()
        torch._dynamo.reset()
        self.assertTrue(os.path.isfile(self.cache_path))

        # Second run with batch=3 — loads same cache
        compiled2 = _compile(model, **compile_kwargs)
        input_bs3 = torch.randn(3, 3, 16, 16).cuda()
        ref_bs3 = model(input_bs3)
        out_bs3 = compiled2(input_bs3)

        cos_sim = cosine_similarity(ref_bs3, out_bs3)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"Cross-shape cache reuse cosine similarity {cos_sim} below threshold",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCachePerformance(TestCase):
    """Informational timing tests for runtime cache warm-up behavior."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    @parameterized.expand(_RUNTIMES)
    def test_warmup_timing(self, _name, use_python_runtime):
        """Measure cold vs warm cache inference time. Informational — no strict assertion."""
        _skip_if_cpp_unavailable(self, use_python_runtime)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 512)
                self.fc2 = torch.nn.Linear(512, 256)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = MLP().eval().cuda()
        input_tensor = torch.randn(16, 256).cuda()

        compile_kwargs = {
            "inputs": [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            "use_python_runtime": use_python_runtime,
            "runtime_cache_path": self.cache_path,
        }

        compiled1 = _compile(model, **compile_kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled1(input_tensor)
        torch.cuda.synchronize()
        cold_time = time.perf_counter() - start
        del compiled1
        gc.collect()
        torch._dynamo.reset()

        compiled2 = _compile(model, **compile_kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled2(input_tensor)
        torch.cuda.synchronize()
        warm_time = time.perf_counter() - start

        print(f"\n  [{_name}] Cold cache first inference: {cold_time*1000:.1f}ms")
        print(f"  [{_name}] Warm cache first inference: {warm_time*1000:.1f}ms")
        print(f"  [{_name}] Speedup: {cold_time/warm_time:.2f}x")
        self.assertTrue(True, "Timing test completed (informational)")


if __name__ == "__main__":
    run_tests()
