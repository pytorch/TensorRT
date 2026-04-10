import gc
import importlib
import os
import shutil
import tempfile
import time
import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
class TestRuntimeCacheModels(TestCase):
    """End-to-end model tests with runtime cache enabled."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    def test_resnet18_with_runtime_cache(self):
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()

        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
            runtime_cache_path=self.cache_path,
        )

        ref_output = model(input_tensor)
        trt_output = compiled(input_tensor)

        cos_sim = cosine_similarity(ref_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"ResNet18 cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD}",
        )

        # Verify runtime cache is saved on cleanup
        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(self.cache_path),
            "Runtime cache should be saved after ResNet18 inference",
        )

    def test_resnet18_cache_reuse(self):
        """Compile + infer twice with same cache path. Second run should load cached data."""
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()
        ref_output = model(input_tensor)

        compile_kwargs = {
            "ir": "dynamo",
            "inputs": [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            "enabled_precisions": {torch.float32},
            "use_python_runtime": True,
            "min_block_size": 1,
            "runtime_cache_path": self.cache_path,
        }

        # First compilation — cold cache
        compiled1 = torchtrt.compile(model, **compile_kwargs)
        _ = compiled1(input_tensor)
        del compiled1
        gc.collect()
        torch._dynamo.reset()
        self.assertTrue(os.path.isfile(self.cache_path))
        cache_size_1 = os.path.getsize(self.cache_path)

        # Second compilation — warm cache
        compiled2 = torchtrt.compile(model, **compile_kwargs)
        output2 = compiled2(input_tensor)

        cos_sim = cosine_similarity(ref_output, output2)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"ResNet18 (cached) cosine similarity {cos_sim} below threshold",
        )

        del compiled2
        gc.collect()
        cache_size_2 = os.path.getsize(self.cache_path)
        # Cache should exist and be non-empty after both runs
        self.assertGreater(cache_size_1, 0)
        self.assertGreater(cache_size_2, 0)

    def test_mobilenet_v2_with_runtime_cache(self):
        import torchvision.models as models

        model = models.mobilenet_v2(pretrained=True).eval().cuda()
        input_tensor = torch.randn(1, 3, 224, 224).cuda()

        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
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
    """Tests runtime cache with dynamic input shapes."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    def test_dynamic_batch_with_cache(self):
        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = ConvModel().eval().cuda()

        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[
                torchtrt.Input(
                    min_shape=(1, 3, 32, 32),
                    opt_shape=(4, 3, 32, 32),
                    max_shape=(8, 3, 32, 32),
                    dtype=torch.float32,
                )
            ],
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
            runtime_cache_path=self.cache_path,
        )

        # Test with batch size 1
        input_bs1 = torch.randn(1, 3, 32, 32).cuda()
        ref_bs1 = model(input_bs1)
        out_bs1 = compiled(input_bs1)
        cos_sim_1 = cosine_similarity(ref_bs1, out_bs1)
        self.assertTrue(
            cos_sim_1 > COSINE_THRESHOLD,
            f"BS=1 cosine similarity {cos_sim_1} below threshold",
        )

        # Test with batch size 4
        input_bs4 = torch.randn(4, 3, 32, 32).cuda()
        ref_bs4 = model(input_bs4)
        out_bs4 = compiled(input_bs4)
        cos_sim_4 = cosine_similarity(ref_bs4, out_bs4)
        self.assertTrue(
            cos_sim_4 > COSINE_THRESHOLD,
            f"BS=4 cosine similarity {cos_sim_4} below threshold",
        )

        # Verify cache is saved
        del compiled
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))

    def test_cache_valid_across_shapes(self):
        """Save cache from one shape, load and verify it works with another shape in range."""

        class SimpleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleConv().eval().cuda()

        compile_kwargs = {
            "ir": "dynamo",
            "inputs": [
                torchtrt.Input(
                    min_shape=(1, 3, 16, 16),
                    opt_shape=(2, 3, 16, 16),
                    max_shape=(4, 3, 16, 16),
                    dtype=torch.float32,
                )
            ],
            "enabled_precisions": {torch.float32},
            "use_python_runtime": True,
            "min_block_size": 1,
            "runtime_cache_path": self.cache_path,
        }

        # First run with batch=2 — saves cache
        compiled1 = torchtrt.compile(model, **compile_kwargs)
        input_bs2 = torch.randn(2, 3, 16, 16).cuda()
        _ = compiled1(input_bs2)
        del compiled1
        gc.collect()
        torch._dynamo.reset()
        self.assertTrue(os.path.isfile(self.cache_path))

        # Second run with batch=3 — loads same cache
        compiled2 = torchtrt.compile(model, **compile_kwargs)
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

    def test_warmup_timing(self):
        """Measure cold vs warm cache inference time. Informational only — no strict pass/fail."""

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
            "ir": "dynamo",
            "inputs": [torchtrt.Input(input_tensor.shape, dtype=torch.float32)],
            "enabled_precisions": {torch.float32},
            "use_python_runtime": True,
            "min_block_size": 1,
            "runtime_cache_path": self.cache_path,
        }

        # Cold cache compilation + inference
        compiled1 = torchtrt.compile(model, **compile_kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled1(input_tensor)
        torch.cuda.synchronize()
        cold_time = time.perf_counter() - start
        del compiled1
        gc.collect()
        torch._dynamo.reset()

        # Warm cache compilation + inference
        compiled2 = torchtrt.compile(model, **compile_kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled2(input_tensor)
        torch.cuda.synchronize()
        warm_time = time.perf_counter() - start

        print(f"\n  Cold cache first inference: {cold_time*1000:.1f}ms")
        print(f"  Warm cache first inference: {warm_time*1000:.1f}ms")
        print(f"  Speedup: {cold_time/warm_time:.2f}x")

        # No strict assertion — just log for visibility
        self.assertTrue(True, "Timing test completed (informational)")


if __name__ == "__main__":
    run_tests()
