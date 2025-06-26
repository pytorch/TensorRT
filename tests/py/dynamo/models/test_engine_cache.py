# type: ignore
import importlib
import os
import shutil
import unittest
from typing import Optional

import pytest
import torch
import torch_tensorrt as torch_trt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_dir: str,
    ) -> None:
        self.engine_cache_dir = engine_cache_dir
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

        self.hashes = {}

    def save(
        self,
        hash: str,
        blob: bytes,
        prefix: str = "blob",
    ):
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

        path = os.path.join(
            self.engine_cache_dir,
            f"{prefix}_{hash}.bin",
        )
        with open(path, "wb") as f:
            f.write(blob)

        self.hashes[hash] = 0

    def load(self, hash: str, prefix: str = "blob") -> Optional[bytes]:
        path = os.path.join(self.engine_cache_dir, f"{prefix}_{hash}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                blob = f.read()
            self.hashes[hash] += 1
            return blob
        return None


class TestHashFunction(TestCase):
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_reexport_is_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings2 = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertEqual(hash1, hash2)

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_input_shape_change_is_not_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 300, 300),
                opt_shape=(100, 3, 300, 300),
                max_shape=(200, 3, 300, 300),
            ),
        )
        settings2 = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertNotEqual(hash1, hash2)

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_engine_settings_is_not_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            immutable_weights=False,
            cache_built_engines=True,
            reuse_cached_engines=True,
            enabled_precisions={torch.float32},
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings2 = CompilationSettings(
            immutable_weights=False,
            cache_built_engines=True,
            reuse_cached_engines=True,
            enabled_precisions={torch.float32, torch.float16},
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertNotEqual(hash1, hash2)


class TestEngineCache(TestCase):
    @pytest.mark.xfail
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_dynamo_compile_with_default_disk_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        engine_cache_dir = "/tmp/test_torch_dynamo_with_default_disk_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            # remove timing cache and reset dynamo for engine caching messurement
            remove_timing_cache()
            torch._dynamo.reset()
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            torch.cuda.synchronize()
            start.record()
            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=True,
                enabled_precisions={torch.float},
                min_block_size=1,
                immutable_weights=False,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            times[0] > times[2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {times[0]} ms, time taken with engine caching: {times[2]} ms",
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_dynamo_compile_with_custom_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_dynamo_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)

        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]
        results = []
        for i in range(3):
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=True,
                enabled_precisions={torch.float},
                min_block_size=1,
                immutable_weights=False,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                custom_engine_cache=custom_engine_cache,
            )
            results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        [
            assertions.assertTrue(
                count == 1,
                f"cache was not hit exactly once for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_dynamo_compile_change_input_shape(self):
        """Runs compilation 3 times, the cache should miss each time"""
        model = models.resnet18(pretrained=True).eval().to("cuda")
        # Mark the dim0 of inputs as dynamic

        engine_cache_dir = "/tmp/test_torch_dynamo_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)

        for i in range(3):
            inputs = (torch.rand((4 * (i + 1), 3, 224, 224)).to("cuda"),)
            trt_gm = torch_trt.dynamo.compile(
                torch.export.export(model, args=inputs),
                inputs=inputs,
                use_python_runtime=False,
                enabled_precisions={torch.float},
                min_block_size=1,
                immutable_weights=False,
                cache_built_engines=True,
                reuse_cached_engines=True,
            )

        [
            assertions.assertTrue(
                count == 0, f"Unintended cache hit for entry ({h}, hit: {count})"
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    @pytest.mark.xfail
    def test_torch_compile_with_default_disk_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_compile_with_default_disk_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            # remove timing cache and reset dynamo for engine caching measurement
            remove_timing_cache()
            torch._dynamo.reset()
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            torch.cuda.synchronize()
            start.record()
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": False,
                    "enabled_precisions": {torch.float},
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "engine_cache_dir": engine_cache_dir,
                    "engine_cache_size": 1 << 30,  # 1GB
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            times[0] > times[2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {times[0]} ms, time taken with engine caching: {times[2]} ms",
        )

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_torch_compile_with_custom_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_compile_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            start.record()
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": False,
                    "enabled_precisions": {torch.float},
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "custom_engine_cache": custom_engine_cache,
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation
            end.record()
            torch.cuda.synchronize()
            torch._dynamo.reset()
            times.append(start.elapsed_time(end))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        [
            assertions.assertTrue(
                count == 1,
                f"cache was not hit exactly once for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_torch_trt_compile_change_input_shape(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")
        engine_cache_dir = "/tmp/test_torch_trt_compile_change_input_shape"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        for i in range(3):
            inputs = [torch.rand((4 * (i + 1), 3, 224, 224)).to("cuda")]
            compiled_model = torch_trt.compile(
                model,
                inputs=inputs,
                **{
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "custom_engine_cache": custom_engine_cache,
                },
            )
            compiled_model(*inputs)
        [
            assertions.assertTrue(
                count == 0, f"Unintended cache hit for entry ({h}, hit: {count})"
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    def test_torch_compile_graph_break(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                x = x + x
                x = x + x
                x = torch.ops.aten.relu.default(x)
                x = x + x
                x = x + x
                x = torch.ops.aten.relu.default(x)
                x = x + x
                x = x + x
                return x

        model = MyModel().eval().cuda()
        engine_cache_dir = "/tmp/test_torch_compile_graph_break"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        inputs = [torch.rand((3, 3, 224, 224)).to("cuda")]
        for i in range(3):
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "custom_engine_cache": custom_engine_cache,
                    "torch_executed_ops": {"torch.ops.aten.relu.default"},
                },
            )
            compiled_model(*inputs)

        [
            assertions.assertTrue(
                count == 2,
                f"cache was not hit exactly twice for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    def test_isomorphic_graphs(self):
        class MyModel1(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        class MyModel2(torch.nn.Module):
            def forward(self, c, d):
                return c + d

        model1 = MyModel1().eval().cuda()
        model2 = MyModel2().eval().cuda()

        inputs1 = (torch.randn((2, 3)).to("cuda"), torch.randn((2, 3)).to("cuda"))
        inputs2 = (torch.randn((2, 3)).to("cuda"), torch.randn((2, 3)).to("cuda"))

        exp_program1 = torch.export.export(model1, args=inputs1)
        exp_program2 = torch.export.export(model2, args=inputs2)

        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3),
                opt_shape=(2, 3),
                max_shape=(10, 3),
            ),
        )

        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3),
                opt_shape=(2, 3),
                max_shape=(10, 3),
            ),
        )

        settings1 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )

        settings2 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )

        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        assertions.assertEqual(hash1, hash2)

    # @unittest.skip("benchmark on small models")
    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_caching_small_model(self):
        from torch_tensorrt.dynamo._refit import refit_module_weights

        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_caching_small_model"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        inputs = (torch.rand((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(model, args=inputs)

        # warm up
        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            inputs,
            use_python_runtime=True,
            enabled_precisions={torch.float},
            min_block_size=1,
            immutable_weights=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
            strip_engine_weights=False,
            refit_identical_engine_weights=False,
        )
        torch.cuda.empty_cache()

        compile_times = [[] for _ in range(3)]
        inference_times = [[] for _ in range(3)]
        results = [[] for _ in range(3)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        interval = 3
        for i in range(interval * 3):
            if i < interval:
                # non-refittable
                immutable_weights = True
                strip_engine_weights = False
                refit_identical_engine_weights = False
                cache_built_engines = reuse_cached_engines = False
                # continue
            elif i < interval * 2:
                # REFIT w/ engine caching
                immutable_weights = False
                strip_engine_weights = False
                refit_identical_engine_weights = False
                cache_built_engines = reuse_cached_engines = True
                # continue
            else:
                # REFIT_IDENTICAL w/ engine caching
                immutable_weights = False
                strip_engine_weights = False
                refit_identical_engine_weights = True
                cache_built_engines = reuse_cached_engines = True
                # continue

            if i % interval == 0:
                remove_timing_cache()

            torch._dynamo.reset()

            torch.cuda.synchronize()
            start.record()

            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=True,
                enabled_precisions={torch.float},
                min_block_size=1,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
                engine_cache_size=1 << 40,
                immutable_weights=immutable_weights,
                strip_engine_weights=strip_engine_weights,
                refit_identical_engine_weights=refit_identical_engine_weights,
            )

            if strip_engine_weights:
                trt_gm = refit_module_weights(trt_gm, exp_program)

            end.record()
            torch.cuda.synchronize()
            compile_times[i // interval].append(start.elapsed_time(end))

            # inference
            torch.cuda.synchronize()
            start.record()
            out = trt_gm(*inputs)
            end.record()
            torch.cuda.synchronize()
            inference_times[i // interval].append(start.elapsed_time(end))

            results[i // interval].append(out)

            torch.cuda.empty_cache()

        cos_sim = cosine_similarity(torch.stack(results[0]), torch.stack(results[1]))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(torch.stack(results[1]), torch.stack(results[2]))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            compile_times[1][0] > compile_times[1][1],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[1][0]} ms, time taken with engine caching: {compile_times[1][1]} ms",
        )

        assertions.assertTrue(
            compile_times[1][0] > compile_times[1][2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[1][0]} ms, time taken with engine caching: {compile_times[1][2]} ms",
        )

        assertions.assertTrue(
            compile_times[2][0] > compile_times[2][1],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[2][0]} ms, time taken with engine caching: {compile_times[2][1]} ms",
        )

        assertions.assertTrue(
            compile_times[2][0] > compile_times[2][2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[2][0]} ms, time taken with engine caching: {compile_times[2][2]} ms",
        )

        assertions.assertTrue(
            compile_times[0][2] > compile_times[1][2],
            msg=f"Engine caching is slower than recompiling a non-refittable engine. Recompile a non-refittable engine: {compile_times[0][2]} ms, time taken with engine caching: {compile_times[1][2]} ms",
        )

        assertions.assertTrue(
            compile_times[0][2] > compile_times[2][2],
            msg=f"Engine caching is slower than recompiling a non-refittable engine. Recompile a non-refittable engine: {compile_times[0][2]} ms, time taken with engine caching: {compile_times[2][2]} ms",
        )

    @unittest.skip("benchmark on llama2")
    def test_caching_llama2_model(self):
        import torch
        from torch_tensorrt.dynamo._refit import refit_module_weights
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteriaList,
        )
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            MaxLengthCriteria,
        )

        def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
            """
            Exports the LLM model into an ExportedProgram with dynamic shapes.
            In the case of guard failures due to some PyTorch kernel implements, we also
            try to re-export the graph by expressing them as runtime assert nodes
            """
            with torch.no_grad():
                # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
                seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
                try:
                    print("Trying to export the model using torch.export.export()..")
                    # strict=False only enables aotautograd tracing and excludes dynamo.
                    ep = torch.export.export(
                        model, (inputs,), dynamic_shapes=({1: seq_len},), strict=False
                    )
                except:
                    print(
                        "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
                    )
                    # This API is used to express the constraint violation guards as asserts in the graph.
                    ep = torch.export._trace._export(
                        model,
                        (inputs,),
                        dynamic_shapes=({1: seq_len},),
                        strict=False,
                        allow_complex_guards_as_runtime_asserts=True,
                    )

            return ep

        def generate(model, input_seq, max_tokens, eos_token_id):
            """
            Greedy decoding of the model. This generates up to max_tokens.
            """
            # Max length of output seq = current input_seq length + max_tokens allowed to generate
            max_output_seq_length = input_seq.shape[1] + max_tokens
            stopping_criteria = StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=max_output_seq_length),
                    EosTokenCriteria(eos_token_id=eos_token_id),
                ]
            )

            while True:
                outputs = model(input_seq)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
                # TODO: Handle batch in this check
                if stopping_criteria(input_seq, logits).item():
                    break

            return input_seq

        MAX_TOKENS = 32
        DEVICE = torch.device("cuda:0")

        llama_path = "meta-llama/Llama-2-7b-chat-hf"
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                llama_path, use_cache=False, attn_implementation="eager"
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(llama_path)

        prompt = "What is dynamic programming?"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs.input_ids

        llama2_ep = export_llm(model, input_ids, max_seq_len=64)

        engine_cache_dir = "/tmp/test_caching_llama2_model"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        timing_cache_path = os.path.join(
            engine_cache_dir, "llama2_timing_cache_original.bin"
        )

        def remove_timing_cache(path=timing_cache_path):
            if os.path.exists(path):
                os.remove(path)

        input_ids = input_ids.to(DEVICE)

        # warm up
        trt_gm = torch_trt.dynamo.compile(
            llama2_ep,
            inputs=[input_ids],
            use_python_runtime=True,
            enabled_precisions={torch.float32},
            min_block_size=1,
            immutable_weights=False,
            truncate_double=True,
            device=DEVICE,
            disable_tf32=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
            strip_engine_weights=False,
            refit_identical_engine_weights=False,
            timing_cache_path=timing_cache_path,
        )
        torch.cuda.empty_cache()

        compile_times = [[] for _ in range(3)]
        inference_times = [[] for _ in range(3)]
        results = [[] for _ in range(3)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        interval = 3
        for i in range(interval * 3):
            if i < interval:
                # non-refittable
                immutable_weights = True
                strip_engine_weights = False
                refit_identical_engine_weights = False
                cache_built_engines = reuse_cached_engines = False
            elif i < interval * 2:
                # REFIT w/ engine caching
                immutable_weights = False
                strip_engine_weights = False
                refit_identical_engine_weights = False
                cache_built_engines = reuse_cached_engines = True
            else:
                # REFIT_IDENTICAL w/ engine caching
                immutable_weights = False
                strip_engine_weights = False
                refit_identical_engine_weights = True
                cache_built_engines = reuse_cached_engines = True

            if i % interval == 0:
                remove_timing_cache()

            torch._dynamo.reset()

            torch.cuda.synchronize()
            start.record()

            trt_gm = torch_trt.dynamo.compile(
                llama2_ep,
                inputs=[input_ids],
                use_python_runtime=True,
                enabled_precisions={torch.float32},
                min_block_size=1,
                truncate_double=True,
                device=DEVICE,
                disable_tf32=True,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
                engine_cache_size=1 << 40,
                immutable_weights=immutable_weights,
                strip_engine_weights=strip_engine_weights,
                refit_identical_engine_weights=refit_identical_engine_weights,
                timing_cache_path=timing_cache_path,
            )

            if strip_engine_weights:
                trt_gm = refit_module_weights(trt_gm, llama2_ep)

            end.record()
            torch.cuda.synchronize()

            compile_times[i // interval].append(start.elapsed_time(end))

            # inference
            torch.cuda.synchronize()
            start.record()

            trt_gen_tokens = generate(
                trt_gm, input_ids, MAX_TOKENS, tokenizer.eos_token_id
            )
            # trt_gen_text = tokenizer.batch_decode(
            #     trt_gen_tokens,
            #     skip_special_tokens=True,
            #     clean_up_tokenization_spaces=False,
            # )[0],
            results[i // interval].append(trt_gen_tokens)

            end.record()
            torch.cuda.synchronize()

            inference_times[i // interval].append(start.elapsed_time(end))

            torch.cuda.empty_cache()

        cos_sim = cosine_similarity(torch.stack(results[0]), torch.stack(results[1]))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(torch.stack(results[1]), torch.stack(results[2]))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            compile_times[1][0] > compile_times[1][1],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[1][0]} ms, time taken with engine caching: {compile_times[1][1]} ms",
        )

        assertions.assertTrue(
            compile_times[1][0] > compile_times[1][2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[1][0]} ms, time taken with engine caching: {compile_times[1][2]} ms",
        )

        assertions.assertTrue(
            compile_times[2][0] > compile_times[2][1],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[2][0]} ms, time taken with engine caching: {compile_times[2][1]} ms",
        )

        assertions.assertTrue(
            compile_times[2][0] > compile_times[2][2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {compile_times[2][0]} ms, time taken with engine caching: {compile_times[2][2]} ms",
        )

        assertions.assertTrue(
            compile_times[0][2] > compile_times[1][2],
            msg=f"Engine caching is slower than recompiling a non-refittable engine. Recompile a non-refittable engine: {compile_times[0][2]} ms, time taken with engine caching: {compile_times[1][2]} ms",
        )

        assertions.assertTrue(
            compile_times[0][2] > compile_times[2][2],
            msg=f"Engine caching is slower than recompiling a non-refittable engine. Recompile a non-refittable engine: {compile_times[0][2]} ms, time taken with engine caching: {compile_times[2][2]} ms",
        )
