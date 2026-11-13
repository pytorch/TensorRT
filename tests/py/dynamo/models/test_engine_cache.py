# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
import importlib
import os
import pickle
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from typing import Optional
from unittest import mock

import pytest
import torch
import torch_tensorrt as torch_trt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_cache import (
    BaseEngineCache,
    _canonicalize_setting_value,
)
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias

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
    def test_unordered_settings_are_hashed_in_a_stable_order(self):
        """Engine-invariant settings that are sets must not depend on hash order.

        str() of a set lays elements out by hash, and string hashing is
        randomized per process, so hashing one directly would give the same
        settings a different cache key on every run.
        """
        self.assertEqual(
            _canonicalize_setting_value({"b", "a", "c"}), "['a', 'b', 'c']"
        )
        self.assertEqual(
            _canonicalize_setting_value(frozenset({"b", "a"})),
            _canonicalize_setting_value({"a", "b"}),
        )
        self.assertEqual(_canonicalize_setting_value(True), "True")

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_reexport_is_equal(self):
        pyt_model = models.resnet18(weights=None).eval().to("cuda")
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
        pyt_model = models.resnet18(weights=None).eval().to("cuda")
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
        pyt_model = models.resnet18(weights=None).eval().to("cuda")
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
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(400, 3, 224, 224),
            ),
        )
        settings2 = CompilationSettings(
            immutable_weights=False,
            cache_built_engines=True,
            reuse_cached_engines=True,
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertNotEqual(hash1, hash2)

    def _hash_on(self, capability, settings, trt_version=None):
        """Hash one graph as if it were built on a GPU of ``capability``."""
        x = torch.randn(4, 8)
        gm = torch.export.export(torch.nn.Linear(8, 8), (x,)).module()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                )
            )
            if trt_version is not None:
                stack.enter_context(mock.patch.object(trt, "__version__", trt_version))
            return BaseEngineCache.get_hash(
                gm, (torch_trt.Input(shape=(4, 8)),), settings
            )

    def test_engine_built_for_another_gpu_is_not_equal(self):
        """An engine only deserializes on the GPU architecture it was built for, so
        a cache directory shared by two hosts must not hand one host's engine to the
        other."""
        settings = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        self.assertNotEqual(
            self._hash_on((9, 0), settings), self._hash_on((10, 3), settings)
        )

    def test_hardware_compatible_engine_ignores_gpu(self):
        settings = CompilationSettings(
            immutable_weights=False,
            hardware_compatible=True,
            cache_built_engines=True,
            reuse_cached_engines=True,
        )
        self.assertEqual(
            self._hash_on((9, 0), settings), self._hash_on((10, 3), settings)
        )

    def test_engine_built_by_another_tensorrt_is_not_equal(self):
        settings = CompilationSettings(
            immutable_weights=False, cache_built_engines=True, reuse_cached_engines=True
        )
        self.assertNotEqual(
            self._hash_on((9, 0), settings),
            self._hash_on((9, 0), settings, trt_version="0.0.0"),
        )


class TestUndeserializableCacheEntry(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    def test_entry_that_does_not_deserialize_is_rebuilt(self):
        """A hit whose engine TensorRT cannot load (another TensorRT version, another
        GPU) is a miss: compile rebuilds instead of failing inside the refitter."""
        model = torch.nn.Linear(8, 8).eval().cuda()
        x = torch.randn(4, 8, device="cuda")
        exp_program = torch.export.export(model, (x,))
        engine_cache_dir = tempfile.mkdtemp(prefix="test_undeserializable_entry_")
        self.addCleanup(shutil.rmtree, engine_cache_dir, ignore_errors=True)
        engine_cache = MyEngineCache(engine_cache_dir)

        def _compile():
            return torch_trt.dynamo.compile(
                exp_program,
                inputs=[x],
                min_block_size=1,
                immutable_weights=False,
                cache_built_engines=True,
                reuse_cached_engines=True,
                custom_engine_cache=engine_cache,
            )

        _compile()
        self.assertEqual(len(engine_cache.hashes), 1)
        # Replace the stored engine with bytes TensorRT cannot deserialize, the way
        # an entry written for another GPU or by another TensorRT version looks.
        for name in os.listdir(engine_cache_dir):
            path = os.path.join(engine_cache_dir, name)
            with open(path, "rb") as f:
                unpacked = pickle.loads(f.read())
            unpacked["serialized_engine"] = b"not a TensorRT engine"
            with open(path, "wb") as f:
                f.write(pickle.dumps(unpacked))

        trt_gm = _compile()
        torch.testing.assert_close(trt_gm(x), model(x), rtol=1e-2, atol=1e-2)


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
    def test_cache_hit_with_a_calibration_dataloader(self):
        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
        model = model.eval().to("cuda")
        x = torch.randn(4, 16, device="cuda")
        # A lambda collate_fn cannot be pickled, as a saved engine's settings are.
        loader = torch.utils.data.DataLoader(
            [x], batch_size=None, collate_fn=lambda batch: (batch,)
        )
        with tempfile.TemporaryDirectory() as engine_cache_dir:
            custom_engine_cache = MyEngineCache(engine_cache_dir)
            outputs = []
            for _ in range(2):
                trt_gm = torch_trt.dynamo.compile(
                    torch.export.export(model, (x,)),
                    (x,),
                    min_block_size=1,
                    immutable_weights=False,
                    cache_built_engines=True,
                    reuse_cached_engines=True,
                    custom_engine_cache=custom_engine_cache,
                    enable_autocast=True,
                    autocast_low_precision_type=torch.float16,
                    autocast_calibration_dataloader=loader,
                )
                outputs.append(trt_gm(x))
        assertions.assertEqual(list(custom_engine_cache.hashes.values()), [1])
        torch.testing.assert_close(outputs[1], outputs[0])

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
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "engine_cache_dir": engine_cache_dir,
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

    def test_graphs_that_differ_in_more_than_weight_values_are_not_equal(self):
        class Permute(torch.nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return x.permute(self.dims)

        class Scale(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.register_buffer("w", torch.ones(3, dtype=dtype, device="cuda"))

            def forward(self, x):
                return x * self.w

        class Cond(torch.nn.Module):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch

            def forward(self, x):
                return torch.cond(x.sum() > 0, self.branch, lambda t: -t, (x,))

        x = torch.randn(3, 3, device="cuda")
        pairs = {
            "argument": (Permute((0, 1)), Permute((1, 0))),
            "cond branch": (Cond(lambda t: t.sin()), Cond(lambda t: t.cos())),
            "weight shape": (
                torch.nn.Linear(3, 4).cuda(),
                torch.nn.Linear(3, 5).cuda(),
            ),
            # A converter can build other layers for another weight dtype.
            "weight dtype": (Scale(torch.float32), Scale(torch.float16)),
        }
        for name, models in pairs.items():
            with self.subTest(name):
                hashes = {
                    BaseEngineCache.get_hash(
                        torch.export.export(model, (x,)).module(),
                        (torch_trt.Input(shape=(3, 3)),),
                        CompilationSettings(),
                    )
                    for model in models
                }
                self.assertEqual(len(hashes), 2)

    # @unittest.skip("benchmark on small models")
    @unittest.skipIf(
        torch_trt.ENABLED_FEATURES.tensorrt_rtx,
        "TensorRT-RTX compiles without build-time autotuning, so engine caching does not speed up compilation",
    )
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

        # Do not use the global timing cache here.  Pytest-xdist workers share it,
        # and Windows does not allow one worker to remove it while another has it
        # open.
        timing_cache_path = os.path.join(engine_cache_dir, "timing_cache.bin")

        def remove_timing_cache(path=timing_cache_path):
            if os.path.exists(path):
                os.remove(path)

        inputs = (torch.rand((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(model, args=inputs)

        # warm up
        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            inputs,
            min_block_size=1,
            immutable_weights=False,
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
                min_block_size=1,
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
                        prefer_deferred_runtime_asserts_over_guards=True,
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


class TestAliasedIOCacheRoundTrip(TestCase):
    """``aliased_io`` must survive the cache, or a cache hit changes output arity.

    When TensorRT writes a buffer in place it appends the aliased output as a network
    output, and ``TorchTensorRTModule.forward`` uses ``aliased_io`` to hide that
    plumbing again. If the cache drops the map, the guard is skipped and the internal
    tensor leaks into the caller's return value.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cache_hit_keeps_aliased_io_and_output_arity(self):
        """End to end: the second (cache-hit) compile must match the first."""

        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        args = (
            torch.zeros(2, 4, 16, 8, device="cuda"),
            torch.ones(2, 4, 1, 8, device="cuda") * 7.0,
        )
        ep = torch.export.export(M().cuda(), args)

        engine_cache_dir = "/tmp/test_aliased_io_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def _compile():
            return torch_trt.dynamo.compile(
                ep,
                inputs=list(args),
                enabled_precisions={torch.float32},
                min_block_size=1,
                # Required, not incidental: both cache paths are gated on
                # `not settings.immutable_weights` and the default is immutable, so
                # without this nothing is stored, nothing is pulled, and the second
                # compile below is a second fresh build. The test would then pass
                # whether or not aliased_io survives the cache.
                immutable_weights=False,
                cache_built_engines=True,
                reuse_cached_engines=True,
                engine_cache_dir=engine_cache_dir,
            )

        def _aliased_io(compiled):
            for _name, mod in compiled.named_modules():
                if hasattr(mod, "aliased_io") and mod.aliased_io:
                    return dict(mod.aliased_io)
            return {}

        fresh = _compile()
        fresh_aliased = _aliased_io(fresh)
        # Guard the guard: if the fresh build records nothing there is nothing to lose.
        self.assertEqual(len(fresh_aliased), 1)
        # And prove the cache was actually populated. Without this, a configuration
        # that silently disables caching turns the second compile into another fresh
        # build and the assertions below stop testing anything.
        self.assertTrue(
            os.path.isdir(engine_cache_dir) and os.listdir(engine_cache_dir),
            msg=f"first compile wrote no engine cache entry to {engine_cache_dir}",
        )
        fresh_out = fresh(*[a.clone() for a in args])

        cached = _compile()
        self.assertEqual(_aliased_io(cached), fresh_aliased)

        cached_out = cached(*[a.clone() for a in args])
        n_fresh = len(fresh_out) if isinstance(fresh_out, tuple) else 1
        n_cached = len(cached_out) if isinstance(cached_out, tuple) else 1
        self.assertEqual(n_cached, n_fresh)
