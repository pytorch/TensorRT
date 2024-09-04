# type: ignore
import os
import shutil
import unittest
from typing import Optional

import pytest
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import ENGINE_CACHE_DIR
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_dir: str,
    ) -> None:
        self.engine_cache_dir = engine_cache_dir
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

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

    def load(self, hash: str, prefix: str = "blob") -> Optional[bytes]:
        path = os.path.join(self.engine_cache_dir, f"{prefix}_{hash}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                blob = f.read()
            return blob
        return None


class TestEngineCache(TestCase):

    def test_dynamo_compile_with_default_disk_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        engine_cache_dir = ENGINE_CACHE_DIR
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

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
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            start.record()
            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=False,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                make_refitable=True,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
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

    def test_dynamo_compile_with_custom_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/your_dir"
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
            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=False,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                make_refitable=True,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                custom_engine_cache=custom_engine_cache,
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

    def test_torch_compile_with_default_disk_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_compile_with_default_disk_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

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
            # remove timing cache and reset dynamo for engine caching messurement
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
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "make_refitable": True,
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

    def test_torch_compile_with_custom_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/your_dir"
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
            # remove timing cache and reset dynamo for engine caching messurement
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
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "make_refitable": True,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "custom_engine_cache": custom_engine_cache,
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
