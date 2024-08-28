# type: ignore
import os
import shutil
import unittest
from typing import Optional

import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import ENGINE_CACHE_DIR
from torch_tensorrt.dynamo._engine_caching import BaseEngineCache
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_dir: str,
    ) -> None:
        self.engine_cache_dir = engine_cache_dir

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

    def test_dynamo_compile(self):
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
                use_python_runtime=False,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                make_refitable=True,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_size=1 << 30,  # 1GB
            )
            results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_dynamo_compile TRT without engine caching doesn't match with that with engine caching. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_dynamo_compile TRT with engine caching doesn't match with that cached engine. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_torch_compile(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/your_dir"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        engine_cache = MyEngineCache(engine_cache_dir)
        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        results = []
        for i in range(3):
            # remove timing cache and reset dynamo for engine caching messurement
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

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
                    "custom_engine_cache": engine_cache,  # use custom engine cache
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_torch_compile TRT without engine caching doesn't match with that with engine caching. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_torch_compile TRT with engine caching doesn't match with that cached engine. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )
