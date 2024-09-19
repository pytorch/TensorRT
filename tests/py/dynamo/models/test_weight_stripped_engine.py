import os
import shutil
import unittest

import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class TestEngineCache(TestCase):
    def test_weight_stripped_engine(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        engine_cache_dir = "/tmp/test_weight_stripped_engine"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]
        results = []

        # run pytorch model
        results.append(model(*inputs))

        remove_timing_cache()
        torch._dynamo.reset()

        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(inputs),
            use_python_runtime=True,
            enabled_precisions={torch.float},
            debug=False,
            min_block_size=1,
            make_refittable=True,
            refit_identical_engine_weights=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
            engine_cache_dir=engine_cache_dir,
            strip_engine_weights=True,
        )
        results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])

        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_dynamo_compile_with_refittable_weight_stripped_engine(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        engine_cache_dir = (
            "/tmp/test_dynamo_compile_with_refittable_weight_stripped_engine"
        )
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
                debug=False,
                min_block_size=1,
                make_refittable=True,
                refit_identical_engine_weights=True,
                strip_engine_weights=True,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            results.append(trt_gm(*inputs))

        assertions.assertNotEqual(results[0].sum(), 0, msg="results[0] are all zeros")
        assertions.assertNotEqual(results[1].sum(), 0, msg="results[1] are all zeros")
        assertions.assertNotEqual(results[2].sum(), 0, msg="results[2] are all zeros")

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
