import importlib
import os
import pickle
import shutil
import unittest

import torch
import torch_tensorrt as torch_trt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._refit import refit_module_weights
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


class TestWeightStrippedEngine(TestCase):
    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_three_ways_to_compile(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(pyt_model, example_inputs)

        settings = {
            "use_python_runtime": False,
            "enabled_precisions": {torch.float},
            "debug": False,
            "min_block_size": 1,
            "immutable_weights": False,
            "strip_engine_weights": False,
            "refit_identical_engine_weights": False,
        }

        # 1. Compile with torch_trt.dynamo.compile
        gm1 = torch_trt.dynamo.compile(
            exp_program,
            example_inputs,
            **settings,
        )
        gm1_output = gm1(*example_inputs)

        # 2. Compile with torch.compile using tensorrt backend
        gm2 = torch.compile(
            pyt_model,
            backend="tensorrt",
            options=settings,
        )
        gm2_output = gm2(*example_inputs)

        pyt_model_output = pyt_model(*example_inputs)

        assert torch.allclose(
            pyt_model_output, gm1_output, 1e-2, 1e-2
        ), "gm1_output is not correct"

        assert torch.allclose(
            gm1_output, gm2_output, 1e-2, 1e-2
        ), "gm2_output is not correct"

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_three_ways_to_compile_weight_stripped_engine(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)

        settings = {
            "use_python_runtime": False,
            "enabled_precisions": {torch.float},
            "debug": False,
            "min_block_size": 1,
            "immutable_weights": False,
            "strip_engine_weights": True,
            "refit_identical_engine_weights": False,
        }

        # 1. Compile with torch_trt.compile using dynamo backend
        gm1 = torch_trt.compile(
            pyt_model, ir="dynamo", inputs=example_inputs, **settings
        )
        gm1_output = gm1(*example_inputs)

        # 2. Compile with torch.compile using tensorrt backend, which is not supported to set strip_engine_weights=True
        # gm2 = torch.compile(
        #     pyt_model,
        #     backend="tensorrt",
        #     options=settings,
        # )
        # gm2_output = gm2(*example_inputs)

        assertions.assertEqual(
            gm1_output.sum(), 0, msg="gm1_output should be all zeros"
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_weight_stripped_engine_sizes(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(pyt_model, example_inputs)
        weight_included_engine = convert_exported_program_to_serialized_trt_engine(
            exp_program,
            example_inputs,
            immutable_weights=False,
            strip_engine_weights=False,
            refit_identical_engine_weights=False,
        )
        weight_stripped_engine = convert_exported_program_to_serialized_trt_engine(
            exp_program,
            example_inputs,
            immutable_weights=False,
            strip_engine_weights=True,
            refit_identical_engine_weights=False,
        )
        weight_stripped_refit_identical_engine = (
            convert_exported_program_to_serialized_trt_engine(
                exp_program,
                example_inputs,
                immutable_weights=False,
                strip_engine_weights=True,
                refit_identical_engine_weights=True,
            )
        )
        assertions.assertTrue(
            len(bytes(weight_included_engine)) > len(bytes(weight_stripped_engine)),
            msg=f"Weight-stripped engine size is not smaller than the weight included engine size. Weight included engine size: {len(bytes(weight_included_engine))}, weight-stripped engine size: {len(bytes(weight_stripped_engine))}",
        )
        assertions.assertTrue(
            len(bytes(weight_included_engine))
            > len(bytes(weight_stripped_refit_identical_engine)),
            msg=f"Weight-stripped refit-identical engine size is not smaller than the weight included engine size. Weight included engine size: {len(bytes(weight_included_engine))}, weight-stripped refit-identical engine size: {len(bytes(weight_stripped_refit_identical_engine))}",
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_weight_stripped_engine_results(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]

        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(inputs),
            use_python_runtime=True,
            enabled_precisions={torch.float},
            debug=False,
            min_block_size=1,
            immutable_weights=False,
            strip_engine_weights=True,
            refit_identical_engine_weights=False,
        )
        output = trt_gm(*inputs)
        assertions.assertEqual(
            output.sum(), 0, msg="weight-stripped engine results should be all zeros"
        )

        # Refit the weight-stripped engine with the same weights
        refitted_trt_gm = refit_module_weights(trt_gm, exp_program)
        refitted_output = refitted_trt_gm(*inputs)
        assertions.assertNotEqual(
            refitted_output.sum(),
            0,
            msg="refitted engine results should not be all zeros",
        )

        compiled_model = torch.compile(
            pyt_model,
            backend="tensorrt",
            options={
                "use_python_runtime": False,
                "enabled_precisions": {torch.float},
                "debug": False,
                "min_block_size": 1,
                "immutable_weights": False,
                "cache_built_engines": False,
                "reuse_cached_engines": False,
                "refit_identical_engine_weights": False,
                "strip_engine_weights": False,
            },
        )
        compiled_model_output = compiled_model(*inputs)
        cos_sim = cosine_similarity(refitted_output, compiled_model_output)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"refitted_output doesn't match with compiled_model_output. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    @unittest.skip(
        "For now, torch-trt will save weighted engine if strip_engine_weights is False. In the near future, we plan to save weight-stripped engine regardless of strip_engine_weights, which is pending on TRT's feature development: NVBug #4914602"
    )
    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_engine_caching_saves_weight_stripped_engine(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(pyt_model, example_inputs)

        engine_cache_dir = "/tmp/test_engine_caching_saves_weight_stripped_engine"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        weight_included_engine = convert_exported_program_to_serialized_trt_engine(
            exp_program,
            example_inputs,
            strip_engine_weights=False,
            refit_identical_engine_weights=False,
        )

        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(example_inputs),
            use_python_runtime=True,
            enabled_precisions={torch.float},
            debug=False,
            min_block_size=1,
            immutable_weights=False,
            strip_engine_weights=False,
            refit_identical_engine_weights=True,
            cache_built_engines=True,
            reuse_cached_engines=True,
            engine_cache_dir=engine_cache_dir,
        )
        output = trt_gm(*example_inputs)
        assertions.assertNotEqual(output.sum(), 0, msg="results shouldn't be all zeros")

        blob_path = os.path.join(
            engine_cache_dir, os.listdir(engine_cache_dir)[0], "blob.bin"
        )
        with open(blob_path, "rb") as f:
            blob = f.read()
        unpacked = pickle.loads(blob)
        cached_stripped_engine = unpacked["serialized_engine"]

        assertions.assertTrue(
            len(bytes(weight_included_engine)) > len(bytes(cached_stripped_engine)),
            msg=f"cached engine size is not smaller than the weight included engine size. Weight included engine size: {len(bytes(weight_included_engine))}, cached stripped engine size: {len(bytes(cached_stripped_engine))}",
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_dynamo_compile_with_refittable_weight_stripped_engine(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(pyt_model, args=example_inputs)

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
                immutable_weights=False,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
                strip_engine_weights=False,
                refit_identical_engine_weights=False,
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            results.append(trt_gm(*inputs))

        assertions.assertNotEqual(
            results[0].sum(), 0, msg="results[0] shouldn't be all zeros"
        )
        assertions.assertNotEqual(
            results[1].sum(), 0, msg="results[1] shouldn't be all zeros"
        )
        assertions.assertNotEqual(
            results[2].sum(), 0, msg="results[2] shouldn't be all zeros"
        )

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
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_torch_compile_with_refittable_weight_stripped_engine(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = (
            "/tmp/test_torch_compile_with_refittable_weight_stripped_engine"
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
            compiled_model = torch.compile(
                pyt_model,
                backend="tensorrt",
                options={
                    "use_python_runtime": False,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "engine_cache_dir": engine_cache_dir,
                    "strip_engine_weights": False,
                    "refit_identical_engine_weights": True,
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        assertions.assertNotEqual(
            results[0].sum(), 0, msg="results[0] shouldn't be all zeros"
        )
        assertions.assertNotEqual(
            results[1].sum(), 0, msg="results[1] shouldn't be all zeros"
        )
        assertions.assertNotEqual(
            results[2].sum(), 0, msg="results[2] shouldn't be all zeros"
        )

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
    def test_different_args_dont_share_cached_engine(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = self.conv(x)
                out = self.relu(out)
                return out

        pyt_model = MyModel().eval().to("cuda")

        engine_cache_dir = "/tmp/test_different_args_dont_share_cached_engine"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        inputs = [torch.rand((4, 3, 32, 32)).to("cuda")]

        for i in range(2):
            if i == 0:
                strip_engine_weights = False
            else:
                strip_engine_weights = True

            compiled_model = torch.compile(
                pyt_model,
                backend="tensorrt",
                options={
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "immutable_weights": False,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "engine_cache_dir": engine_cache_dir,
                    "strip_engine_weights": strip_engine_weights,
                },
            )
            compiled_model(*inputs)

        assertions.assertEqual(
            len(os.listdir(engine_cache_dir)),
            2,
            msg=f"It has {len(os.listdir(engine_cache_dir))} cached engine(s) but should have 2 engines",
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    def test_constant_mul_in_refitting(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.tensor(0.5, requires_grad=False)

            def forward(self, x):
                out = x * self.weight
                return out

        pyt_model = MyModel().eval().cuda()
        inputs = [torch.randn((1, 3, 4, 4)).to("cuda")]

        exp_program = torch.export.export(pyt_model, args=tuple(inputs))

        trt_module = torch_trt.compile(
            pyt_model,
            ir="dynamo",
            inputs=tuple(inputs),
            min_block_size=1,
            immutable_weights=False,
            use_python_runtime=True,
            strip_engine_weights=True,
            refit_identical_engine_weights=False,
        )

        refitted_trt_gm = refit_module_weights(trt_module, exp_program)

        outputs_pyt = pyt_model(*inputs)
        outputs_trt = refitted_trt_gm(*inputs)

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_two_TRTRuntime_in_refitting(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]

        pyt_results = pyt_model(*inputs)

        for i in range(2):
            if i == 0:
                use_python_runtime = True
            else:
                use_python_runtime = False

            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=use_python_runtime,
                debug=False,
                min_block_size=1,
                immutable_weights=False,
                strip_engine_weights=True,
                refit_identical_engine_weights=False,
            )

            output = trt_gm(*inputs)
            assertions.assertEqual(output.sum(), 0, msg="results should be all zeros")

            refitted_trt_gm = refit_module_weights(trt_gm, exp_program)
            refitted_output = refitted_trt_gm(*inputs)
            cos_sim = cosine_similarity(pyt_results, refitted_output)
            assertions.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                msg=f"{'PythonTorchTensorRTModule' if use_python_runtime else 'TorchTensorRTModule'} outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
            )

    @unittest.skip("Waiting for implementation")
    @unittest.skipIf(
        not torch_trt.ENABLED_FEATURES.refit,
        "Engine caching requires refit feature that is not supported in Python 3.13 or higher",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"),
        "torchvision is not installed",
    )
    def test_refit_identical_engine_weights(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        exp_program = torch.export.export(pyt_model, example_inputs)

        engine_cache_dir = "/tmp/test_refit_identical_engine_weights"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(example_inputs),
            use_python_runtime=True,
            enabled_precisions={torch.float},
            debug=False,
            min_block_size=1,
            immutable_weights=False,
            strip_engine_weights=True,
            refit_identical_engine_weights=True,
        )
        output = trt_gm(*example_inputs)

        pyt_model2 = models.resnet18(pretrained=False).eval().to("cuda")
        exp_program2 = torch.export.export(pyt_model2, example_inputs)

        try:
            refit_module_weights(trt_gm, exp_program)
        except Exception as e:
            assertions.fail(
                f"Refitting the engine with the same weights failed with the following error: {e}"
            )

        try:
            refit_module_weights(trt_gm, exp_program2)
            assertions.fail(
                "Refitting the engine with different weights should have failed but it didn't"
            )
        except Exception as e:
            pass
