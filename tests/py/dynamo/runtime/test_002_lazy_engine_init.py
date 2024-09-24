# type: ignore
import os
import tempfile
import unittest

import torch
import torch_tensorrt
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from torch_tensorrt.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

assertions = unittest.TestCase()


def assert_close(outputs, ref_outputs):
    if type(outputs) not in (list, tuple):
        outputs = [outputs]

    if type(ref_outputs) not in (
        list,
        tuple,
        torch.return_types.max,
        torch.return_types.min,
    ):
        ref_outputs = [ref_outputs]

    for out, ref in zip(outputs, ref_outputs):
        if not isinstance(ref, torch.Tensor):
            if len(out.shape) == 0:
                ref = torch.tensor(ref)
            else:
                ref = torch.tensor([ref])
        ref = ref.cpu()  # to_dtype test has cases with gpu output
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-03,
            atol=1e-03,
            equal_nan=True,
            check_dtype=True,
        )


class TestLazyEngineInit(TestCase):

    def test_lazy_engine_init_py(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)), torch.randn((2, 4))

        # Create a model
        model = Test()
        exp_program = torch.export.export(model, (input_data_0, input_data_1))

        # Convert to TensorRT engine
        trt_engine_str = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exp_program, inputs=(input_data_0, input_data_1)
            )
        )

        # Inference on TRT Engine
        trt_module = PythonTorchTensorRTModule(
            trt_engine_str,
            ["a", "b"],
            ["output0"],
            settings=CompilationSettings(lazy_engine_init=True),
        )

        assertions.assertTrue(
            trt_module.engine is None,
            msg="Engine was proactively instantiated even though lazy engine loading was enabled",
        )

        with assertions.assertRaises(Exception):
            trt_output = trt_module(input_data_0, input_data_1).cpu()

        trt_module.setup_engine()
        assertions.assertTrue(trt_module.engine, msg="Engine was not setup")

        trt_output = trt_module(input_data_0, input_data_1).cpu()

        # Inference on PyTorch model
        model_output = model(input_data_0, input_data_1)

        assert_close(trt_output, model_output)

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_lazy_engine_init_cpp(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)), torch.randn((2, 4))

        # Create a model
        model = Test()
        exp_program = torch.export.export(model, (input_data_0, input_data_1))

        # Convert to TensorRT engine
        trt_engine_str = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exp_program, inputs=(input_data_0, input_data_1)
            )
        )

        # Inference on TRT Engine
        trt_module = TorchTensorRTModule(
            trt_engine_str,
            ["a", "b"],
            ["output0"],
            settings=CompilationSettings(lazy_engine_init=True),
        )

        assertions.assertTrue(
            trt_module.engine is None,
            msg="Engine was proactively instantiated even though lazy engine loading was enabled",
        )

        with assertions.assertRaises(Exception):
            trt_output = trt_module(
                input_data_0.to("cuda"), input_data_1.to("cuda")
            ).cpu()

        trt_module.setup_engine()
        assertions.assertTrue(trt_module.engine is not None, msg="Engine was not setup")

        trt_output = trt_module(input_data_0.to("cuda"), input_data_1.to("cuda")).cpu()

        # Inference on PyTorch model
        model_output = model(input_data_0, input_data_1)

        assert_close(trt_output, model_output)

    def test_lazy_engine_init_py_e2e(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    input.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "optimization_level": 1,
            "min_block_size": 1,
            "ir": "dynamo",
            "lazy_engine_init": True,
            "use_python_runtime": True,
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }

        trt_mod = torchtrt.compile(model, **compile_spec)
        cos_sim = cosine_similarity(model(input), trt_mod(input))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        # Clean up model env
        torch._dynamo.reset()

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_lazy_engine_init_cpp_e2e(self):
        model = models.resnet18(pretrained=False).eval().to("cuda")
        input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    input.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "optimization_level": 1,
            "min_block_size": 1,
            "ir": "dynamo",
            "lazy_engine_init": True,
            "use_python_runtime": False,
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }

        trt_mod = torchtrt.compile(model, **compile_spec)
        cos_sim = cosine_similarity(model(input), trt_mod(input))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        # Clean up model env
        torch._dynamo.reset()

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_lazy_engine_init_cpp_serialization(self):
        model = models.resnet18(pretrained=False).eval().to("cuda")
        input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    input.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "optimization_level": 1,
            "min_block_size": 1,
            "ir": "dynamo",
            "lazy_engine_init": True,
            "use_python_runtime": False,
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }

        trt_mod = torchtrt.compile(model, **compile_spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch_tensorrt.save(
                trt_mod, os.path.join(tmpdir, "tmp_trt_mod.ep"), inputs=[input]
            )
            new_trt_mod = torch.export.load(os.path.join(tmpdir, "tmp_trt_mod.ep"))

        loaded_trt_mod = new_trt_mod.module()
        cos_sim = cosine_similarity(model(input), trt_mod(input))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )
        # Clean up model env
        torch._dynamo.reset()

    def test_lazy_engine_init_py_hybrid_graph(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                w = torch.add(a, b)
                x = 2 * b
                y = torch.sub(w, a)
                z = torch.add(y, x)
                return w, x, y, z

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)).to("cuda"), torch.randn(
            (2, 4)
        ).to("cuda")

        # Create a model
        model = Test()
        exp_program = torch.export.export(model, (input_data_0, input_data_1))

        compile_spec = {
            "inputs": (input_data_0, input_data_1),
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "optimization_level": 1,
            "min_block_size": 1,
            "ir": "dynamo",
            "lazy_engine_init": True,
            "use_python_runtime": True,
            "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }

        trt_mod = torchtrt.dynamo.compile(exp_program, **compile_spec)
        assert_close(
            trt_mod(input_data_0, input_data_1), model(input_data_0, input_data_1)
        )

        # Clean up model env
        torch._dynamo.reset()

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_lazy_engine_init_cpp_hybrid_graph(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                x = torch.add(a, b)
                y = torch.sub(x, 2 * b)
                z = torch.add(y, b)
                return z

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)).to("cuda"), torch.randn(
            (2, 4)
        ).to("cuda")

        # Create a model
        model = Test()
        exp_program = torch.export.export(model, (input_data_0, input_data_1))

        compile_spec = {
            "inputs": (input_data_0, input_data_1),
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "optimization_level": 1,
            "min_block_size": 1,
            "ir": "dynamo",
            "lazy_engine_init": True,
            "use_python_runtime": False,
            "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
            "cache_built_engines": False,
            "reuse_cached_engines": False,
        }

        trt_mod = torchtrt.dynamo.compile(exp_program, **compile_spec)
        assert_close(
            trt_mod(input_data_0, input_data_1), model(input_data_0, input_data_1)
        )

        # Clean up model env
        torch._dynamo.reset()
