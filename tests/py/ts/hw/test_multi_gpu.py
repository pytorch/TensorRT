import importlib.util
import unittest

import torch
import torch_tensorrt as torchtrt
from model_test_case import ModelTestCase

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
class TestMultiGpuSwitching(ModelTestCase):
    def setUp(self):
        if torch.cuda.device_count() < 2:
            self.fail(
                "Test is not relevant for this platform since number of available CUDA devices is less than 2"
            )

        torchtrt.set_device(0)
        self.target_gpu = 1
        self.input = torch.randn((1, 3, 224, 224)).to("cuda:1")
        self.model = self.model.to("cuda:1")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        torchtrt.set_device(0)
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        torchtrt.set_device(self.target_gpu)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        torchtrt.set_device(0)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TestMultiGpuSwitching traced TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_compile_script(self):
        torchtrt.set_device(0)
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        torchtrt.set_device(self.target_gpu)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        torchtrt.set_device(0)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TestMultiGpuSwitching scripted TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
class TestMultiGpuSerializeDeserializeSwitching(ModelTestCase):
    def setUp(self):
        if torch.cuda.device_count() < 2:
            self.fail(
                "Test is not relevant for this platform since number of available CUDA devices is less than 2"
            )

        self.target_gpu = 0
        torchtrt.set_device(0)
        self.input = torch.randn((1, 3, 224, 224)).to("cuda:0")
        self.model = self.model.to("cuda:0")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        torchtrt.set_device(0)
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        # Changing the device ID deliberately. It should still run on correct device ID by context switching
        torchtrt.set_device(1)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TestMultiGpuSerializeDeserializeSwitching traced TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_compile_script(self):
        torchtrt.set_device(0)
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        # Changing the device ID deliberately. It should still run on correct device ID by context switching
        torchtrt.set_device(1)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TestMultiGpuSerializeDeserializeSwitching scripted TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(
        TestMultiGpuSwitching.parametrize(
            TestMultiGpuSwitching, model=models.resnet18(pretrained=True)
        )
    )
    suite.addTest(
        TestMultiGpuSerializeDeserializeSwitching.parametrize(
            TestMultiGpuSwitching, model=models.resnet18(pretrained=True)
        )
    )

    return suite


if importlib.util.find_spec("torchvision"):
    suite = test_suite()

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    exit(int(not result.wasSuccessful()))
