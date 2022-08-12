import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models

from model_test_case import ModelTestCase


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
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        torchtrt.set_device(0)
        self.assertTrue(same < 2e-3)

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
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        torchtrt.set_device(0)
        self.assertTrue(same < 2e-3)


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
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

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
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


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


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
