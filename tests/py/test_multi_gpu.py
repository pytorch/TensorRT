import unittest
import trtorch
import torch
import torchvision.models as models

from model_test_case import ModelTestCase

class TestMultiGpuSwitching(ModelTestCase):
    def setUp(self):
        if torch.cuda.device_count() < 2:
            self.fail("Test is not relevant for this platform since number of available CUDA devices is less than 2")

        trtorch.set_device(0)
        self.target_gpu = 1
        self.input = torch.randn((1, 3, 224, 224)).to("cuda:1")
        self.model = self.model.to("cuda:1")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        trtorch.set_device(0)
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_mod = trtorch.compile(self.traced_model, compile_spec)
        trtorch.set_device(self.target_gpu)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        trtorch.set_device(0)
        self.assertTrue(same < 2e-3)

    def test_compile_script(self):
        trtorch.set_device(0)
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": self.target_gpu,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_mod = trtorch.compile(self.scripted_model, compile_spec)
        trtorch.set_device(self.target_gpu)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        trtorch.set_device(0)
        self.assertTrue(same < 2e-3)

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestMultiGpuSwitching.parametrize(TestMultiGpuSwitching, model=models.resnet18(pretrained=True)))

    return suite

suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
