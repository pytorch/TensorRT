import unittest
import trtorch
import torch
import torchvision.models as models

from multi_gpu_test_case import MultiGpuTestCase

gpu_id = 1
class TestCompile(MultiGpuTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": gpu_id,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_mod = trtorch.compile(self.traced_model, compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

    def test_compile_script(self):
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": gpu_id,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_mod = trtorch.compile(self.scripted_model, compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)



def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True)))

    return suite

if not torch.cuda.device_count() > 1:
    raise ValueError("This test case is applicable for multi-gpu configurations only")
        
# Setting it up here so that all CUDA allocations are done on correct device
trtorch.set_device(gpu_id)
suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
