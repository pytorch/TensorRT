import unittest
import trtorch
import torch
import torchvision.models as models

from model_test_case import ModelTestCase


class TestToBackendLowering(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 300, 300)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.spec = {
            "forward":
                trtorch.TensorRTCompileSpec({
                    "input_shapes": [[1, 3, 300, 300]],
                    "op_precision": torch.float,
                    "refit": False,
                    "debug": False,
                    "strict_types": False,
                    "device": {
                        "device_type": trtorch.DeviceType.GPU,
                        "gpu_id": 0,
                        "allow_gpu_fallback": True
                    },
                    "capability": trtorch.EngineCapability.default,
                    "num_min_timing_iters": 2,
                    "num_avg_timing_iters": 1,
                    "max_batch_size": 0,
                })
        }

    def test_to_backend_lowering(self):
        trt_mod = torch._C._jit_to_backend("tensorrt", self.scripted_model, self.spec)
        same = (trt_mod.forward(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestToBackendLowering.parametrize(TestToBackendLowering, model=models.resnet18(pretrained=True)))

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
