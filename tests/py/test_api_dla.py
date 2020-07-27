import unittest
import trtorch
import torch
import torchvision.models as models


class ModelTestCaseOnDLA(unittest.TestCase):
    def __init__(self, methodName='runTest', model=None):
        super(ModelTestCaseOnDLA, self).__init__(methodName)
        self.model = model.half()
        self.model.eval().to("cuda").half()

    @staticmethod
    def parametrize(testcase_class, model=None):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_class(name, model=model))
        return suite

class TestCompile(ModelTestCaseOnDLA):
    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda").half()
        self.traced_model = torch.jit.trace(self.model, [self.input]).half()
        self.scripted_model = torch.jit.script(self.model).half()

    def test_compile_traced(self):
        extra_info = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.DLA,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True
                },
            "op_precision": torch.half
        }

        trt_mod = trtorch.compile(self.traced_model, extra_info)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_script(self):
        extra_info = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.DLA, 
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True
                },
            "op_precision": torch.half
        }

        trt_mod = trtorch.compile(self.scripted_model, extra_info)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True)))

    return suite

suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
