import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
from utils import cosine_similarity, COSINE_THRESHOLD


class ModelTestCaseOnDLA(unittest.TestCase):
    def __init__(self, methodName="runTest", model=None):
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
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.DLA,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True,
            },
            "enabled_precisions": {torch.half},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"ModelTestCaseOnDLA traced TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_compile_script(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.DLA,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True,
            },
            "enabled_precisions": {torch.half},
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"ModelTestCaseOnDLA scripted TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(
        TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True))
    )

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
