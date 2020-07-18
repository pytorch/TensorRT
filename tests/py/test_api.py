import unittest
import trtorch
import torch
import torchvision.models as models


class ModelTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest', model=None):
        super(ModelTestCase, self).__init__(methodName)
        self.model = model
        self.model.eval().to("cuda")

    @staticmethod
    def parametrize(testcase_class, model=None):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_class(name, model=model))
        return suite

class TestCompile(ModelTestCase):
    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        extra_info = {
            "input_shapes": [self.input.shape],
        }

        trt_mod = trtorch.compile(self.traced_model, extra_info)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

    def test_compile_script(self):
        extra_info = {
            "input_shapes": [self.input.shape],
        }

        trt_mod = trtorch.compile(self.scripted_model, extra_info)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

class TestCheckMethodOpSupport(unittest.TestCase):
    def setUp(self):
        module = models.alexnet(pretrained=True).eval().to("cuda")
        self.module = torch.jit.trace(module, torch.ones((1, 3, 224, 224)).to("cuda"))

    def test_check_support(self):
        self.assertTrue(trtorch.check_method_op_support(self.module, "forward"))

class TestLoggingAPIs(unittest.TestCase):
    def test_logging_prefix(self):
        new_prefix = "TEST"
        trtorch.logging.set_logging_prefix(new_prefix)
        logging_prefix = trtorch.logging.get_logging_prefix()
        self.assertEqual(new_prefix, logging_prefix)

    def test_reportable_log_level(self):
        new_level = trtorch.logging.Level.Warning
        trtorch.logging.set_reportable_log_level(new_level)
        level = trtorch.logging.get_reportable_log_level()
        self.assertEqual(new_level, level)

    def test_is_colored_output_on(self):
        trtorch.logging.set_is_colored_output_on(True)
        color = trtorch.logging.get_is_colored_output_on()
        self.assertTrue(color)

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True)))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet50(pretrained=True)))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.mobilenet_v2(pretrained=True)))
    suite.addTest(unittest.makeSuite(TestCheckMethodOpSupport))
    suite.addTest(unittest.makeSuite(TestLoggingAPIs))

    return suite

suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))