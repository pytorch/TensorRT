import unittest
import trtorch
import torch
import torchvision.models as models

from model_test_case import ModelTestCase


class TestCompile(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": 0,
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
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_mod = trtorch.compile(self.scripted_model, compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

class TestPTtoTRTtoPT(ModelTestCase):
    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.ts_model = torch.jit.script(self.model)

    def test_pt_to_trt_to_pt(self):
        compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_engine = trtorch.convert_method_to_trt_engine(self.ts_model, "forward", compile_spec)
        trt_mod = trtorch.embed_engine_in_new_module(trt_engine)
        same = (trt_mod(self.input) - self.ts_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

class TestCheckMethodOpSupport(unittest.TestCase):

    def setUp(self):
        module = models.alexnet(pretrained=True).eval().to("cuda")
        self.module = torch.jit.trace(module, torch.ones((1, 3, 224, 224)).to("cuda"))

    def test_check_support(self):
        self.assertTrue(trtorch.check_method_op_support(self.module, "forward"))


class TestLoggingAPIs(unittest.TestCase):

    def test_logging_prefix(self):
        new_prefix = "Python API Test: "
        trtorch.logging.set_logging_prefix(new_prefix)
        logging_prefix = trtorch.logging.get_logging_prefix()
        self.assertEqual(new_prefix, logging_prefix)

    def test_reportable_log_level(self):
        new_level = trtorch.logging.Level.Error
        trtorch.logging.set_reportable_log_level(new_level)
        level = trtorch.logging.get_reportable_log_level()
        self.assertEqual(new_level, level)

    def test_is_colored_output_on(self):
        trtorch.logging.set_is_colored_output_on(True)
        color = trtorch.logging.get_is_colored_output_on()
        self.assertTrue(color)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLoggingAPIs))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True)))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.mobilenet_v2(pretrained=True)))
    suite.addTest(TestPTtoTRTtoPT.parametrize(TestPTtoTRTtoPT, model=models.mobilenet_v2(pretrained=True)))
    suite.addTest(unittest.makeSuite(TestCheckMethodOpSupport))

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
