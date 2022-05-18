import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models

from model_test_case import ModelTestCase, REPO_ROOT

class TestStandardTensorInput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape),
                      torchtrt.Input(self.input.shape)],
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float}
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (trt_mod(self.input, self.input) - self.model(self.input, self.input)).abs().max()
        self.assertTrue(same < 2e-2)

class TestTupleInput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "input_signature": ((torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)),),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "require_full_compilation": False,
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (trt_mod((self.input, self.input)) - self.model((self.input, self.input))).abs().max()
        self.assertTrue(same < 2e-2)

class TestListInput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "input_signature": ([torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "require_full_compilation": False,
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (trt_mod([self.input, self.input]) - self.model([self.input, self.input])).abs().max()
        self.assertTrue(same < 2e-2)

class TestTupleInputOutput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "input_signature": ((torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)),),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "require_full_compilation": False,
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))

class TestListInputOutput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "input_signature": ([torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "require_full_compilation": False,
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))


class TestListInputTupleOutput(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_compile(self):
        compile_spec = {
            "input_signature": ([torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "require_full_compilation": False,
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestStandardTensorInput.parametrize(TestStandardTensorInput, model=torch.jit.load(REPO_ROOT + "/tests/modules/standard_tensor_input.jit.pt")))
    suite.addTest(TestTupleInput.parametrize(TestTupleInput, model=torch.jit.load(REPO_ROOT + "/tests/modules/tuple_input.jit.pt")))
    suite.addTest(TestListInput.parametrize(TestListInput, model=torch.jit.load(REPO_ROOT + "/tests/modules/list_input.jit.pt")))
    suite.addTest(TestTupleInputOutput.parametrize(TestTupleInputOutput, model=torch.jit.load(REPO_ROOT + "/tests/modules/tuple_input_output.jit.pt")))
    suite.addTest(TestListInputOutput.parametrize(TestListInputOutput, model=torch.jit.load(REPO_ROOT + "/tests/modules/list_input_output.jit.pt")))
    suite.addTest(TestListInputTupleOutput.parametrize(TestListInputTupleOutput, model=torch.jit.load(REPO_ROOT + "/tests/modules/complex_model.jit.pt")))

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
