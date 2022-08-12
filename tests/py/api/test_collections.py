import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import os


def find_repo_root(max_depth=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(max_depth):
        files = os.listdir(dir_path)
        if "WORKSPACE" in files:
            return dir_path
        else:
            dir_path = os.path.dirname(dir_path)

    raise RuntimeError("Could not find repo root")


MODULE_DIR = find_repo_root() + "/tests/modules"


class TestStandardTensorInput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/standard_tensor_input_scripted.jit.pt")
            .eval()
            .to("cuda")
        )

        compile_spec = {
            "inputs": [
                torchtrt.Input(self.input.shape),
                torchtrt.Input(self.input.shape),
            ],
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (
            (trt_mod(self.input, self.input) - self.model(self.input, self.input))
            .abs()
            .max()
        )
        self.assertTrue(same < 2e-2)


class TestTupleInput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/tuple_input_scripted.jit.pt")
            .eval()
            .to("cuda")
        )

        compile_spec = {
            "input_signature": (
                (torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)),
            ),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (
            (trt_mod((self.input, self.input)) - self.model((self.input, self.input)))
            .abs()
            .max()
        )
        self.assertTrue(same < 2e-2)


class TestListInput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/list_input_scripted.jit.pt").eval().to("cuda")
        )

        compile_spec = {
            "input_signature": (
                [torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],
            ),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        same = (
            (trt_mod([self.input, self.input]) - self.model([self.input, self.input]))
            .abs()
            .max()
        )
        self.assertTrue(same < 2e-2)


class TestTupleInputOutput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/tuple_input_output_scripted.jit.pt")
            .eval()
            .to("cuda")
        )

        compile_spec = {
            "input_signature": (
                (torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)),
            ),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))


class TestListInputOutput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/list_input_output_scripted.jit.pt")
            .eval()
            .to("cuda")
        )

        compile_spec = {
            "input_signature": (
                [torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],
            ),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))


class TestListInputTupleOutput(unittest.TestCase):
    def test_compile(self):

        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.model = (
            torch.jit.load(MODULE_DIR + "/list_input_tuple_output_scripted.jit.pt")
            .eval()
            .to("cuda")
        )

        compile_spec = {
            "input_signature": (
                [torchtrt.Input(self.input.shape), torchtrt.Input(self.input.shape)],
            ),
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        trt_out = trt_mod((self.input, self.input))
        pyt_out = self.model((self.input, self.input))
        results = [(t - p).abs().max() < 2e-2 for (t, p) in zip(trt_out, pyt_out)]
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
