import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict


class TestCompile(unittest.TestCase):
    def test_compile_traced(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_script(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        with torch.no_grad():
            trt_mod = torchtrt.ts.compile(
                self.scripted_model,
                inputs=[self.input],
                device=torchtrt.Device(gpu_id=0),
                enabled_precisions={torch.float},
            )
            same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
            self.assertTrue(same < 2e-2)

    def test_compile_global(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        trt_mod = torchtrt.compile(
            self.scripted_model,
            inputs=[self.input],
            device=torchtrt.Device(gpu_id=0),
            enabled_precisions={torch.float},
        )
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_global_nn_mod(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        with torch.no_grad():
            trt_mod = torchtrt.compile(
                self.model,
                inputs=[self.input],
                device=torchtrt.Device(gpu_id=0),
                enabled_precisions={torch.float},
            )
            same = (trt_mod(self.input) - self.model(self.input)).abs().max()
            self.assertTrue(same < 2e-2)

    def test_from_torch_tensor(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {
            "inputs": [self.input],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_device(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {
            "inputs": [self.input],
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_default_device(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {"inputs": [self.input], "enabled_precisions": {torch.float}}

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_script_from_dict(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)


class TestPTtoTRTtoPT(unittest.TestCase):
    def test_pt_to_trt_to_pt(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.ts_model = torch.jit.trace(self.model, [self.input])

        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_engine = torchtrt.ts.convert_method_to_trt_engine(
            self.ts_model, "forward", **compile_spec
        )
        trt_mod = torchtrt.ts.embed_engine_in_new_module(
            trt_engine, torchtrt.Device("cuda:0")
        )
        same = (trt_mod(self.input) - self.ts_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


class TestCheckMethodOpSupport(unittest.TestCase):
    def test_check_support(self):
        module = models.alexnet(pretrained=True).eval().to("cuda")
        self.module = torch.jit.trace(module, torch.ones((1, 3, 224, 224)).to("cuda"))

        self.assertTrue(torchtrt.ts.check_method_op_support(self.module, "forward"))


class TestModuleIdentification(unittest.TestCase):
    def test_module_type(self):
        nn_module = models.alexnet(pretrained=True).eval().to("cuda")
        ts_module = torch.jit.trace(nn_module, torch.ones([1, 3, 224, 224]).to("cuda"))
        fx_module = torch.fx.symbolic_trace(nn_module)

        self.assertEqual(
            torchtrt._compile._parse_module_type(nn_module),
            torchtrt._compile._ModuleType.nn,
        )
        self.assertEqual(
            torchtrt._compile._parse_module_type(ts_module),
            torchtrt._compile._ModuleType.ts,
        )
        self.assertEqual(
            torchtrt._compile._parse_module_type(fx_module),
            torchtrt._compile._ModuleType.fx,
        )


if __name__ == "__main__":
    unittest.main()
