import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict

from model_test_case import ModelTestCase


class TestCompile(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_traced(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape, dtype=torch.float, format=torch.contiguous_format)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float}
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_script(self):
        trt_mod = torchtrt.ts.compile(self.scripted_model,
                                  inputs=[self.input],
                                  device=torchtrt.Device(gpu_id=0),
                                  enabled_precisions={torch.float})
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)


    def test_compile_global(self):
        trt_mod = torchtrt.compile(self.scripted_model,
                                  inputs=[self.input],
                                  device=torchtrt.Device(gpu_id=0),
                                  enabled_precisions={torch.float})
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_global_nn_mod(self):
        trt_mod = torchtrt.compile(self.model,
                                  inputs=[self.input],
                                  device=torchtrt.Device(gpu_id=0),
                                  enabled_precisions={torch.float})
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_from_torch_tensor(self):
        compile_spec = {
            "inputs": [self.input],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float}
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_device(self):
        compile_spec = {"inputs": [self.input], "device": torchtrt.Device("gpu:0"), "enabled_precisions": {torch.float}}

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_default_device(self):
        compile_spec = {"inputs": [self.input], "enabled_precisions": {torch.float}}

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)

    def test_compile_script_from_dict(self):
        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float}
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        same = (trt_mod(self.input) - self.traced_model(self.input)).abs().max()
        self.assertTrue(same < 2e-2)


class TestCompileHalf(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.scripted_model.half()

    def test_compile_script_half(self):
        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape, dtype=torch.half)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.half}
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input.half()) - self.scripted_model(self.input.half())).abs().max()
        torchtrt.logging.log(torchtrt.logging.Level.Debug, "Max diff: " + str(same))
        self.assertTrue(same < 3e-2)


class TestCompileHalfDefault(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.scripted_model.half()

    def test_compile_script_half_by_default(self):
        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float, torch.half}
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input.half()) - self.scripted_model(self.input.half())).abs().max()
        torchtrt.logging.log(torchtrt.logging.Level.Debug, "Max diff: " + str(same))
        self.assertTrue(same < 3e-2)


class TestFallbackToTorch(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_script(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            },
            "require_full_compilation": False,
            "torch_executed_ops": ["aten::max_pool2d"],
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


class TestModuleFallbackToTorch(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)

    def test_compile_script(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            },
            "require_full_compilation": False,
            "torch_executed_modules": ["torchvision.models.resnet.BasicBlock"],
            "min_block_size": 1
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


class TestPTtoTRTtoPT(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.ts_model = torch.jit.trace(self.model, [self.input])

    def test_pt_to_trt_to_pt(self):
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False
            }
        }

        trt_engine = torchtrt.ts.convert_method_to_trt_engine(self.ts_model, "forward", **compile_spec)
        trt_mod = torchtrt.ts.embed_engine_in_new_module(trt_engine, torchtrt.Device("cuda:0"))
        same = (trt_mod(self.input) - self.ts_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


class TestInputTypeDefaultsFP32Model(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_input_use_default_fp32(self):
        ts_model = torch.jit.script(self.model)
        trt_mod = torchtrt.ts.compile(ts_model,
                                  inputs=[torchtrt.Input(self.input.shape)],
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input)

    def test_input_respect_user_setting_fp32_weights_fp16_in(self):
        ts_model = torch.jit.script(self.model)
        trt_mod = torchtrt.ts.compile(ts_model,
                                  inputs=[self.input.half()],
                                  require_full_compilation=True,
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input.half())

    def test_input_respect_user_setting_fp32_weights_fp16_in_non_constructor(self):
        ts_model = torch.jit.script(self.model)
        input_spec = torchtrt.Input(self.input.shape)
        input_spec.dtype = torch.half

        trt_mod = torchtrt.ts.compile(ts_model,
                                  inputs=[input_spec],
                                  require_full_compilation=True,
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input.half())


class TestInputTypeDefaultsFP16Model(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

    def test_input_use_default_fp16(self):
        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(half_mod,
                                  inputs=[torchtrt.Input(self.input.shape)],
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input.half())

    def test_input_use_default_fp16_without_fp16_enabled(self):
        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(half_mod, inputs=[torchtrt.Input(self.input.shape)])
        trt_mod(self.input.half())

    def test_input_respect_user_setting_fp16_weights_fp32_in(self):
        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(half_mod,
                                  inputs=[self.input],
                                  require_full_compilation=True,
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input)

    def test_input_respect_user_setting_fp16_weights_fp32_in_non_constuctor(self):
        half_mod = torch.jit.script(self.model)
        half_mod.half()

        input_spec = torchtrt.Input(self.input.shape)
        input_spec.dtype = torch.float

        trt_mod = torchtrt.ts.compile(half_mod,
                                  inputs=[input_spec],
                                  require_full_compilation=True,
                                  enabled_precisions={torch.float, torch.half})
        trt_mod(self.input)


class TestCheckMethodOpSupport(unittest.TestCase):

    def setUp(self):
        module = models.alexnet(pretrained=True).eval().to("cuda")
        self.module = torch.jit.trace(module, torch.ones((1, 3, 224, 224)).to("cuda"))

    def test_check_support(self):
        self.assertTrue(torchtrt.ts.check_method_op_support(self.module, "forward"))


class TestLoggingAPIs(unittest.TestCase):

    def test_logging_prefix(self):
        new_prefix = "Python API Test: "
        torchtrt.logging.set_logging_prefix(new_prefix)
        logging_prefix = torchtrt.logging.get_logging_prefix()
        self.assertEqual(new_prefix, logging_prefix)

    def test_reportable_log_level(self):
        new_level = torchtrt.logging.Level.Error
        torchtrt.logging.set_reportable_log_level(new_level)
        level = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(new_level, level)

    def test_is_colored_output_on(self):
        torchtrt.logging.set_is_colored_output_on(True)
        color = torchtrt.logging.get_is_colored_output_on()
        self.assertTrue(color)


class TestDevice(unittest.TestCase):

    def test_from_string_constructor(self):
        device = torchtrt.Device("cuda:0")
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 0)

        device = torchtrt.Device("gpu:1")
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 1)

    def test_from_string_constructor_dla(self):
        device = torchtrt.Device("dla:0")
        self.assertEqual(device.device_type, torchtrt.DeviceType.DLA)
        self.assertEqual(device.gpu_id, 0)
        self.assertEqual(device.dla_core, 0)

        device = torchtrt.Device("dla:1", allow_gpu_fallback=True)
        self.assertEqual(device.device_type, torchtrt.DeviceType.DLA)
        self.assertEqual(device.gpu_id, 0)
        self.assertEqual(device.dla_core, 1)
        self.assertEqual(device.allow_gpu_fallback, True)

    def test_kwargs_gpu(self):
        device = torchtrt.Device(gpu_id=0)
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 0)

    def test_kwargs_dla_and_settings(self):
        device = torchtrt.Device(dla_core=1, allow_gpu_fallback=False)
        self.assertEqual(device.device_type, torchtrt.DeviceType.DLA)
        self.assertEqual(device.gpu_id, 0)
        self.assertEqual(device.dla_core, 1)
        self.assertEqual(device.allow_gpu_fallback, False)

        device = torchtrt.Device(gpu_id=1, dla_core=0, allow_gpu_fallback=True)
        self.assertEqual(device.device_type, torchtrt.DeviceType.DLA)
        self.assertEqual(device.gpu_id, 1)
        self.assertEqual(device.dla_core, 0)
        self.assertEqual(device.allow_gpu_fallback, True)

    def test_from_torch(self):
        device = torchtrt.Device._from_torch_device(torch.device("cuda:0"))
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 0)

class TestInput(unittest.TestCase):

    def _verify_correctness(self, struct: torchtrt.Input, target: Dict) -> bool:
        internal = struct._to_internal()

        list_eq = lambda al, bl: all([a == b for (a, b) in zip (al, bl)])

        eq = lambda a, b : a == b

        def field_is_correct(field, equal_fn, a1, a2):
            equal = equal_fn(a1, a2)
            if not equal:
                print("\nField {} is incorrect: {} != {}".format(field, a1, a2))
            return equal

        min_ = field_is_correct("min", list_eq, internal.min, target["min"])
        opt_ = field_is_correct("opt", list_eq, internal.opt, target["opt"])
        max_ = field_is_correct("max", list_eq, internal.max, target["max"])
        is_dynamic_ = field_is_correct("is_dynamic", eq, internal.input_is_dynamic, target["input_is_dynamic"])
        explicit_set_dtype_ = field_is_correct("explicit_dtype", eq, internal._explicit_set_dtype, target["explicit_set_dtype"])
        dtype_ = field_is_correct("dtype", eq, int(internal.dtype), int(target["dtype"]))
        format_ = field_is_correct("format", eq, int(internal.format), int(target["format"]))

        return all([min_,opt_,max_,is_dynamic_,explicit_set_dtype_,dtype_,format_])


    def test_infer_from_example_tensor(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.half,
            "format": torchtrt.TensorFormat.contiguous,
            "explicit_set_dtype": True
        }

        example_tensor = torch.randn(shape).half()
        i = torchtrt.Input._from_tensor(example_tensor)
        self.assertTrue(self._verify_correctness(i, target))


    def test_static_shape(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.TensorFormat.contiguous,
            "explicit_set_dtype": False
        }

        i = torchtrt.Input(shape)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(tuple(shape))
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(torch.randn(shape).shape)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(shape=shape)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(shape=tuple(shape))
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(shape=torch.randn(shape).shape)
        self.assertTrue(self._verify_correctness(i, target))

    def test_data_type(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.half,
            "format": torchtrt.TensorFormat.contiguous,
            "explicit_set_dtype": True
        }

        i = torchtrt.Input(shape, dtype=torchtrt.dtype.half)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(shape, dtype=torch.half)
        self.assertTrue(self._verify_correctness(i, target))

    def test_tensor_format(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.TensorFormat.channels_last,
            "explicit_set_dtype": False
        }

        i = torchtrt.Input(shape, format=torchtrt.TensorFormat.channels_last)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(shape, format=torch.channels_last)
        self.assertTrue(self._verify_correctness(i, target))

    def test_dynamic_shape(self):
        min_shape = [1, 3, 128, 128]
        opt_shape = [1, 3, 256, 256]
        max_shape = [1, 3, 512, 512]
        target = {
            "min": min_shape,
            "opt": opt_shape,
            "max": max_shape,
            "input_is_dynamic": True,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.TensorFormat.contiguous,
            "explicit_set_dtype": False
        }

        i = torchtrt.Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape)
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(min_shape=tuple(min_shape), opt_shape=tuple(opt_shape), max_shape=tuple(max_shape))
        self.assertTrue(self._verify_correctness(i, target))

        tensor_shape = lambda shape: torch.randn(shape).shape
        i = torchtrt.Input(min_shape=tensor_shape(min_shape), opt_shape=tensor_shape(opt_shape), max_shape=tensor_shape(max_shape))
        self.assertTrue(self._verify_correctness(i, target))

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLoggingAPIs))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.resnet18(pretrained=True)))
    suite.addTest(TestCompile.parametrize(TestCompile, model=models.mobilenet_v2(pretrained=True)))
    suite.addTest(TestCompileHalf.parametrize(TestCompileHalf, model=models.resnet18(pretrained=True)))
    suite.addTest(TestCompileHalfDefault.parametrize(TestCompileHalfDefault, model=models.resnet18(pretrained=True)))
    suite.addTest(TestPTtoTRTtoPT.parametrize(TestPTtoTRTtoPT, model=models.resnet18(pretrained=True)))
    suite.addTest(
        TestInputTypeDefaultsFP32Model.parametrize(TestInputTypeDefaultsFP32Model,
                                                   model=models.resnet18(pretrained=True)))
    suite.addTest(
        TestInputTypeDefaultsFP16Model.parametrize(TestInputTypeDefaultsFP16Model,
                                                   model=models.resnet18(pretrained=True)))
    suite.addTest(TestFallbackToTorch.parametrize(TestFallbackToTorch, model=models.resnet18(pretrained=True)))
    suite.addTest(
        TestModuleFallbackToTorch.parametrize(TestModuleFallbackToTorch, model=models.resnet18(pretrained=True)))
    suite.addTest(unittest.makeSuite(TestCheckMethodOpSupport))
    suite.addTest(unittest.makeSuite(TestDevice))
    suite.addTest(unittest.makeSuite(TestInput))

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
