import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict


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

        list_eq = lambda al, bl: all([a == b for (a, b) in zip(al, bl)])

        eq = lambda a, b: a == b

        def field_is_correct(field, equal_fn, a1, a2):
            equal = equal_fn(a1, a2)
            if not equal:
                print("\nField {} is incorrect: {} != {}".format(field, a1, a2))
            return equal

        min_ = field_is_correct("min", list_eq, internal.min, target["min"])
        opt_ = field_is_correct("opt", list_eq, internal.opt, target["opt"])
        max_ = field_is_correct("max", list_eq, internal.max, target["max"])
        is_dynamic_ = field_is_correct(
            "is_dynamic", eq, internal.input_is_dynamic, target["input_is_dynamic"]
        )
        explicit_set_dtype_ = field_is_correct(
            "explicit_dtype",
            eq,
            internal._explicit_set_dtype,
            target["explicit_set_dtype"],
        )
        dtype_ = field_is_correct(
            "dtype", eq, int(internal.dtype), int(target["dtype"])
        )
        format_ = field_is_correct(
            "format", eq, int(internal.format), int(target["format"])
        )

        return all(
            [min_, opt_, max_, is_dynamic_, explicit_set_dtype_, dtype_, format_]
        )

    def test_infer_from_example_tensor(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.half,
            "format": torchtrt.TensorFormat.contiguous,
            "explicit_set_dtype": True,
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
            "explicit_set_dtype": False,
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
            "explicit_set_dtype": True,
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
            "explicit_set_dtype": False,
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
            "explicit_set_dtype": False,
        }

        i = torchtrt.Input(
            min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape
        )
        self.assertTrue(self._verify_correctness(i, target))

        i = torchtrt.Input(
            min_shape=tuple(min_shape),
            opt_shape=tuple(opt_shape),
            max_shape=tuple(max_shape),
        )
        self.assertTrue(self._verify_correctness(i, target))

        tensor_shape = lambda shape: torch.randn(shape).shape
        i = torchtrt.Input(
            min_shape=tensor_shape(min_shape),
            opt_shape=tensor_shape(opt_shape),
            max_shape=tensor_shape(max_shape),
        )
        self.assertTrue(self._verify_correctness(i, target))


if __name__ == "__main__":
    unittest.main()
