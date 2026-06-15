import copy
import unittest
from typing import Dict
from unittest import mock

import torch
import torch_tensorrt as torchtrt
import torch_tensorrt._enums as torchtrt_enums
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule


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
        self.assertEqual(
            device.gpu_id, 0
        )  # Override since AGX platforms use iGPU to manage DLA
        self.assertEqual(device.dla_core, 0)
        self.assertEqual(device.allow_gpu_fallback, True)

    def test_from_torch(self):
        device = torchtrt.Device._from_torch_device(torch.device("cuda:0"))
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 0)


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestPlatform(unittest.TestCase):
    def test_current_platform(self):
        py_plat_str = torchtrt.Platform.current_platform()._to_serialized_rt_platform()
        cpp_plat_str = torch.ops.tensorrt.get_current_platform()
        self.assertEqual(py_plat_str, cpp_plat_str)

    def test_current_platform_prefers_loaded_runtime_platform(self):
        with mock.patch("platform.system", return_value="Windows"), mock.patch(
            "platform.machine", return_value="ARM64"
        ), mock.patch.object(
            torch.ops.tensorrt,
            "get_current_platform",
            return_value=torch.ops.tensorrt._platform_win_x86_64(),
        ):
            self.assertEqual(
                torchtrt.Platform.current_platform(), torchtrt.Platform.WIN_X86_64
            )

    def test_current_platform_python_fallback_uses_python_windows_arch(self):
        disabled_runtime_features = torchtrt_enums.ENABLED_FEATURES._replace(
            torch_tensorrt_runtime=False
        )
        with mock.patch.object(
            torchtrt_enums, "ENABLED_FEATURES", disabled_runtime_features
        ), mock.patch("platform.system", return_value="Windows"), mock.patch(
            "platform.machine", return_value="ARM64"
        ), mock.patch(
            "sysconfig.get_platform", return_value="win-amd64"
        ):
            self.assertEqual(
                torchtrt.Platform.current_platform(), torchtrt.Platform.WIN_X86_64
            )

    def test_unknown_platform_serializes_to_runtime_token(self):
        self.assertEqual(
            torchtrt.Platform.UNKNOWN._to_serialized_rt_platform(),
            torch.ops.tensorrt._platform_unknown(),
        )
