import unittest
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
import torch
import torchvision.models as models
import tensorrt as trt
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
        self.assertEqual(device.gpu_id, 0) # Override since AGX platforms use iGPU to manage DLA
        self.assertEqual(device.dla_core, 0)
        self.assertEqual(device.allow_gpu_fallback, True)

    def test_from_torch(self):
        device = torchtrt.Device._from_torch_device(torch.device("cuda:0"))
        self.assertEqual(device.device_type, torchtrt.DeviceType.GPU)
        self.assertEqual(device.gpu_id, 0)
