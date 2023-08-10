import unittest

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.utils import (
    prepare_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)
from utils import same_output_format


class TestToTorchDevice(unittest.TestCase):
    def test_cast_cuda_device(self):
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        prepared_device = to_torch_device(device)
        self.assertTrue(isinstance(prepared_device, torch.device))
        self.assertTrue(prepared_device.index == gpu_id)

    def test_cast_trt_device(self):
        gpu_id = 4
        device = torch_tensorrt.Device(gpu_id=gpu_id)
        prepared_device = to_torch_device(device)
        self.assertTrue(isinstance(prepared_device, torch.device))
        self.assertTrue(prepared_device.index == gpu_id)

    def test_cast_str_device(self):
        gpu_id = 2
        device = f"cuda:{2}"
        prepared_device = to_torch_device(device)
        self.assertTrue(isinstance(prepared_device, torch.device))
        self.assertTrue(prepared_device.index == gpu_id)


class TestToTorchTRTDevice(unittest.TestCase):
    def test_cast_cuda_device(self):
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        prepared_device = to_torch_tensorrt_device(device)
        self.assertTrue(isinstance(prepared_device, torch_tensorrt.Device))
        self.assertTrue(prepared_device.gpu_id == gpu_id)

    def test_cast_trt_device(self):
        gpu_id = 4
        device = torch_tensorrt.Device(gpu_id=gpu_id)
        prepared_device = to_torch_tensorrt_device(device)
        self.assertTrue(isinstance(prepared_device, torch_tensorrt.Device))
        self.assertTrue(prepared_device.gpu_id == gpu_id)

    def test_cast_str_device(self):
        gpu_id = 2
        device = f"cuda:{2}"
        prepared_device = to_torch_tensorrt_device(device)
        self.assertTrue(isinstance(prepared_device, torch_tensorrt.Device))
        self.assertTrue(prepared_device.gpu_id == gpu_id)


class TestPrepareInputs(unittest.TestCase):
    def test_prepare_single_tensor_input(self):
        inputs = [torch.ones((4, 4))]
        prepared_inputs_trt, prepared_inputs_torch = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
        )
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_torch, enforce_tensor_type=False)
        )

    def test_prepare_trt_input(self):
        inputs = [torch_tensorrt.Input(shape=(4, 3), dtype=torch.float)]
        prepared_inputs_trt, prepared_inputs_torch = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
        )
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_torch, enforce_tensor_type=False)
        )

    def test_prepare_mixed_type_compound_tensor_input(self):
        inputs = {
            "first": [
                torch.ones((4, 4)),
                torch_tensorrt.Input(shape=(4, 3), dtype=torch.float),
            ],
            "second": (
                torch.rand((5, 1)),
                (torch.rand((5, 1)), torch_tensorrt.Input(shape=(2, 3))),
            ),
        }
        prepared_inputs_trt, prepared_inputs_torch = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
        )
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_torch, enforce_tensor_type=False)
        )


if __name__ == "__main__":
    unittest.main()
