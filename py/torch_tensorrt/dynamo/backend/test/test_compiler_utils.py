from torch_tensorrt.dynamo.backend.utils import prepare_device, prepare_inputs
from utils import same_output_format
import torch_tensorrt
import unittest
import torch


class TestPrepareDevice(unittest.TestCase):
    def test_prepare_cuda_device(self):
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        prepared_device = prepare_device(device)
        self.assertTrue(isinstance(prepared_device, torch.device))
        self.assertTrue(prepared_device.index == gpu_id)

    def test_prepare_trt_device(self):
        gpu_id = 4
        device = torch_tensorrt.Device(gpu_id=gpu_id)
        prepared_device = prepare_device(device)
        self.assertTrue(isinstance(prepared_device, torch.device))
        self.assertTrue(prepared_device.index == gpu_id)


class TestPrepareInputs(unittest.TestCase):
    def test_prepare_single_tensor_input(self):
        inputs = [torch.ones((4, 4))]
        prepared_inputs = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs, enforce_tensor_type=False)
        )

    def test_prepare_trt_input(self):
        inputs = [torch_tensorrt.Input(shape=(4, 3), dtype=torch.float)]
        prepared_inputs = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs, enforce_tensor_type=False)
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
        prepared_inputs = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs, enforce_tensor_type=False)
        )


if __name__ == "__main__":
    unittest.main()
