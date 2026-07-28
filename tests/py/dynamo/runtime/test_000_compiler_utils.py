import unittest

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.utils import (
    prepare_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)

from ..testing_utilities import same_output_format


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
        prepared_inputs_trt = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
        )

    def test_prepare_trt_input(self):
        inputs = [torch_tensorrt.Input(shape=(4, 3), dtype=torch.float)]
        prepared_inputs_trt = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
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
        prepared_inputs_trt = prepare_inputs(inputs)
        self.assertTrue(
            same_output_format(inputs, prepared_inputs_trt, enforce_tensor_type=False)
        )

    def test_prepare_tensor_does_not_copy_data(self):
        """Verify that prepare_inputs does not duplicate GPU tensor data.

        When torch.compile lifts model parameters as graph inputs,
        prepare_inputs receives every weight tensor. Previously,
        torch.tensor(t) created a full copy of each tensor, doubling GPU
        memory usage. Input.from_tensor only needs shape/dtype metadata,
        so no copy is necessary.
        """
        original = torch.randn(1024, 1024, device="cuda")
        before = torch.cuda.memory_allocated()
        result = prepare_inputs([original])
        after = torch.cuda.memory_allocated()
        # No significant new allocation (allow small overhead, but not a full copy)
        self.assertLess(
            after - before,
            original.nelement() * original.element_size(),
            "prepare_inputs should not allocate a full copy of the input tensor",
        )
        # Result should preserve shape and dtype
        self.assertEqual(result[0].shape, original.shape)
        self.assertEqual(result[0].dtype, original.dtype)

    def test_prepare_scalar_inputs(self):
        """Verify that scalar inputs are still converted to tensors."""
        int_result = prepare_inputs(42)
        self.assertIsInstance(int_result, torch_tensorrt.Input)

        float_result = prepare_inputs(3.14)
        self.assertIsInstance(float_result, torch_tensorrt.Input)

        bool_result = prepare_inputs(True)
        self.assertIsInstance(bool_result, torch_tensorrt.Input)


if __name__ == "__main__":
    unittest.main()
