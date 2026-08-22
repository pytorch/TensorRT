import unittest

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.utils import (
    get_torch_tensor,
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


class TestGetTorchTensor(unittest.TestCase):
    def test_shape_tensor_preserves_multiple_values(self):
        shape = torch_tensorrt.Input(
            min_shape=(3, 5),
            opt_shape=(3, 7),
            max_shape=(4, 10),
            dtype=torch.int64,
            is_shape_tensor=True,
        )

        value = get_torch_tensor(shape, torch.device("cpu"))

        self.assertEqual(value, [3, 7])


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


class TestDeprecatedInputsAlias(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_empty_inputs_tuple_with_kwarg_inputs(self):
        class KwargsOnly(torch.nn.Module):
            def forward(self, *, x):
                return x + 1

        model = KwargsOnly().eval().cuda()
        x = torch.randn(2, 3, device="cuda")
        ep = torch.export.export(model, (), {"x": x})
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=(),
            kwarg_inputs={"x": x},
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(trt_gm(x=x), model(x=x), rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_empty_arg_inputs_tuple_convert_method_to_trt_engine(self):
        class KwargsOnly(torch.nn.Module):
            def forward(self, *, x):
                return x + 1

        model = KwargsOnly().eval().cuda()
        x = torch.randn(2, 3, device="cuda")
        engine = torch_tensorrt.convert_method_to_trt_engine(
            model,
            "forward",
            arg_inputs=(),
            kwarg_inputs={"x": x},
            ir="dynamo",
            min_block_size=1,
        )
        self.assertIsInstance(engine, bytes)
        self.assertGreater(len(engine), 0)

    def test_save_rejects_empty_arg_inputs_with_inputs(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        with self.assertRaises(AssertionError):
            torch_tensorrt.save(M(), arg_inputs=(), inputs=(torch.randn(2, 3),))


if __name__ == "__main__":
    unittest.main()
