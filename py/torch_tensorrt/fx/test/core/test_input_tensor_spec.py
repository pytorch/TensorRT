# Owner(s): ["oncall: gpu_enablement"]

from typing import List, Optional

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.fx import generate_input_specs, InputTensorSpec, LowerSetting


class TestTRTModule(TestCase):
    def _validate_spec(
        self,
        spec: InputTensorSpec,
        tensor: torch.Tensor,
        dynamic_dims: Optional[List[int]] = None,
    ):
        expected_shape = list(tensor.shape)
        if dynamic_dims:
            for dim in dynamic_dims:
                expected_shape[dim] = -1
        self.assertSequenceEqual(spec.shape, expected_shape)
        self.assertEqual(spec.dtype, tensor.dtype)
        self.assertEqual(spec.device, tensor.device)
        self.assertTrue(spec.has_batch_dim)

    def test_from_tensor(self):
        tensor = torch.randn(1, 2, 3)
        spec = InputTensorSpec.from_tensor(tensor)
        self._validate_spec(spec, tensor)

    def test_from_tensors(self):
        tensors = [torch.randn(1, 2, 3), torch.randn(2, 4)]
        specs = InputTensorSpec.from_tensors(tensors)
        for spec, tensor in zip(specs, tensors):
            self._validate_spec(spec, tensor)

    def test_from_tensors_with_dynamic_batch_size(self):
        tensors = [torch.randn(1, 2, 3), torch.randn(1, 4)]
        batch_size_range = [2, 3, 4]
        specs = InputTensorSpec.from_tensors_with_dynamic_batch_size(
            tensors, batch_size_range
        )
        for spec, tensor in zip(specs, tensors):
            self._validate_spec(spec, tensor, dynamic_dims=[0])

            for batch_size, shape in zip(batch_size_range, spec.shape_ranges[0]):
                self.assertEqual(batch_size, shape[0])
                self.assertSequenceEqual(tensor.shape[1:], shape[1:])

    def test_from_tensors_with_dynamic_batch_size_different_batch_dims(self):
        tensors = [torch.randn(1, 2, 3), torch.randn(2, 1, 4)]
        batch_size_range = [2, 3, 4]
        specs = InputTensorSpec.from_tensors_with_dynamic_batch_size(
            tensors, batch_size_range, batch_dims=[0, 1]
        )
        for i, spec_and_tensor in enumerate(zip(specs, tensors)):
            spec, tensor = spec_and_tensor
            self._validate_spec(spec, tensor, dynamic_dims=[i])

            for batch_size, shape in zip(batch_size_range, spec.shape_ranges[0]):
                self.assertEqual(batch_size, shape[i])
                tensor_shape = list(tensor.shape)
                tensor_shape[i] = batch_size
                self.assertSequenceEqual(tensor_shape, shape)

    def test_generate_input_specs(self):
        lower_setting = LowerSetting(
            explicit_batch_dimension=False, max_batch_size=256, opt_profile_replica=2
        )

        # Implicit batch dim.
        inputs = [torch.randn(1, 2, 3)]
        specs = generate_input_specs(inputs, lower_setting)
        for spec, tensor in zip(specs, inputs):
            self._validate_spec(spec, tensor)

        # Explicit batch dim without additional inputs.
        lower_setting.explicit_batch_dimension = True
        specs = generate_input_specs(inputs, lower_setting)
        for spec, tensor in zip(specs, inputs):
            self._validate_spec(spec, tensor, dynamic_dims=[0])
            self.assertEqual(len(spec.shape_ranges), lower_setting.opt_profile_replica)

        # Explicit batch dim with additional inputs.
        additional_inputs = [torch.randn(1, 1, 3)]
        specs = generate_input_specs(inputs, lower_setting, additional_inputs)
        for spec, tensor in zip(specs, inputs):
            self._validate_spec(spec, tensor, dynamic_dims=[1])
            self.assertEqual(len(spec.shape_ranges), lower_setting.opt_profile_replica)


if __name__ == "__main__":
    run_tests()
