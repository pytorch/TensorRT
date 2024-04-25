import os
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_tensorrt


class TestHardwareCompatibility(TestCase):
    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    @unittest.skipIf(
        not torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8,
        "HW Compatibility is not supported on cards older than Ampere",
    )
    def test_hw_compat_enabled(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x * 7) @ x.T, dim=0)

        inputs = [torch.randn(5, 7).cuda()]

        # Validate that the hardware compatibility mode has been enabled
        optimized_model_hw_compat = torch_tensorrt.compile(
            SampleModel(),
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            hardware_compatible=True,
            use_python_runtime=False,
            output_format="graph_module",
        )

        self.assertTrue(optimized_model_hw_compat._run_on_acc_0.hardware_compatible)

        cpp_repr = optimized_model_hw_compat._run_on_acc_0.engine._properties.__repr__()

        self.assertIn("Hardware Compatibility: Enabled", cpp_repr)

        # Validate that the hardware compatibility mode has been disabled
        optimized_model_not_hw_compat = torch_tensorrt.compile(
            SampleModel(),
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            hardware_compatible=False,
            use_python_runtime=False,
            output_format="graph_module",
        )

        self.assertFalse(
            optimized_model_not_hw_compat._run_on_acc_0.hardware_compatible
        )

        cpp_repr = (
            optimized_model_not_hw_compat._run_on_acc_0.engine._properties.__repr__()
        )

        self.assertIn("Hardware Compatibility: Disabled", cpp_repr)

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime
        or torch.ops.tensorrt.ABI_VERSION() != "5",
        "Torch-TensorRT runtime is not available or ABI Version is compatible",
    )
    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT runtime is not available",
    )
    @unittest.skipIf(
        not torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8,
        "HW Compatibility is not supported on cards older than Ampere",
    )
    def test_hw_compat_3080_build(self):
        inputs = [torch.randn(1, 3, 224, 224).cuda()]

        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        model = torch.jit.load("../../ts/models/hw_compat.ts").cuda()
        out = model(*inputs)
        self.assertTrue(
            len(out) == 1 and isinstance(out, torch.Tensor),
            "Invalid output detected",
        )
        os.chdir(cwd)


if __name__ == "__main__":
    run_tests()
