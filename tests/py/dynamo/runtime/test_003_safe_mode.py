import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT


@unittest.skipIf(torch.cuda.device_count() == 1, "System does not have multiple GPUs")
class TestSafeMode(TestCase):
    def test_multi_device_safe_mode_on(self):
        torch_tensorrt.runtime.set_multi_device_safe_mode(True)
        self.assertTrue(torch.ops.tensorrt.get_multi_device_safe_mode())

    def test_multi_device_safe_mode_off(self):
        torch_tensorrt.runtime.set_multi_device_safe_mode(False)
        self.assertFalse(torch.ops.tensorrt.get_multi_device_safe_mode())

    def test_multi_device_safe_mode_context(self):
        with torch_tensorrt.runtime.set_multi_device_safe_mode(True):
            self.assertTrue(torch.ops.tensorrt.get_multi_device_safe_mode())
        self.assertFalse(torch.ops.tensorrt.get_multi_device_safe_mode())

    def test_multi_device_safe_mode_enabled_inference_python(self):
        torch_tensorrt.runtime.set_multi_device_safe_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [
            torch.randn(
                3,
                5,
                7,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Safe Mode Python TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT runtime is not available",
    )
    def test_multi_device_safe_mode_enabled_inference_cpp(self):
        torch_tensorrt.runtime.set_multi_device_safe_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [
            torch.randn(
                3,
                5,
                7,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=False,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Safe Mode C++ TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
