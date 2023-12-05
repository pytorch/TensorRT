import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_tensorrt

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class TestSafeMode(TestCase):
    def test_safe_mode_enabled(self):
        torch_tensorrt.enable_safe_inference_mode()
        self.assertTrue(torch.ops.tensorrt.get_safe_mode())

    def test_unsafe_mode_enabled(self):
        torch_tensorrt.enable_unsafe_inference_mode()
        self.assertFalse(torch.ops.tensorrt.get_safe_mode())

    def test_unsafe_mode_enabled_inference_python(self):
        torch_tensorrt.enable_unsafe_inference_mode()

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
            msg=f"Unsafe Mode Python TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_unsafe_mode_enabled_inference_cpp(self):
        torch_tensorrt.enable_unsafe_inference_mode()

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
            msg=f"Unsafe Mode C++ TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
