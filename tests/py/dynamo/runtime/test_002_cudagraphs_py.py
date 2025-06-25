import itertools
import os
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


class TestCudagraphsPython(TestCase):
    def tearDown(self):
        # Reset to default cuda graph mode after each test
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def test_cudagraphs_on(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        self.assertTrue(torch_tensorrt.runtime.get_cudagraphs_mode())

    def test_cudagraphs_off(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        self.assertFalse(torch_tensorrt.runtime.get_cudagraphs_mode())

    def test_cudagraphs_context(self):
        class SampleModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.abs.default(input)

        model = SampleModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        optimized_model = torch_tensorrt.compile(
            model,
            "torch_compile",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )
        with torch_tensorrt.runtime.enable_cudagraphs(optimized_model) as _:
            self.assertTrue(torch_tensorrt.runtime.get_cudagraphs_mode())
        self.assertFalse(torch_tensorrt.runtime.get_cudagraphs_mode())

    def test_cudagraphs_enabled_inference_python(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [torch.randn(*INPUT_SIZE).cuda() for _ in range(TRIALS)]
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
        )

        result_samples = []
        torch_results_samples = []
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for i in inputs:
                result_samples.append(cudagraphs_module(i).detach().cpu())
                torch_results_samples.append(fx_graph(i).detach().cpu())

        for i, (optimized_model_results, torch_model_results) in enumerate(
            zip(result_samples, torch_results_samples)
        ):
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"CUDA Graph Python TRT outputs don't match with the original model. (trial: {i})",
            )

        torch._dynamo.reset()

    def test_cudagraphs_enabled_inference_python_cpu_offload(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [torch.randn(*INPUT_SIZE).cuda() for _ in range(TRIALS)]
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
            offload_module_to_cpu=True,
        )
        optimized_model.cuda()
        result_samples = []
        torch_results_samples = []
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for i in inputs:
                result_samples.append(cudagraphs_module(i).detach().cpu())
                torch_results_samples.append(fx_graph(i).detach().cpu())

        for i, (optimized_model_results, torch_model_results) in enumerate(
            zip(result_samples, torch_results_samples)
        ):
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"CUDA Graph Python TRT outputs don't match with the original model. (trial: {i})",
            )

        torch._dynamo.reset()

    def test_cudagraphs_enabled_fallback_inference_python(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [torch.randn(*INPUT_SIZE).cuda() for _ in range(TRIALS)]
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=True,
        )

        result_samples = []
        torch_results_samples = []
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for i in inputs:
                result_samples.append(cudagraphs_module(i).detach().cpu())
                torch_results_samples.append(fx_graph(i).detach().cpu())

        for i, (optimized_model_results, torch_model_results) in enumerate(
            zip(result_samples, torch_results_samples)
        ):
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"CUDA Graph Python TRT outputs don't match with the original model. (trial: {i})",
            )
        torch._dynamo.reset()

    def test_cudagraphs_enabled_fallback_inference_python_cpu_offload(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        inputs = [torch.randn(*INPUT_SIZE).cuda() for _ in range(TRIALS)]
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=True,
            offload_module_to_cpu=True,
        )
        optimized_model.cuda()

        result_samples = []
        torch_results_samples = []
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for i in inputs:
                result_samples.append(cudagraphs_module(i).detach().cpu())
                torch_results_samples.append(fx_graph(i).detach().cpu())

        for i, (optimized_model_results, torch_model_results) in enumerate(
            zip(result_samples, torch_results_samples)
        ):
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"CUDA Graph Python TRT outputs don't match with the original model. (trial: {i})",
            )
        torch._dynamo.reset()

    @unittest.skipIf(
        os.environ.get("CI_BUILD") == "1",
        "Skipping test due to CI resource constraints",
    )
    def test_cudagraphs_recapture_py(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5)

        inputs = [
            TRIALS * [torch.randn(*(2 * (i + 1), 2 * (i + 1))).cuda()]
            for i in range(TRIALS)
        ]
        inputs = list(itertools.chain.from_iterable(inputs))
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=True,
        )

        result_samples = []
        torch_results_samples = []
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for i in inputs:
                result_samples.append(cudagraphs_module(i).detach().cpu())
                torch_results_samples.append(fx_graph(i).detach().cpu())

        for i, (optimized_model_results, torch_model_results) in enumerate(
            zip(result_samples, torch_results_samples)
        ):
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                DECIMALS_OF_AGREEMENT,
                msg=f"CUDA Graph Python TRT outputs don't match with the original model. (trial: {i})",
            )


if __name__ == "__main__":
    run_tests()
