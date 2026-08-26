import itertools
import os
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestCudagraphsCPP(TestCase):
    def setUp(self):
        # Ensure a clean cudagraphs state regardless of prior test ordering
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        # Reset to default cuda graph mode after each test
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def test_cudagraphs_on(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        self.assertTrue(torch.ops.tensorrt.get_cudagraphs_mode())

    def test_cudagraphs_off(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        self.assertFalse(torch.ops.tensorrt.get_cudagraphs_mode())

    def test_cudagraphs_context(self):
        class SampleModel(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.abs.default(input)

        fx_graph = torch.fx.symbolic_trace(SampleModel())
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
        )
        with torch_tensorrt.runtime.enable_cudagraphs(optimized_model) as _:
            self.assertTrue(torch.ops.tensorrt.get_cudagraphs_mode())
        self.assertFalse(torch.ops.tensorrt.get_cudagraphs_mode())

    def _assert_context_invalidation_recaptures(self, trigger):
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1)

        model = SampleModel().eval().cuda()
        first_input = torch.randn((2, 3), device="cuda")
        second_input = torch.randn_like(first_input)
        exported = torch.export.export(model, (first_input,))
        optimized_model = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[first_input],
            min_block_size=1,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        trt_modules = [
            module
            for module in optimized_model.modules()
            if isinstance(module, TorchTensorRTModule)
        ]
        self.assertEqual(len(trt_modules), 1)
        trt_module = trt_modules[0]
        engine = trt_module.get_engine()

        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            first_output = cudagraphs_module(first_input)
            captures_before = engine.num_cudagraph_captures()
            contexts_before = engine.num_execution_contexts_created()
            self.assertEqual(captures_before, 1)

            trigger(trt_module)
            second_output = cudagraphs_module(second_input)

            self.assertEqual(engine.num_cudagraph_captures(), captures_before + 1)
            self.assertEqual(
                engine.num_execution_contexts_created(), contexts_before + 1
            )

        torch.testing.assert_close(first_output, model(first_input))
        torch.testing.assert_close(second_output, model(second_input))

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "TRT-RTX owns CUDA graph capture internally",
    )
    def test_disable_profiling_recaptures_cudagraph_cpp(self):
        self._assert_context_invalidation_recaptures(
            lambda trt_module: trt_module.disable_profiling()
        )

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "TRT-RTX owns CUDA graph capture internally",
    )
    def test_resource_allocation_change_recaptures_cudagraph_cpp(self):
        self._assert_context_invalidation_recaptures(
            lambda trt_module: trt_module.use_dynamically_allocated_resources(True)
        )

    def test_cudagraphs_enabled_inference_cpp(self):
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
                msg=f"CUDA Graph C++ TRT outputs don't match with the original model. (trial: {i})",
            )

        torch._dynamo.reset()

    def test_cudagraphs_enabled_inference_cpp_cpu_offload(self):
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
                msg=f"CUDA Graph C++ TRT outputs don't match with the original model. (trial: {i})",
            )

        torch._dynamo.reset()

    def test_cudagraphs_enabled_fallback_inference_cpp(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5)

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

    def test_cudagraphs_enabled_fallback_inference_cpp_cpu_offload(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5)

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
    def test_cudagraphs_recapture_cpp(self):
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
