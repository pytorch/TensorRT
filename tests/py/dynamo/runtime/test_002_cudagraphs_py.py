# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
    def setUp(self):
        # Ensure a clean cudagraphs state regardless of prior test ordering
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

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

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Checks the Python runtime engine; test_002_cudagraphs_cpp.py covers the C++ runtime.",
    )
    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "On TensorRT-RTX the Python runtime does not capture a CUDA graph of its own.",
    )
    def test_dropping_execution_context_releases_captured_graph(self):
        model = (
            torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU()).eval().cuda()
        )
        x = torch.randn(1, 3, 32, 32, device="cuda")
        trt_model = torch_tensorrt.compile(
            model, ir="dynamo", inputs=[x], min_block_size=1
        )
        engines = [
            m.engine
            for m in trt_model.modules()
            if isinstance(m, torch_tensorrt.runtime.TorchTensorRTModule)
        ]
        self.assertTrue(engines)
        expected = model(x)
        torch_tensorrt.runtime.set_cudagraphs_mode(True)
        for name, drop_context in (
            ("disable_profiling", lambda e: e.disable_profiling()),
            (
                "use_dynamically_allocated_resources",
                lambda e: e.use_dynamically_allocated_resources(True),
            ),
            ("invalidate_context", lambda e: e.invalidate_context()),
        ):
            trt_model(x)
            self.assertTrue(all(e.cudagraph is not None for e in engines))
            for e in engines:
                drop_context(e)
            self.assertTrue(
                all(e.cudagraph is None for e in engines),
                msg=f"{name} kept the captured graph",
            )
            torch.testing.assert_close(
                trt_model(x),
                expected,
                rtol=1e-2,
                atol=1e-2,
                msg=f"The call after {name} does not match the original model.",
            )


if __name__ == "__main__":
    run_tests()
