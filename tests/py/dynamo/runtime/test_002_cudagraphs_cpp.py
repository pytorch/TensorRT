# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os
import subprocess
import sys
import textwrap
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

    def test_engine_with_captured_graph_destroys_graph_first(self):
        if torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        # A wrong teardown order can crash the process, so the engine is destroyed in a child.
        # The profiler records runtime calls but not the driver's module unloads, so the
        # first cudaFree, cudaFreeAsync or cudaStreamDestroy stands in for the start of
        # TensorRT's teardown.
        child = textwrap.dedent("""
            import gc
            import torch
            import torch_tensorrt
            from torch.profiler import ProfilerActivity, profile

            model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU()).eval().cuda()
            x = torch.randn(1, 3, 32, 32, device="cuda")
            trt_model = torch_tensorrt.compile(model, ir="dynamo", inputs=[x], min_block_size=1)
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
            trt_model(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                del trt_model
                gc.collect()
                torch.cuda.synchronize()
            events = sorted(prof.events(), key=lambda e: e.time_range.start)
            print("CUDA_CALLS=" + ",".join(e.name for e in events))
            """)
        result = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        line = next(
            l for l in result.stdout.splitlines() if l.startswith("CUDA_CALLS=")
        )
        calls = line[len("CUDA_CALLS=") :].split(",")
        self.assertIn("cudaGraphExecDestroy", calls, msg=calls)
        teardown = [
            i
            for i, c in enumerate(calls)
            if c in ("cudaFree", "cudaFreeAsync", "cudaStreamDestroy")
        ]
        self.assertTrue(
            teardown, msg=f"no free or stream destroy was recorded: {calls}"
        )
        self.assertLess(
            calls.index("cudaGraphExecDestroy"),
            teardown[0],
            msg=f"the graph was destroyed after TensorRT started freeing: {calls}",
        )

    def test_dropping_execution_context_releases_captured_graph(self):
        if torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        # Replaying a graph captured on a dropped context can crash, so this runs in a child process.
        # A stale replay does not always crash, so the child also checks that the drop itself
        # destroys the graph.
        child = textwrap.dedent("""
            import torch
            import torch_tensorrt
            from torch.profiler import ProfilerActivity, profile

            class Attention(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.qkv = torch.nn.Linear(64, 192)
                    self.out = torch.nn.Linear(64, 64)

                def forward(self, x):
                    q, k, v = self.qkv(x).unflatten(-1, (3, 4, 16)).permute(2, 0, 3, 1, 4)
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    return self.out(y.transpose(1, 2).flatten(2))

            model = Attention().eval().cuda()
            x = torch.randn(2, 128, 64, device="cuda")
            trt_model = torch_tensorrt.compile(model, ir="dynamo", inputs=[x], min_block_size=1)
            engines = [
                m
                for m in trt_model.modules()
                if isinstance(m, torch_tensorrt.runtime.TorchTensorRTModule)
            ]
            assert engines
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
            with torch.no_grad():
                expected = model(x)
                for name, drop_context in (
                    ("disable_profiling", lambda m: m.disable_profiling()),
                    ("use_dynamically_allocated_resources", lambda m: m.use_dynamically_allocated_resources(True)),
                ):
                    trt_model(x)
                    trt_model(x)
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for m in engines:
                            drop_context(m)
                    calls = [e.name for e in prof.events()]
                    assert "cudaGraphExecDestroy" in calls, f"{name} kept the graph: {calls}"
                    torch.testing.assert_close(trt_model(x), expected, rtol=1e-2, atol=1e-2)
            """)
        result = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
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
