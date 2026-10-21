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


@torch.library.custom_op(
    "torchtrt_cudagraphs_test::accumulate_", mutates_args=("cache",)
)
def _accumulate_(cache: torch.Tensor, x: torch.Tensor) -> None:
    cache.add_(x)


@_accumulate_.register_fake
def _(cache, x):
    return None


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

    def test_cudagraphs_whole_graph_kwargs_order_cpp(self):
        # A graph break puts the whole module under one CUDA graph. Keyword arguments
        # in another order than at export must still reach the right inputs
        class SampleModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.relu((x + 2) * 0.5) - y

        model = SampleModel().eval().cuda()
        kwargs = {
            "x": torch.randn(*INPUT_SIZE).cuda(),
            "y": torch.randn(*INPUT_SIZE).cuda(),
        }
        optimized_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (), kwargs),
            arg_inputs=(),
            kwarg_inputs=kwargs,
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
        )
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for _ in range(TRIALS):
                x, y = torch.randn(*INPUT_SIZE).cuda(), torch.randn(*INPUT_SIZE).cuda()
                torch.testing.assert_close(cudagraphs_module(y=y, x=x), model(x=x, y=y))

        torch._dynamo.reset()

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

    def test_cudagraphs_whole_graph_input_updated_in_place_cpp(self):
        # The op without a converter puts the module under one CUDA graph. Its write
        # to the cache must land in the caller's tensor at every replay
        class SampleModel(torch.nn.Module):
            def forward(self, x, cache):
                torch.ops.torchtrt_cudagraphs_test.accumulate_(cache, x)
                return torch.relu(cache * 0.5)

        model = SampleModel().eval().cuda()
        x = torch.randn(*INPUT_SIZE).cuda()
        optimized_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x, torch.zeros(*INPUT_SIZE).cuda())),
            arg_inputs=(x, torch.zeros(*INPUT_SIZE).cuda()),
            min_block_size=1,
        )
        cache, expected = (
            torch.zeros(*INPUT_SIZE).cuda(),
            torch.zeros(*INPUT_SIZE).cuda(),
        )
        with torch_tensorrt.runtime.enable_cudagraphs(
            optimized_model
        ) as cudagraphs_module:
            for _ in range(TRIALS):
                x = torch.randn(*INPUT_SIZE).cuda()
                out = cudagraphs_module(x, cache)
                torch.testing.assert_close(out, model(x, expected))
                torch.testing.assert_close(cache, expected)

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
