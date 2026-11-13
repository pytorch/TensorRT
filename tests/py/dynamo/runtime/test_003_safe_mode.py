# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from contextlib import nullcontext

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

    def test_safe_mode_runs_on_the_engine_device(self):
        """An engine built on cuda:0 and called while cuda:1 is current must still
        run on cuda:0, and must leave the caller's current device alone."""
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 256)
            )
            .eval()
            .to("cuda:0")
        )
        x = torch.randn(8, 256, device="cuda:0")
        with torch.cuda.device(0):
            trt_model = torch_tensorrt.compile(
                model, ir="dynamo", inputs=[x], min_block_size=1
            )
            with torch.no_grad():
                ref = model(x)

        for cudagraphs in (False, True):
            with self.subTest(cudagraphs=cudagraphs):
                with torch_tensorrt.runtime.set_multi_device_safe_mode(True):
                    with (
                        torch_tensorrt.runtime.enable_cudagraphs(trt_model)
                        if cudagraphs
                        else nullcontext(trt_model)
                    ) as runner:
                        torch.cuda.reset_peak_memory_stats(1)
                        allocated_on_caller = torch.cuda.memory_allocated(1)
                        with torch.cuda.device(1):
                            out = runner(x)
                            self.assertEqual(torch.cuda.current_device(), 1)
                if not cudagraphs:
                    # Nothing of this call belongs on the caller's GPU, inputs included.
                    self.assertEqual(
                        torch.cuda.max_memory_allocated(1), allocated_on_caller
                    )
                self.assertEqual(out.device, torch.device("cuda:0"))
                torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
