# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import unittest

import torch
import torch_tensorrt as torch_trt
from torch.profiler import ProfilerActivity, profile
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias


class TestDynamicWorkspaceAllocation(unittest.TestCase):
    def test_workspace_is_allocated_in_bytes(self):
        self.addCleanup(torch._dynamo.reset)

        class ConvNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, 3, 1)
                self.conv2 = torch.nn.Conv2d(6, 16, 3, 1)

            def forward(self, x):
                return torch.relu(self.conv2(torch.relu(self.conv1(x))))

        inputs = [torch.rand((100, 3, 224, 224), device="cuda")]
        compiled_module = torch_trt.compile(
            ConvNet().eval().cuda(),
            inputs=inputs,
            ir="dynamo",
            immutable_weights=False,
            lazy_engine_init=True,
            dynamically_allocate_resources=True,
            min_block_size=1,
        )

        warmup_output = compiled_module(*inputs)
        runtime_modules = [
            module
            for module in compiled_module.modules()
            if isinstance(module, TorchTensorRTModule)
        ]
        self.assertEqual(len(runtime_modules), 1)

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(runtime_modules[0].serialized_engine)
        self.assertIsNotNone(engine)
        workspace_bytes = engine.device_memory_size_v2
        self.assertGreater(workspace_bytes, 0)
        del engine, runtime, warmup_output
        gc.collect()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        output = compiled_module(*inputs)
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline_allocated
        output_bytes = output.numel() * output.element_size()
        allocator_tolerance = max(1024**2, workspace_bytes // 8)

        self.assertLessEqual(
            peak_delta,
            workspace_bytes + output_bytes + allocator_tolerance,
            msg=(
                "Dynamic allocation exceeded the TensorRT byte requirement: "
                f"peak delta={peak_delta}, workspace={workspace_bytes}, "
                f"output={output_bytes}, tolerance={allocator_tolerance}"
            ),
        )

    def _compile_with_dynamic_allocation(self):
        self.addCleanup(torch._dynamo.reset)
        self.addCleanup(torch_trt.runtime.set_cudagraphs_mode, False)
        model = (
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.Conv2d(8, 16, 3)
            )
            .eval()
            .cuda()
        )
        inputs = [torch.rand((8, 3, 64, 64), device="cuda")]
        compiled_module = torch_trt.compile(
            model,
            inputs=inputs,
            ir="dynamo",
            dynamically_allocate_resources=True,
            min_block_size=1,
        )
        (runtime_module,) = [
            module
            for module in compiled_module.modules()
            if isinstance(module, TorchTensorRTModule)
        ]
        return model, inputs, compiled_module, runtime_module

    def _workspace_bytes(self, runtime_module):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(runtime_module.serialized_engine)
        workspace_bytes = engine.device_memory_size_v2
        self.assertGreater(workspace_bytes, 0)
        return workspace_bytes

    def test_cudagraph_keeps_its_workspace(self):
        if not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the C++ runtime.")
        if torch_trt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        model, inputs, compiled_module, runtime_module = (
            self._compile_with_dynamic_allocation()
        )
        workspace_bytes = self._workspace_bytes(runtime_module)

        def captures():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                compiled_module(*inputs)
                torch.cuda.synchronize()
            return [e.name for e in prof.events()].count("cudaStreamBeginCapture")

        with torch.no_grad():
            expected = model(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.assertEqual(captures(), 1)
            # The caching allocator hands out any workspace that an earlier call freed.
            buffer = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
            self.assertEqual(
                captures(),
                0,
                msg="an allocation between calls made a call record again",
            )
            output = compiled_module(*inputs)
            torch.cuda.synchronize()
            self.assertEqual(
                torch.count_nonzero(buffer).item(),
                0,
                msg="a CUDA graph replay wrote to a workspace freed after recording",
            )
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

            torch_trt.runtime.set_cudagraphs_mode(False)
            held = torch.cuda.memory_allocated()
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                # Delays the call below until after the default stream zeroes its buffer.
                torch.cuda._sleep(500_000_000)
                compiled_module(*inputs)
            # The default stream allocated the workspace that the call above released.
            late_buffer = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            self.assertEqual(
                torch.count_nonzero(late_buffer).item(),
                0,
                msg="a call on another stream wrote to a workspace handed out again",
            )
            del late_buffer
            torch.cuda.empty_cache()
            self.assertGreaterEqual(
                held - torch.cuda.memory_allocated(),
                workspace_bytes,
                msg="the workspace outlived the CUDA graph",
            )

    def test_outer_capture_after_cudagraphs_off_owns_its_workspace(self):
        if not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the C++ runtime.")
        if torch_trt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        model, inputs, compiled_module, runtime_module = (
            self._compile_with_dynamic_allocation()
        )
        workspace_bytes = self._workspace_bytes(runtime_module)

        with torch.no_grad():
            expected = model(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(True)
            compiled_module(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(False)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            # The captured call destroys the engine's own graph, which the default mode forbids.
            with torch.cuda.graph(graph, capture_error_mode="relaxed"):
                output = compiled_module(*inputs)
            torch.cuda.synchronize()
            # The caching allocator hands out any workspace that the captured call freed.
            buffer = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
            graph.replay()
            torch.cuda.synchronize()

        self.assertEqual(
            torch.count_nonzero(buffer).item(),
            0,
            msg="a replay of the caller's graph wrote to a workspace freed after capture",
        )
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_output_allocator_call_drops_the_cudagraph(self):
        if not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the C++ runtime.")
        if torch_trt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        model, inputs, compiled_module, _ = self._compile_with_dynamic_allocation()

        with torch.no_grad():
            expected = model(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(True)
            compiled_module(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(False)
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                with torch_trt.runtime.enable_output_allocator(compiled_module):
                    compiled_module(*inputs)
                torch_trt.runtime.set_cudagraphs_mode(True)
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    output = compiled_module(*inputs)
                    torch.cuda.synchronize()

        self.assertEqual(
            [e.name for e in prof.events()].count("cudaStreamBeginCapture"),
            1,
            msg="a call after the output-allocator path replayed the graph recorded on another stream",
        )
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_recording_call_that_runs_out_of_memory_can_be_retried(self):
        if not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the C++ runtime.")
        if torch_trt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the C++ runtime does not capture a CUDA graph of its own."
            )
        model, inputs, compiled_module, runtime_module = (
            self._compile_with_dynamic_allocation()
        )
        workspace_bytes = self._workspace_bytes(runtime_module)
        self.addCleanup(torch.cuda.set_per_process_memory_fraction, 1.0)

        with torch.no_grad():
            expected = model(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # No new memory, and fillers take every cached block the workspace could reuse.
            torch.cuda.set_per_process_memory_fraction(
                torch.cuda.memory_reserved()
                / torch.cuda.get_device_properties(0).total_memory
            )
            fillers = []
            with self.assertRaises(torch.OutOfMemoryError):
                for _ in range(1000):
                    fillers.append(
                        torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
                    )
            with torch_trt.runtime.enable_pre_allocated_outputs(compiled_module):
                with self.assertRaises(torch.OutOfMemoryError):
                    compiled_module(*inputs)
                del fillers
                torch.cuda.set_per_process_memory_fraction(1.0)
                output = compiled_module(*inputs)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_python_runtime_cudagraph_keeps_its_workspace(self):
        if torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the Python runtime.")
        if torch_trt.ENABLED_FEATURES.tensorrt_rtx:
            self.skipTest(
                "On TensorRT-RTX the Python runtime does not capture a CUDA graph of its own."
            )
        model, inputs, compiled_module, runtime_module = (
            self._compile_with_dynamic_allocation()
        )
        engine = runtime_module.engine

        with torch.no_grad():
            expected = model(*inputs)
            torch_trt.runtime.set_cudagraphs_mode(True)
            compiled_module(*inputs)
            graph, workspace = engine.cudagraph, engine._dynamic_workspace
            output = compiled_module(*inputs)
            torch.cuda.synchronize()

        self.assertIs(
            engine._dynamic_workspace,
            workspace,
            msg="the engine replaced the workspace its CUDA graph replays into",
        )
        self.assertIs(
            engine.cudagraph,
            graph,
            msg="a same-shape call recorded the CUDA graph again",
        )
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_python_runtime_releases_workspace_with_the_context(self):
        if torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self.skipTest("Checks the Python runtime.")
        _, inputs, compiled_module, runtime_module = (
            self._compile_with_dynamic_allocation()
        )
        engine = runtime_module.engine

        with torch.no_grad():
            compiled_module(*inputs)
        self.assertIsNotNone(engine._dynamic_workspace)
        runtime_module.use_dynamically_allocated_resources(False)
        self.assertIsNone(
            engine._dynamic_workspace,
            msg="the engine kept its dynamic workspace after switching to static",
        )
