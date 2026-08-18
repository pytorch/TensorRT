import gc
import unittest

import torch
import torch_tensorrt as torch_trt
import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


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
        engine = runtime.deserialize_cuda_engine(
            runtime_modules[0].serialized_engine
        )
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
