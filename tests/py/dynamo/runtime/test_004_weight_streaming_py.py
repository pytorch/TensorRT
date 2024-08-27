import itertools
import os
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

INPUT_SIZE = (64, 128)


class TestWeightStreamingPython(TestCase):
    @parameterized.expand(
        [
            ("use_python_runtime", True),
            ("not_use_python_runtime", False),
        ]
    )
    def test_weight_streaming(self, _, use_python_runtime):
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mat1 = torch.randn((128, 32)).cuda()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = torch.matmul(x, self.mat1)
                out = self.relu((out + 2) * 0.5)
                return out

        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        fx_graph = torch.fx.symbolic_trace(model)

        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input,
            ir="dynamo",
            min_block_size=1,
            debug=True,
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.enable_weight_streaming(optimized_model) as ws_context:
            streamable_weight_bytes = ws_context.get_streamable_weight_bytes()
            ws_budget_bytes = ws_context.set_streamable_weight_bytes(
                int(streamable_weight_bytes * 0.7)
            )
            assert ws_budget_bytes > 0
            print("ws_budget1", ws_budget_bytes)
            optimized_model(*input)

            streamable_weight_bytes = ws_context.get_streamable_weight_bytes()
            ws_budget_bytes = ws_context.set_streamable_weight_bytes(
                streamable_weight_bytes + 1
            )
            print("ws_budget2", ws_budget_bytes)
            optimized_model(*input)

            ws_budget_bytes = ws_context.set_streamable_weight_bytes(0)
            print("ws_budget3", ws_budget_bytes)
            optimized_model(*input)

        with torchtrt.runtime.enable_weight_streaming(optimized_model) as ws_context:
            ws_budget_bytes = ws_context.set_automatic_streaming_budget()
            print("automatic ws_budget", ws_budget_bytes)
            optimized_model(*input)

        with torch.no_grad():
            for _ in range(3):
                optimized_model(*input)
                torch.cuda.synchronize()

        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("use_python_runtime", True),
            ("not_use_python_runtime", False),
        ]
    )
    def test_weight_streaming_multi_rt(self, _, use_python_runtime):
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mat1 = torch.randn((128, 32)).cuda()
                self.mat2 = torch.randn((32, 512)).cuda()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = torch.matmul(x, self.mat1)
                out = self.relu((out + 2) * 0.5)
                out = torch.matmul(out, self.mat2)
                return out

        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        fx_graph = torch.fx.symbolic_trace(model)

        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input,
            ir="dynamo",
            min_block_size=1,
            debug=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.enable_weight_streaming(optimized_model) as mod:
            streamable_weight_bytes = mod.get_streamable_weight_bytes()
            print("streamable_weight_bytes = ", streamable_weight_bytes)
            mod.set_streamable_weight_bytes(10640)
            ret = optimized_model(*input)

            mod.set_streamable_weight_bytes(10640 * 2)
            ret = optimized_model(*input)
            for pct in range(10, 100, 10):
                bytes = int(pct * 0.01 * streamable_weight_bytes)
                mod.set_streamable_weight_bytes(bytes)
                optimized_model(*input)

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
