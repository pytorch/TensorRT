import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

INPUT_SIZE = (64, 128)


# Helper to get current weight streaming budet in runtime module
def get_current_weight_streaming_bytes(runtime_module):
    total_bytes = 0
    for name, rt_mod in runtime_module.named_children():
        if "_run_on_acc" in name and (
            isinstance(rt_mod, PythonTorchTensorRTModule)
            or isinstance(rt_mod, TorchTensorRTModule)
        ):
            total_bytes += rt_mod.get_weight_streaming_budget()
    return total_bytes


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.randn((128, 32)).cuda()
        self.mat2 = torch.randn((32, 512)).cuda()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = torch.matmul(x, self.mat1)
        out = self.relu((out + 2.0) * 0.05)
        out = torch.matmul(out, self.mat2)
        return out


class TestWeightStreamingPython(TestCase):

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_default(self, _, use_python_runtime):
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
        # Checking default weight streaming budget(automatic) is applied
        current_ws_budget_bytes = get_current_weight_streaming_bytes(optimized_model)
        assert current_ws_budget_bytes > 0

        ref = model(*input)
        out = optimized_model(*input)

        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_manual(self, _, use_python_runtime):
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
        # Weight streaming budget is applied manually.
        with torchtrt.runtime.enable_weight_streaming(optimized_model) as ws_context:
            streamable_weight_bytes = ws_context.get_streamable_weight_bytes()
            ws_budget_bytes = ws_context.set_streamable_weight_bytes(
                int(streamable_weight_bytes * 0.7)
            )
            assert ws_budget_bytes > 0
            optimized_model(*input)

            ws_budget_bytes = ws_context.set_streamable_weight_bytes(
                int(streamable_weight_bytes * 0.5)
            )
            assert ws_budget_bytes > 0
            out = optimized_model(*input)

        ref = model(*input)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_invalid_size(self, _, use_python_runtime):
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
        # Expects weight streaming is disabled if invalid budget size is set
        with torchtrt.runtime.enable_weight_streaming(optimized_model) as ws_context:
            streamable_weight_bytes = ws_context.get_streamable_weight_bytes()

            # Values is larger than streamable_weights_size.
            ws_budget_bytes = ws_context.set_streamable_weight_bytes(
                streamable_weight_bytes + 1
            )
            assert ws_budget_bytes == streamable_weight_bytes
            optimized_model(*input)

            # zero weight budget size
            ws_budget_bytes = ws_context.set_streamable_weight_bytes(0)
            current_ws_budget_bytes = get_current_weight_streaming_bytes(
                optimized_model
            )
            assert current_ws_budget_bytes == streamable_weight_bytes
            optimized_model(*input)

        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_multi_rt(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        fx_graph = torch.fx.symbolic_trace(model)

        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input,
            ir="dynamo",
            min_block_size=1,
            # debug=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.enable_weight_streaming(optimized_model) as ws_context:
            streamable_weight_bytes = ws_context.get_streamable_weight_bytes()
            for pct in [0.1, 0.5, 0.9]:
                bytes = int(pct * streamable_weight_bytes)
                ws_budget_bytes = ws_context.set_streamable_weight_bytes(bytes)
                assert ws_budget_bytes > 0
                out = optimized_model(*input)

        ref = model(*input)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
