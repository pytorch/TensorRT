import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

INPUT_SIZE = (64, 100)


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
        self.layer1 = torch.nn.Linear(100, 128)
        self.layer2 = torch.nn.Linear(32, 64)
        self.mat1 = torch.randn((128, 32)).cuda()
        self.relu = torch.nn.ReLU()
        self.mat2 = torch.randn((64, 512)).cuda()

    def forward(self, x):
        out = self.layer1(x)
        out = torch.matmul(out, self.mat1)
        out = self.relu((out + 2.0) * 0.05)
        out = self.layer2(out)
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
            cache_built_engines=False,
            reuse_cached_engines=False,
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
            cache_built_engines=False,
            reuse_cached_engines=False,
            min_block_size=1,
            debug=True,
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )
        # Weight streaming budget is applied manually.
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget

            weight_streaming_ctx.device_budget = current_budget * 0.7
            current_ws_budget_bytes = get_current_weight_streaming_bytes(
                optimized_model
            )
            assert weight_streaming_ctx.device_budget == current_ws_budget_bytes
            optimized_model(*input)

            # There are no weights on the GPU so full streaming
            weight_streaming_ctx.device_budget = 0
            current_ws_budget_bytes = get_current_weight_streaming_bytes(
                optimized_model
            )
            assert weight_streaming_ctx.device_budget == current_ws_budget_bytes

            weight_streaming_ctx.device_budget = current_budget * 0.5
            current_ws_budget_bytes = get_current_weight_streaming_bytes(
                optimized_model
            )
            assert weight_streaming_ctx.device_budget == current_ws_budget_bytes
            out = optimized_model(*input)

        # Weight streaming is disabled after the exit from weight streaming context
        current_ws_budget_bytes = get_current_weight_streaming_bytes(optimized_model)
        assert current_ws_budget_bytes == current_budget

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
    def test_weight_streaming_invalid_usage(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        fx_graph = torch.fx.symbolic_trace(model)

        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input,
            ir="dynamo",
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            debug=True,
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )

        # Setting weight streaming context to unsupported module
        with torchtrt.runtime.weight_streaming(model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget
            assert current_budget == -1

        # Expects weight streaming is disabled if invalid budget size is set
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget

            # Values is larger than streamable weights size
            weight_streaming_ctx.device_budget = current_budget + 1
            assert weight_streaming_ctx.device_budget == current_budget
            optimized_model(*input)

            # negative weight budget size
            weight_streaming_ctx.device_budget = -1
            assert weight_streaming_ctx.device_budget == current_budget
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
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
            enable_weight_streaming=True,
        )
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget
            for pct in [0.0, 0.1, 0.5, 0.9, 1.0]:
                weight_streaming_ctx.device_budget = int(pct * current_budget)
                current_ws_budget_bytes = get_current_weight_streaming_bytes(
                    optimized_model
                )
                assert weight_streaming_ctx.device_budget == current_ws_budget_bytes
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
