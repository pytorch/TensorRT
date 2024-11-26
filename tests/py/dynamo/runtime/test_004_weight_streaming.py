import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

INPUT_SIZE = (64, 100)


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 128)
        self.layer2 = torch.nn.Linear(30, 64)
        self.mat1 = torch.randn((128, 32)).cuda()
        self.mat2 = torch.randn((64, 512)).cuda()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv1d(64, 6, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.matmul(out, self.mat1)
        out = self.relu((out + 2.0) * 0.05)
        out = self.conv(out)
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
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )
        # Checking if default weight streaming budget(automatic) is applied when compiler option was provided
        weight_streaming_ctx = torchtrt.runtime.weight_streaming(optimized_model)
        assert weight_streaming_ctx.device_budget > 0

        requested_budget = int(weight_streaming_ctx.total_device_budget * 0.5)
        weight_streaming_ctx.device_budget = requested_budget

        # Weight streaming context keeps current device budget size
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget
            assert current_budget == requested_budget

            new_budget = int(weight_streaming_ctx.total_device_budget * 0.2)
            weight_streaming_ctx.device_budget = new_budget

        # Weight streaming budget is reverted after the exit from weight streaming context
        assert weight_streaming_ctx.device_budget == requested_budget

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
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )
        # Weight streaming budget is applied manually.
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget

            requested_budget = int(streamable_budget * 0.7)
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

            optimized_model(*input)

            # Full streaming by applying 0 budget
            weight_streaming_ctx.device_budget = 0
            assert weight_streaming_ctx.device_budget == 0

            # Automatic weight streaming size
            requested_budget = (
                weight_streaming_ctx.get_automatic_weight_streaming_budget()
            )
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

            requested_budget = int(streamable_budget * 0.5)
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

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
            ("python_runtime", True, False),
            ("python_runtime_multi_rt", True, True),
            ("cpp_runtime", False, False),
            ("cpp_runtime_multi_rt", False, True),
        ]
    )
    def test_weight_streaming_invalid_usage(self, _, use_python_runtime, multi_rt):
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
            torch_executed_ops=(
                {"torch.ops.aten.convolution.default"} if multi_rt else {}
            ),
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )

        # Setting weight streaming context to unsupported module
        with torchtrt.runtime.weight_streaming(model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget
            assert streamable_budget == 0

        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget

            # Values is larger than max budget disables weight streaming
            weight_streaming_ctx.device_budget = streamable_budget + 1
            assert weight_streaming_ctx.device_budget == streamable_budget

            try:
                # Runtime error if requested budget is negative
                weight_streaming_ctx.device_budget = -1
                assert False
            except RuntimeError:
                assert True

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
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget
            for pct in [0.05, 0.2, 0.4, 0.8, 1.0]:
                requested_budget = int(streamable_budget * pct)
                weight_streaming_ctx.device_budget = requested_budget

                # Budget distribution to multiple submodule may result in integer differences of at most 1
                assert abs(weight_streaming_ctx.device_budget - requested_budget) <= 1
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
