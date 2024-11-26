import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime._WrapperTorchTensorRTModule import (
    WrapperTorchTensorRTModule,
)

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


class TestWrapperCudagraphs(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True, False),
            ("python_runtime_multi_out", True, True),
            ("cpp_runtime", False, False),
            ("cpp_runtime_multi_out", False, True),
        ]
    )
    def test_wrapper_cudagraphs(self, _, use_python_runtime, multi_output):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5)

        class SampleModelMultiOutput(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5), torch.relu((x - 2) * 2.1)

        input_list = []
        for _ in range(TRIALS):
            input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
            input_list.append(input)

        model = SampleModel() if multi_output else SampleModelMultiOutput()
        fx_graph = torch.fx.symbolic_trace(model)

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torchtrt.compile(
            fx_graph,
            "dynamo",
            input_list[0],
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
        )

        ref_out_list = []
        trt_out_list = []

        for enable_cuda_graphs in [False, True]:
            for i in range(len(input_list)):
                # Toggles cuda graph at all index in TRIALS
                if i % TRIALS == i // TRIALS:
                    cuda_graphs = enable_cuda_graphs
                else:
                    cuda_graphs = not enable_cuda_graphs
                torchtrt.runtime.set_cudagraphs_mode(cuda_graphs)

                trt_out_list.append(optimized_model(*input_list[i]))
                ref_out_list.append(fx_graph(*input_list[i]))

        for optimized_model_results, torch_model_results in zip(
            trt_out_list, ref_out_list
        ):
            torch.testing.assert_close(
                torch_model_results,
                optimized_model_results,
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
    def test_wrapper_cudagraphs_dynamic(self, _, use_python_runtime):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu((x + 2) * 0.5)

        inputs = torchtrt.Input(
            min_shape=(1, 3, 128, 224),
            opt_shape=(8, 3, 192, 224),
            max_shape=(16, 3, 224, 224),
            dtype=torch.float,
            name="x",
        )
        fx_graph = torch.fx.symbolic_trace(SampleModel())

        optimized_model = torchtrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
        )

        input_list = []
        ref_out_list = []
        trt_out_list = []
        # Alternating cuda_graphs enable and input shapes at every five iterations.
        for i in [1, 3, 8, 11, 16]:
            for j in [128, 128, 222, 222, 224]:
                input_list.append(torch.randn((i, 3, j, 224)).cuda())

        for enable_cuda_graphs in [False, True]:
            for i in range(len(input_list)):
                # Toggles cuda graph at all index in TRIALS
                if i % TRIALS == i // TRIALS:
                    cuda_graphs = enable_cuda_graphs
                else:
                    cuda_graphs = not enable_cuda_graphs
                torchtrt.runtime.set_cudagraphs_mode(cuda_graphs)

                trt_out_list.append(fx_graph(input_list[i]))
                ref_out_list.append(optimized_model(input_list[i]))

        for optimized_model_results, torch_model_results in zip(
            trt_out_list, ref_out_list
        ):
            torch.testing.assert_close(
                torch_model_results,
                optimized_model_results,
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
    def test_wrapper_cudagraphs_conv(self, _, use_python_runtime):
        """
        Graph break at torch convolution that may have memory allocation
        and it's not expected to be recorded.
        """

        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(64, 6, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = 1 + self.conv(x)
                out = self.relu(out)
                return out

        model = SampleModel().eval().cuda()
        input_list = []
        trt_out_list = []
        ref_out_list = []

        for _ in range(TRIALS):
            input = [torch.randn((64, 32), dtype=torch.float32).cuda()]
            input_list.append(input)
        fx_graph = torch.fx.symbolic_trace(model)

        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input_list[0],
            ir="dynamo",
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
        )

        torchtrt.runtime.set_cudagraphs_mode(True)
        for i in range(TRIALS):
            trt_out_list.append(optimized_model(*input_list[i]))
            ref_out_list.append(fx_graph(*input_list[i]))

        for optimized_model_results, torch_model_results in zip(
            trt_out_list, ref_out_list
        ):
            torch.testing.assert_close(
                torch_model_results,
                optimized_model_results,
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
    def test_wrapper_cudagraphs_api(self, _, use_python_runtime):
        """
        3 api draft
        """

        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(64, 6, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = 1 + self.conv(x)
                out = self.relu(out)
                return out

        model = SampleModel().eval().cuda()
        input_list = []
        trt_out_list = []
        ref_out_list = []

        for _ in range(TRIALS):
            input = [torch.randn((64, 32), dtype=torch.float32).cuda()]
            input_list.append(input)
        fx_graph = torch.fx.symbolic_trace(model)

        # 1. Compiler option: enable_wrapper_module=True
        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input_list[0],
            ir="dynamo",
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
            enable_wrapper_module=True,
        )

        with torchtrt.runtime.enable_cudagraphs():
            for i in range(TRIALS):
                trt_out_list.append(optimized_model(*input_list[i]))
                ref_out_list.append(fx_graph(*input_list[i]))

        # Compiler again to generate normal module
        optimized_model = torchtrt.compile(
            fx_graph,
            inputs=input_list[0],
            ir="dynamo",
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
        )
        # This is current cuda runtime api
        with torchtrt.runtime.enable_cudagraphs():
            for i in range(TRIALS):
                trt_out_list.append(optimized_model(*input_list[i]))
                ref_out_list.append(fx_graph(*input_list[i]))

        # 2. Optional parameter in existing cuda runtime api
        # WrapperTorchTensorRTModule can be simplified to have only cuda graph path
        with torchtrt.runtime.enable_cudagraphs(optimized_model) as wrapped_module:
            for i in range(TRIALS):
                trt_out_list.append(wrapped_module(*input_list[i]))
                ref_out_list.append(fx_graph(*input_list[i]))

        # 3. Use Wrapper module directly
        wrapped_module = WrapperTorchTensorRTModule(optimized_model)
        with torchtrt.runtime.enable_cudagraphs():
            for i in range(TRIALS):
                trt_out_list.append(wrapped_module(*input_list[i]))
                ref_out_list.append(fx_graph(*input_list[i]))

        for optimized_model_results, torch_model_results in zip(
            trt_out_list, ref_out_list
        ):
            torch.testing.assert_close(
                torch_model_results,
                optimized_model_results,
                rtol=5e-03,
                atol=5e-03,
                equal_nan=True,
                check_dtype=True,
            )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
