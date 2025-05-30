import pytest
import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


class TestPreAllocatedOutputs(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    @pytest.mark.critical
    def test_pre_allocated_outputs_default(self, _, use_python_runtime):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        model = SampleModel().eval().cuda()
        inputs = [torch.randn(*INPUT_SIZE).cuda() for _ in range(TRIALS)]
        fx_graph = torch.fx.symbolic_trace(model)

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torchtrt.compile(
            fx_graph,
            "torch_compile",
            inputs[0],
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=use_python_runtime,
        )

        ref_out_list = []
        trt_out_list = []
        with torchtrt.runtime.enable_pre_allocated_outputs(optimized_model):
            for i in inputs:
                ref_out_list.append(fx_graph(i).detach().cpu())
                trt_out_list.append(optimized_model(i).detach().cpu())

        for torch_model_results, optimized_model_results in zip(
            ref_out_list, trt_out_list
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
    def test_pre_allocated_outputs_dynamic(self, _, use_python_runtime):
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

        pre_allocated_output_ctx = torchtrt.runtime.enable_pre_allocated_outputs(
            optimized_model
        )
        pre_allocated_output = False
        for enable_cuda_graphs in [False, True]:
            for i in range(len(input_list)):
                # Toggles cuda graph at all index in TRIALS
                if i % TRIALS == i // TRIALS:
                    cuda_graphs = enable_cuda_graphs
                else:
                    cuda_graphs = not enable_cuda_graphs
                if i % 3 == 0:
                    pre_allocated_output = not pre_allocated_output

                torchtrt.runtime.set_cudagraphs_mode(cuda_graphs)
                pre_allocated_output_ctx.set_pre_allocated_output(pre_allocated_output)

                ref_out_list.append(fx_graph(input_list[i]))
                trt_out_list.append(optimized_model(input_list[i]))

        for torch_model_results, optimized_model_results in zip(
            ref_out_list, trt_out_list
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
