import itertools

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
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(100, 128)
                self.layer2 = torch.nn.Linear(128, 64)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = self.layer1(x)
                out = self.relu((out + 2.0) * 0.05)
                out = self.layer2(out)
                return out

        inputs = torchtrt.Input(
            min_shape=(1, 100),
            opt_shape=(64, 100),
            max_shape=(128, 100),
            dtype=torch.float,
            name="x",
        )
        model = SampleModel().eval().cuda()
        fx_graph = torch.fx.symbolic_trace(model)

        input_list = []
        input_list.append(torch.randn((8, 100)).cuda())
        input_list.append(torch.randn((12, 100)).cuda())
        input_list.append(torch.randn((12, 100)).cuda())
        input_list.append(torch.randn((8, 100)).cuda())
        input_list.append(torch.randn((8, 100)).cuda())

        optimized_model = torchtrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_explicit_typing=True,
            enable_weight_streaming=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
            use_python_runtime=use_python_runtime,
        )

        # List of tuples representing different configurations for three features:
        # Cuda graphs, pre-allocated output buffer, weight streaming change
        states = list(itertools.product((True, False), repeat=3))
        # Create pairs of these configurations, representing an initial state and a changed state
        states_permutations = itertools.permutations(states, 2)

        pre_allocated_output_ctx = torchtrt.runtime.enable_pre_allocated_outputs(
            optimized_model
        )
        weight_streaming_ctx = torchtrt.runtime.weight_streaming(optimized_model)
        streamable_budget = weight_streaming_ctx.total_device_budget

        for init_state, changed_state in states_permutations:
            for cuda_graphs, pre_allocated_output, weight_streaming in [
                init_state,
                changed_state,
            ]:
                torchtrt.runtime.set_cudagraphs_mode(cuda_graphs)
                pre_allocated_output_ctx.set_pre_allocated_output(pre_allocated_output)

                if weight_streaming:
                    weight_streaming_ctx.device_budget = int(streamable_budget * 0.8)
                else:
                    weight_streaming_ctx.device_budget = streamable_budget

                ref_out_list = []
                trt_out_list = []
                # Input shape changes
                for i in range(len(input_list)):
                    if weight_streaming and i == 4:
                        weight_streaming_ctx.device_budget = int(
                            streamable_budget * 0.6
                        )
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
