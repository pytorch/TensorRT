import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests


class TestLowRankInputs(TestCase):
    def test_0D_input(self):
        class Tensor0DInput(torch.nn.Module):
            def forward(self, x):
                return x * 7

        inputs = [
            torch.tensor(
                3,
            )
            .cuda()
            .int(),
        ]

        fx_graph = torch.fx.symbolic_trace(Tensor0DInput())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"0D-Tensor TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_1D_input(self):
        class Tensor1DInput(torch.nn.Module):
            def forward(self, x, y):
                return (x + 7.1) / (y * 2.1)

        inputs = [torch.rand((3, 1)).cuda(), torch.rand((3, 1)).cuda()]

        fx_graph = torch.fx.symbolic_trace(Tensor1DInput())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"1D-Tensor TRT outputs don't match with the original model.",
        )

        # Validate that the runtime moves cpu inputs to cuda
        optimized_model(torch.rand((3, 1)), torch.rand((3, 1)))

        torch._dynamo.reset()

    def test_1D_input_cpu_offload(self):
        class Tensor1DInput(torch.nn.Module):
            def forward(self, x, y):
                return (x + 7.1) / (y * 2.1)

        inputs = [torch.rand((3, 1)).cuda(), torch.rand((3, 1)).cuda()]

        fx_graph = torch.fx.symbolic_trace(Tensor1DInput())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=True,
            offload_module_to_cpu=True,
        )
        fx_graph.cuda()
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"1D-Tensor TRT outputs don't match with the original model.",
        )

        # Validate that the runtime moves cpu inputs to cuda
        optimized_model(torch.rand((3, 1)), torch.rand((3, 1)))

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
