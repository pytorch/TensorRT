# type: ignore
from copy import deepcopy

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.partitioning import fast_partition

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestTRTModuleNextCompilation(TestCase):
    def test_trt_module_next_full_support(self):
        class FullySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out + x
                out = 2 * out
                out = out + y
                return torch.mean(out, dim=1)

        fx_graph = torch.fx.symbolic_trace(FullySupportedMultiOp())
        partitioned_graph, _ = fast_partition(deepcopy(fx_graph), min_block_size=3)

        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graph.named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            1,
            "All operators are supported, there should be one segment",
        )

        inputs = [
            torch.randint(-5, 5, (16, 7), dtype=torch.float).cuda(),
            torch.randint(-5, 5, (16, 7), dtype=torch.float).cuda(),
        ]

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            use_python_runtime=False,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"TRT outputs don't match with the original model.",
        )

    def test_trt_module_next_partial_support(self):
        class PartiallySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out - 3 * x
                out = out + y
                out = out.to(torch.float)
                out = 2 * out
                return torch.mean(out, dim=-1)

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        unexpected_ops = {torch.ops.aten.add.Tensor}

        inputs = [
            torch.randint(-40, 40, (16, 7, 5), dtype=torch.int).cuda(),
            torch.randint(1, 40, (16, 7, 5), dtype=torch.int).cuda(),
        ]

        (
            unexpected_ops_seen,
            _,
            partitioned_graphs,
        ) = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            testing_partitioning=True,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )
        self.assertEqual(
            len(partitioned_graphs),
            1,
            "Without control flow breaks, there should only be a single graph",
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graphs[0].named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            2,
            "Certain operators are set to run in Torch, expected 2 segments",
        )

        torch._dynamo.reset()

        model = PartiallySupportedMultiOp().eval().cuda()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            model,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            use_python_runtime=False,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = model(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"TRT outputs don't match with the original model.",
        )


class TestCompilationOptions(TestCase):
    def test_trt_specific_options(self):
        class SupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                out = x - y
                out = out - 3 * x
                out = out + y
                out = out - y / 5
                out = 2 * out
                return torch.mean(out, dim=-1)

        fx_graph = torch.fx.symbolic_trace(SupportedMultiOp())

        inputs = [
            torch.randint(-40, 40, (16, 7, 5), dtype=torch.float).cuda(),
            torch.randint(1, 40, (16, 7, 5), dtype=torch.float).cuda(),
        ]

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=False,
            optimization_level=4,
            version_compatible=True,
            max_aux_streams=5,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"TRT outputs don't match with the original model.",
        )


class Test64BitInput(TestCase):
    def test_float64_input_full_support(self):
        class FullySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.mean.dim(
                    torch.ops.aten.mul.Tensor(torch.ops.aten.add.Tensor(x, y), 2), [0]
                )

        fx_graph = torch.fx.symbolic_trace(FullySupportedMultiOp())
        partitioned_graph, _ = fast_partition(deepcopy(fx_graph), min_block_size=3)

        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            1,
            "All operators are supported, there should be one segment",
        )

        inputs = [
            torch.randint(-5, 5, (16, 7), dtype=torch.double).cuda(),
            torch.randint(-5, 5, (16, 7), dtype=torch.double).cuda(),
        ]

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            truncate_double=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"TRT outputs don't match with the original model.",
        )

    def test_int64_input_partial_support(self):
        class PartiallySupportedMultiOp(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.abs(torch.ops.aten.add.Tensor(x, y))

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        unexpected_ops = {torch.ops.aten.add.Tensor}

        inputs = [
            torch.randint(-40, 40, (1, 16, 7, 5), dtype=torch.long).cuda(),
            torch.randint(1, 40, (1, 16, 7, 5), dtype=torch.long).cuda(),
        ]

        (
            unexpected_ops_seen,
            _,
            partitioned_graphs,
        ) = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            testing_partitioning=True,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )
        self.assertEqual(
            len(partitioned_graphs),
            1,
            "Without control flow breaks, there should only be a single graph",
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graphs[0].named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            1,
            "Certain operators are set to run in Torch, expected 1 segment",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            truncate_double=False,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"TRT outputs don't match with the original model.",
        )


class TestRegistration(TestCase):
    def test_torch_tensorrt_registration(self):
        self.assertIn("torch_tensorrt", torch._dynamo.list_backends())

    def test_tensorrt_registration(self):
        self.assertIn("tensorrt", torch._dynamo.list_backends())


if __name__ == "__main__":
    run_tests()
