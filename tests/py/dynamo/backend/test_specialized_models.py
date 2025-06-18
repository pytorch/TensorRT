# type: ignore
import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestFakeTensors(TestCase):
    def test_lowering_mul_int(self):
        class MulInt(torch.nn.Module):
            def forward(self, x):
                return x * 7

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.mul.Tensor,
        }

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(MulInt())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
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
            msg=f"MulInt TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_lowering_add_float(self):
        class AddFloat(torch.nn.Module):
            def forward(self, x):
                return x + 84.0

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.add.Tensor,
        }

        inputs = [
            torch.rand(
                1,
                5,
                7,
                9,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddFloat())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
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
            msg=f"AddFloat TRT outputs don't match with the original model.",
        )

        torch._dynamo.reset()


class Test0DTensors(TestCase):
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


class TestTensorFreezing(TestCase):
    def test_tensor_freeze_attr(self):
        class TensorFreeze(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones((8, 2), device="cuda")

            def forward(self, x):
                return x @ self.const

        inputs = [
            torch.ones(
                7,
                8,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(TensorFreeze())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
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
            msg=f"Frozen-Tensor TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_constant_fold(self):
        class Arange(torch.nn.Module):
            def forward(self, x):
                y = torch.arange(10, device="cuda")
                return x + y

        inputs = [
            torch.rand(
                10,
                10,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(Arange())

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
            msg=f"Constant Folded TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestPacketOperator(TestCase):
    def test_packet_operator(self):
        class PacketAsOperator(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.tanh(x)

        # Operations expected to be removed in the traced graph
        expected_ops = {torch.ops.aten.tanh.default}
        unexpected_ops = {
            torch.ops.aten.tanh,
        }

        inputs = [torch.rand((3, 5)).cuda()]

        fx_graph = torch.fx.symbolic_trace(PacketAsOperator())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()


class TestInputModifications(TestCase):
    def test_input_modifications_add(self):
        class InplaceAdd(torch.nn.Module):
            def forward(self, x):
                x += 3
                y = x + 1
                return y

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InplaceAdd())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
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
            msg=f"InplaceAdd TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_input_modifications_mul(self):
        class InplaceMul(torch.nn.Module):
            def forward(self, x, y):
                x *= 5.0
                x *= 1.9
                z = x + y
                z /= 1.3
                return z

        inputs = [
            torch.rand(
                1,
                3,
                5,
                7,
            ).cuda(),
            torch.rand(
                1,
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InplaceMul())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
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
            msg=f"InplaceMul TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestDeconvolution(TestCase):
    def test_ConvTranspose2d(self):
        class Up(torch.nn.Module):
            def __init__(self, in_channels, out_channels, upsample_stride):
                super().__init__()
                self.up = torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    upsample_stride,
                    stride=upsample_stride,
                    bias=False,
                )

            def forward(self, x):
                return self.up(x)

        device = torch.device("cuda:0")
        model = Up(64, 128, 2).to(device)
        model.eval()
        print(model)

        x = torch.rand((1, 64, 100, 100)).to(device)
        model_opt = torch.compile(
            model,
            backend="torch_tensorrt",
            options={
                "min_block_size": 1,
            },
        )
        with torch.no_grad():
            _ = model_opt(x)


if __name__ == "__main__":
    run_tests()
