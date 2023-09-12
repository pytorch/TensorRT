import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestInputAsOutput(TestCase):
    def test_input_as_output(self):
        class InputAsOutput(torch.nn.Module):
            def forward(self, x, y):
                y_new = y + x + 1
                y_new = y_new * 7
                return (y_new, x, y)

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InputAsOutput())
        lower_graph_testing(fx_graph, inputs, min_block_size=1)
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"InputAsOutput TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestLoweringPassMembership(TestCase):
    def insert_at_end(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            add_lowering_pass,
            remove_lowering_pass,
        )

        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        add_lowering_pass(identity_pass)

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[-1])

        remove_lowering_pass(-1)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)

    def insert_at_index(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            add_lowering_pass,
            remove_lowering_pass,
        )

        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        add_lowering_pass(identity_pass, 0)

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[0])

        remove_lowering_pass(0)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)


if __name__ == "__main__":
    run_tests()
