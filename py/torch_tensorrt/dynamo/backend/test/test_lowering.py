from functools import partial
from utils import lower_graph_testing
from torch.testing._internal.common_utils import run_tests, TestCase
import torch


class TestLowering(TestCase):
    def test_lowering_inplace_op(self):
        class InPlace(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                x = torch.ops.aten.add_.Tensor(x, y)
                x = torch.ops.aten.relu_.default(x)
                return x

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {torch.ops.aten.add.Tensor, torch.ops.aten.relu.default}

        inputs = [
            torch.rand(
                5,
            ),
            torch.rand(
                5,
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(InPlace())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph, inputs, expected_ops=expected_ops, min_block_size=2
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

    def test_lowering_alias_replacement(self):
        class Alias(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.ops.aten.alias.default(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten.alias.default}

        inputs = [
            torch.rand(
                5,
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(Alias())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

    def test_lowering_rsqrt(self):
        class Rsqrt(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.ops.aten.rsqrt.default(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.sqrt.default, torch.ops.aten.reciprocal.default}
        unexpected_ops = {torch.ops.aten.rsqrt.default}

        inputs = [
            torch.randint(
                1,
                10,
                (5,),
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(Rsqrt())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )


if __name__ == "__main__":
    run_tests()
