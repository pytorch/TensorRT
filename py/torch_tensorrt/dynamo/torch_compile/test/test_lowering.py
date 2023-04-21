from functools import partial
from utils import fx_dynamo_testing_backend
from torch.testing._internal.common_utils import run_tests, TestCase
import torch


class TestTRTModule(TestCase):
    def test_lowering_inplace_op(self):
        class FullySupported(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                x = torch.ops.aten.add_.Tensor(x, y)
                x = torch.ops.aten.relu_.default(x)
                return x

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {torch.ops.aten.add.Tensor, torch.ops.aten.relu.default}

        # Trace module and set up custom backend to track intermediate graphs
        fx_graph = torch.fx.symbolic_trace(FullySupported())
        partitioned_graphs = []
        custom_backend = partial(
            fx_dynamo_testing_backend,
            store_intermediate_graphs=partitioned_graphs,
        )

        # Invoke compilation
        compiled_graph = torch.compile(fx_graph, backend=custom_backend)
        compiled_graph(
            torch.rand(
                5,
            ).cuda(),
            torch.rand(
                5,
            ).cuda(),
        )

        # Iterate over intermediate graphs, attempt to match nodes
        for fx_module in partitioned_graphs:
            for _, submodule in fx_module.named_children():
                for node in submodule.graph.nodes:

                    if node.op == "call_function" and node.target in expected_ops:
                        expected_ops.remove(node.target)

        self.assertEqual(
            len(expected_ops), 0, "All operators should have been decomposed"
        )


if __name__ == "__main__":
    run_tests()
