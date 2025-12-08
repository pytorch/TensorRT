import torch
import torch.nn as nn
from torch.fx.passes.splitter_base import Subgraph
from torch.ops import aten
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes import post_lowering, pre_export_lowering
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    ResourcePartitioner,
)


class TestResourcePartitioning(TestCase):
    def test_atomic_subgraph_correction(self):
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()
                self.fc = nn.Linear(3 * 224 * 224, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]

        enabled_precisions = {torch.float}
        use_python_runtime = False

        exp_program = torch.export.export(model, tuple(inputs))

        compilation_options = {
            "use_python_runtime": use_python_runtime,
            "enabled_precisions": enabled_precisions,
            "min_block_size": 1,
            "immutable_weights": True,
            "reuse_cached_engines": False,
            "enable_resource_partitioning": True,
        }
        settings = CompilationSettings(**compilation_options)

        exported_program = pre_export_lowering(exp_program, settings)
        exported_program = exported_program.run_decompositions(
            get_decompositions(False)
        )

        gm = exported_program.module()
        gm = post_lowering(gm, settings)

        partitioned_module, supported_ops = partitioning.fast_partition(
            gm,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
            require_full_compilation=settings.require_full_compilation,
            skip_fusion=True,
        )

        for name, _ in partitioned_module.named_children():
            submodule = getattr(partitioned_module, name)
            if (
                not isinstance(submodule, torch.fx.graph_module.GraphModule)
                or "_run_on_acc" not in name
            ):
                continue
            partitioner = ResourcePartitioner(
                submodule,
                submodule_name=name,
                cpu_memory_budget=2 * 1024 * 1024 * 1024,
            )
            subgraphs = partitioner.put_nodes_into_subgraphs()
            new_subgraphs = []
            current_subgraph = []
            # Split the subgraph into two subgraphs by the ReLU node, which breaks the fusion group.
            for node in subgraphs[0].nodes:
                if node.op == "call_function" and node.target == aten.relu.default:
                    new_subgraphs.append(Subgraph(is_acc=True, nodes=current_subgraph))
                    current_subgraph = []
                current_subgraph.append(node)
            if current_subgraph:
                new_subgraphs.append(Subgraph(is_acc=True, nodes=current_subgraph))

            leaf_node = partitioner.get_leaf_node(new_subgraphs[0].nodes)
            broken_fusion = partitioner.step_if_break_fusion(
                new_subgraphs,
                leaf_node,
                set(new_subgraphs[0].nodes),
                set(new_subgraphs[1].nodes),
            )
            # The fusion was broken
            assert broken_fusion

            # The fusion should be fixed after the step
            partitioner._verify_all_fusion_nodes_in_same_subgraph(new_subgraphs)

            break


if __name__ == "__main__":
    run_tests()
