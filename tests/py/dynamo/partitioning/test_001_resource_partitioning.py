from typing import Any, List

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
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
from torch_tensorrt.dynamo.partitioning._atomic_subgraphs import (
    ATOMIC_SUBGRAPHS,
    register_atomic_subgraph,
)
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    ResourcePartitioner,
    resource_partition,
)


class TestResourcePartitioning(TestCase):
    def test_resource_partitioning(self):
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1024, 4096, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(4096)
                self.conv2 = nn.Conv2d(4096, 1024, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(1024)
                self.fc1 = nn.Linear(1024 * 56 * 56, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                return self.fc1(x)

        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 1024, 224, 224)).to("cuda")]

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

        partitioned_module = resource_partition(
            partitioned_module,
            cpu_memory_budget=0.89 * 1024 * 1024 * 1024
            + psutil.Process().memory_info().rss,  # 0.89GB + current memory usage,
        )

        self.assertEqual(
            len(list[Any](partitioned_module.named_children())),
            2,
            "The graph should have 2 subgraphs",
        )

        torch._dynamo.reset()

    def test_resource_partitioning_with_capability_partitioning(self):
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1024, 4096, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(4096)
                self.conv2 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(4096)

                self.conv3 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(4096)
                self.conv4 = nn.Conv2d(4096, 1024, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(1024)

                self.fc1 = nn.Linear(1024 * 56 * 56, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv3(x)
                x = self.bn3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = self.bn4(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                return self.fc1(x)

        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 1024, 224, 224)).to("cuda")]

        enabled_precisions = {torch.float}
        use_python_runtime = False

        exp_program = torch.export.export(model, tuple(inputs))

        compilation_options = {
            "use_python_runtime": use_python_runtime,
            "enabled_precisions": enabled_precisions,
            "min_block_size": 1,
            "immutable_weights": True,
            "reuse_cached_engines": False,
            "torch_executed_ops": {"torch.ops.aten.max_pool2d.default"},
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

        partitioned_module = resource_partition(
            partitioned_module,
            cpu_memory_budget=0.39 * 1024 * 1024 * 1024
            + psutil.Process().memory_info().rss,  # 0.39GB + current memory usage,
        )

        assert (
            len(
                [
                    name
                    for name, _ in partitioned_module.named_children()
                    if "_run_on_acc" in name
                ]
            )
            == 5
        ), "The graph should have 5 accelerated subgraphs"
        assert (
            len(
                [
                    name
                    for name, _ in partitioned_module.named_children()
                    if "_run_on_gpu" in name
                ]
            )
            == 2
        ), "The graph should have 2 non-accelerated subgraphs"

        torch._dynamo.reset()

    def test_resource_partitioning_with_capability_partitioning_and_atomic_subgraphs(
        self,
    ):
        """
        After defining the atomic subgraphs, the resource partitioner will not be able to find valid partition in the subgraph.
        So there should only be 3 accelerated subgraphs and 2 non-accelerated subgraphs.
        """

        @register_atomic_subgraph(init_args=(), is_core_aten=True)
        class ReLUConv(nn.Module):
            def forward(
                self,
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                stride: List[int],
                padding: List[int],
                dilation: List[int],
                transposed: bool,
                output_padding: List[int],
                groups: int,
            ) -> torch.Tensor:
                x = aten.relu.default(x)
                x = aten.convolution.default(
                    x,
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups,
                )
                return x

        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1024, 4096, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(4096)
                self.conv2 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(4096)

                self.conv3 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(4096)
                self.conv4 = nn.Conv2d(4096, 1024, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(1024)

                self.fc1 = nn.Linear(1024 * 56 * 56, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv3(x)
                x = self.bn3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = self.bn4(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                return self.fc1(x)

        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 1024, 224, 224)).to("cuda")]

        enabled_precisions = {torch.float}
        use_python_runtime = False

        exp_program = torch.export.export(model, tuple(inputs))

        compilation_options = {
            "use_python_runtime": use_python_runtime,
            "enabled_precisions": enabled_precisions,
            "min_block_size": 1,
            "immutable_weights": True,
            "reuse_cached_engines": False,
            "torch_executed_ops": {"torch.ops.aten.max_pool2d.default"},
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

        partitioned_module = resource_partition(
            partitioned_module,
            cpu_memory_budget=0.39 * 1024 * 1024 * 1024
            + psutil.Process().memory_info().rss,  # 0.39GB + current memory usage,
        )

        assert (
            len(
                [
                    name
                    for name, _ in partitioned_module.named_children()
                    if "_run_on_acc" in name
                ]
            )
            == 3
        ), "The graph should have 3 accelerated subgraphs"
        assert (
            len(
                [
                    name
                    for name, _ in partitioned_module.named_children()
                    if "_run_on_gpu" in name
                ]
            )
            == 2
        ), "The graph should have 2 non-accelerated subgraphs"

        ATOMIC_SUBGRAPHS.remove((ReLUConv, (), True))

        torch._dynamo.reset()

    def test_resource_partitioning_with_global_capability_partitioning(self):
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1024, 4096, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(4096)
                self.conv2 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(4096)

                self.conv3 = nn.Conv2d(4096, 4096, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(4096)
                self.conv4 = nn.Conv2d(4096, 1024, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(1024)

                self.fc1 = nn.Linear(1024 * 56 * 56, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv3(x)
                x = self.bn3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = self.bn4(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                return self.fc1(x)

        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 1024, 224, 224)).to("cuda")]

        enabled_precisions = {torch.float}
        use_python_runtime = False

        exp_program = torch.export.export(model, tuple(inputs))

        compilation_options = {
            "use_python_runtime": use_python_runtime,
            "enabled_precisions": enabled_precisions,
            "min_block_size": 1,
            "immutable_weights": True,
            "reuse_cached_engines": False,
            "torch_executed_ops": {"torch.ops.aten.max_pool2d.default"},
            "enable_resource_partitioning": True,
        }
        settings = CompilationSettings(**compilation_options)

        exported_program = pre_export_lowering(exp_program, settings)
        exported_program = exported_program.run_decompositions(
            get_decompositions(False)
        )

        gm = exported_program.module()
        gm = post_lowering(gm, settings)

        partitioned_module, supported_ops = partitioning.global_partition(
            gm,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
            require_full_compilation=settings.require_full_compilation,
        )

        partitioned_module = resource_partition(
            partitioned_module,
            cpu_memory_budget=0.39 * 1024 * 1024 * 1024
            + psutil.Process().memory_info().rss,  # 0.39GB + current memory usage,
        )

        assert (
            len(
                [
                    name
                    for name, _ in partitioned_module.named_children()
                    if "_run_on_acc" in name
                ]
            )
            == 5
        ), "The graph should have 5 accelerated subgraphs"

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
