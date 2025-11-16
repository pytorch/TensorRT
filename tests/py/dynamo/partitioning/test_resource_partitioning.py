from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes import post_lowering, pre_export_lowering
from torch_tensorrt.dynamo.partitioning._resource_partitioner import resource_partition


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
        }
        settings = CompilationSettings(**compilation_options)
        with torchtrt.dynamo.Debugger(
            log_level="debug",
            logging_dir="/home/profile/logging/moe",
            engine_builder_monitor=False,
        ):

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
                gm, partitioned_module, cpu_memory_budget=2 * 1024 * 1024 * 1024  # 2GB,
            )

            self.assertEqual(
                len(list[Any](partitioned_module.named_children())),
                2,
                "The graph should have 2 subgraphs",
            )


if __name__ == "__main__":
    run_tests()
