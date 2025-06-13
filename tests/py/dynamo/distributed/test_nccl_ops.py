import os

import torch
import torch.distributed as dist
import torch.nn as nn
from distributed_utils import set_environment_variables_pytest
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

set_environment_variables_pytest()
dist.init_process_group(backend="nccl", init_method="env://")
group = dist.new_group(ranks=[0])
group_name = group.group_name
world_size = 1

from conversion.harness import DispatchTestCase


class TestGatherNcclOpsConverter(DispatchTestCase):
    @parameterized.expand([8])
    def test_nccl_ops(self, linear_layer_dim):
        class DistributedGatherModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, input_dim)

            def forward(self, x):
                x = self.fc(x)
                gathered_tensor = torch.ops._c10d_functional.all_gather_into_tensor(
                    x, world_size, group_name
                )
                gathered_tensor = torch.ops._c10d_functional.wait_tensor(
                    gathered_tensor
                )
                return gathered_tensor

        inputs = [torch.randn(1, linear_layer_dim).to("cuda")]
        self.run_test(
            DistributedGatherModel(linear_layer_dim).cuda(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand([8])
    def test_nccl_ops_scatter(self, linear_layer_dim):

        class DistributedReduceScatterModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, input_dim)

            def forward(self, x):
                x = self.fc(x)
                scatter_reduce_tensor = (
                    torch.ops._c10d_functional.reduce_scatter_tensor(
                        x, "sum", world_size, group_name
                    )
                )
                scatter_reduce_tensor = torch.ops._c10d_functional.wait_tensor(
                    scatter_reduce_tensor
                )
                return scatter_reduce_tensor

        inputs = [torch.zeros(1, linear_layer_dim).to("cuda")]

        self.run_test(
            DistributedReduceScatterModel(linear_layer_dim).cuda(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
