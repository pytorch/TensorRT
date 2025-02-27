import os

import torch
import torch.distributed as dist
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


def set_environment_variables():
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500)
    os.environ["USE_TRTLLM_PLUGINS"] = "1"


set_environment_variables()
dist.init_process_group(backend="nccl", init_method="env://")
group = dist.new_group(ranks=[0])
group_name = group.group_name

from .harness import DispatchTestCase


class TestGatherNcclOpsConverter(DispatchTestCase):
    @parameterized.expand([(8)])
    def test_nccl_ops(self, linear_layer_dim):
        class DistributedGatherModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, input_dim)

            def forward(self, x):
                x = self.fc(x)
                world_size = 1
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
            fuse_distributed_ops=True,
        )

    # TODO: Look at this
    # @parameterized.expand(
    #     [
    #         (8)
    #     ]
    # )
    # def test_nccl_ops_scatter(self, linear_layer_dim):

    #     class DistributedReduceScatterModel(nn.Module):
    #         def __init__(self, input_dim):
    #             super().__init__()
    #         def forward(self, x):
    #             world_size = 1
    #             scatter_reduce_tensor = torch.ops._c10d_functional.reduce_scatter_tensor(x, "sum", world_size, group_name)
    #             scatter_reduce_tensor = torch.ops._c10d_functional.wait_tensor(scatter_reduce_tensor)
    #             return scatter_reduce_tensor
    #     inputs = [torch.zeros(1, linear_layer_dim).to("cuda")]

    #     self.run_test(
    #         DistributedReduceScatterModel(linear_layer_dim).cuda(),
    #         inputs,
    #         use_dynamo_tracer=True,
    #     )


if __name__ == "__main__":
    run_tests()
