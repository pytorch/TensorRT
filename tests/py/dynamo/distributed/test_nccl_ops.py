import os
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from conversion.harness import DispatchTestCase
from distributed_utils import set_environment_variables_pytest
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._enums import Platform
from torch_tensorrt.dynamo.utils import is_platform_supported_for_trtllm


class DistributedGatherModel(nn.Module):
    def __init__(self, input_dim, world_size, group_name):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.world_size = world_size
        self.group_name = group_name

    def forward(self, x):
        x = self.fc(x)
        gathered_tensor = torch.ops._c10d_functional.all_gather_into_tensor(
            x, self.world_size, self.group_name
        )
        return torch.ops._c10d_functional.wait_tensor(gathered_tensor)


class DistributedReduceScatterModel(nn.Module):
    def __init__(self, input_dim, world_size, group_name):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.world_size = world_size
        self.group_name = group_name

    def forward(self, x):
        x = self.fc(x)
        out = torch.ops._c10d_functional.reduce_scatter_tensor(
            x, "sum", self.world_size, self.group_name
        )
        return torch.ops._c10d_functional.wait_tensor(out)


platform_str = str(Platform.current_platform()).lower()


class TestNcclOpsConverter(DispatchTestCase):
    @unittest.skipIf(
        not is_platform_supported_for_trtllm(platform_str),
        "Skipped on Windows, Jetson: NCCL backend is not supported.",
    )
    @classmethod
    def setUpClass(cls):
        set_environment_variables_pytest()
        print("USE_TRTLLM_PLUGINS =", os.environ.get("USE_TRTLLM_PLUGINS"))
        cls.world_size = 1
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        cls.group = dist.new_group(ranks=[0])
        cls.group_name = cls.group.group_name

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    @parameterized.expand([8])
    def test_nccl_ops_gather(self, linear_layer_dim):
        inputs = [torch.randn(1, linear_layer_dim).to("cuda")]
        self.run_test(
            DistributedGatherModel(
                linear_layer_dim, self.world_size, self.group_name
            ).cuda(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand([8])
    def test_nccl_ops_scatter(self, linear_layer_dim):
        inputs = [torch.zeros(1, linear_layer_dim).to("cuda")]
        self.run_test(
            DistributedReduceScatterModel(
                linear_layer_dim, self.world_size, self.group_name
            ).cuda(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
