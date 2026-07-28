import os
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from conversion.harness import DispatchTestCase
from distributed_utils import set_environment_variables_pytest
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._features import ENABLED_FEATURES


def is_distributed_nccl_available():
    """
    Check if torch.distributed with NCCL backend is available.

    Note: torch.distributed is available on Windows but NCCL backend is not.
    NCCL (NVIDIA Collective Communications Library) is Linux/Unix only.
    This function returns False on Windows, Jetson, and other platforms
    where NCCL backend is not supported.
    """
    try:
        import torch.distributed as dist

        # Check if NCCL backend is available (False on Windows, since its gloo. For ORIN some torch distribution it is available
        return dist.is_nccl_available()
    except (ImportError, AttributeError):
        return False


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


class TestNcclOpsConverter(DispatchTestCase):
    # 1. Skip if NCCL backend is not available (e.g., Windows, Jetson) - hard requirement
    # 2. Skip if TRTLLM is unavailable (e.g., CUDA 13) - no converters registered
    @unittest.skipIf(
        not is_distributed_nccl_available(),
        "Skipped: NCCL backend is not available (Windows/Jetson Orin not supported).",
    )
    @unittest.skipIf(
        not ENABLED_FEATURES.trtllm_for_nccl,
        "Skipped: TensorRT-LLM plugin for NCCL is not available (e.g., CUDA 13).",
    )
    @classmethod
    def setUpClass(cls):
        set_environment_variables_pytest()
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
