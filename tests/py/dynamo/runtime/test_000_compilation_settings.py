import unittest
from importlib import metadata

import tensorrt as trt
import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo.utils import is_tegra_platform

from packaging.version import Version

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class TestEnableTRTFlags(TestCase):
    def test_toggle_build_args(self):
        class AddSoftmax(torch.nn.Module):
            def forward(self, x):
                x = 3 * x
                y = x + 1
                return torch.softmax(y, 0)

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddSoftmax())

        # Validate that the results between Torch and Torch-TRT are similar
        # Enable multiple TRT build arguments
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            disable_tf32=True,
            sparse_weights=True,
            refit=True,
            num_avg_timing_iters=5,
            workspace_size=1 << 10,
            truncate_double=True,
        )

        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"AddSoftmax TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    @unittest.skipIf(
        is_tegra_platform() and is_tensorrt_version_supported("10.8"),
        "DLA is not supported on Jetson platform starting TRT 10.8",
    )
    def test_dla_args(self):
        class AddSoftmax(torch.nn.Module):
            def forward(self, x):
                x = 3 * x
                y = x + 1
                return torch.softmax(y, 0)

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddSoftmax())

        # Validate that the results between Torch and Torch-TRT are similar
        # Enable multiple TRT build arguments
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            device=torch_tensorrt.Device("dla:0", allow_gpu_fallback=True),
            pass_through_build_failures=True,
            dla_sram_size=1048577,
            dla_local_dram_size=1073741825,
            dla_global_dram_size=536870913,
        )

        # DLA is not present on the active machine
        with self.assertRaises(RuntimeError):
            optimized_model(*inputs).detach().cpu()

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
