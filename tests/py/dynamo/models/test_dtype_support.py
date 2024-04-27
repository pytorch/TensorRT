# type: ignore

import math
import unittest

import torch
import torch_tensorrt
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class Test64BitSupport(TestCase):

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_truncate_f64_weights_cpp(self):
        class f64_weight_module(nn.Module):
            def __init__(self, h, w):
                super().__init__()
                factory_kwargs = {"dtype": torch.float64}
                self.weight = Parameter(torch.empty((h, w), **factory_kwargs))
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            def forward(self, x):
                return x + self.weight

        h, w = 4, 4
        in_tensor = torch.randn((h, w), dtype=torch.float64, device="cuda")
        mod = f64_weight_module(h, w).to("cuda")

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            truncate_long_and_double=True,
            output_format="fx",
            min_block_size=1,
            use_python_runtime=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    def test_truncate_f64_weights_py(self):
        class f64_weight_module(nn.Module):
            def __init__(self, h, w):
                super().__init__()
                factory_kwargs = {"dtype": torch.float64}
                self.weight = Parameter(torch.empty((h, w), **factory_kwargs))
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            def forward(self, x):
                return x + self.weight

        h, w = 4, 4
        in_tensor = torch.randn((h, w), dtype=torch.float64, device="cuda")
        mod = f64_weight_module(h, w).to("cuda")

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            truncate_long_and_double=True,
            output_format="fx",
            min_block_size=1,
            use_python_runtime=True,
        )

        torch_model_results = mod(in_tensor)
        with torch_tensorrt.logging.debug():
            optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_native_i64_cpp(self):
        class i64_module(nn.Module):
            def __init__(self, h, w):
                super().__init__()
                self.const_tensor = Parameter(
                    torch.randint(0, 100, (h, w), dtype=torch.int64),
                    requires_grad=False,
                )

            def forward(self, x):
                return (x + self.const_tensor) * 10

        h, w = 4, 4
        in_tensor = torch.randint(0, 100, (h, w), dtype=torch.int64, device="cuda")
        mod = i64_module(h, w).to("cuda")

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            truncate_long_and_double=False,
            output_format="fx",
            min_block_size=1,
            use_python_runtime=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    def test_native_i64_py(self):
        class i64_module(nn.Module):
            def __init__(self, h, w):
                super().__init__()
                self.const_tensor = Parameter(
                    torch.randint(0, 100, (h, w), dtype=torch.int64),
                    requires_grad=False,
                )

            def forward(self, x):
                return (x + self.const_tensor) * 10

        h, w = 4, 4
        in_tensor = torch.randint(0, 100, (h, w), dtype=torch.int64, device="cuda")
        mod = i64_module(h, w).to("cuda")

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            truncate_long_and_double=False,
            output_format="fx",
            min_block_size=1,
            use_python_runtime=True,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )
