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
            truncate_double=True,
            min_block_size=1,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
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
            truncate_double=True,
            min_block_size=1,
            use_python_runtime=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
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
            truncate_double=False,
            min_block_size=1,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
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
            truncate_double=False,
            min_block_size=1,
            use_python_runtime=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
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


@unittest.skipIf(
    torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8
    or (
        torch.cuda.get_device_properties(torch.cuda.current_device()).major == 8
        and torch.cuda.get_device_properties(torch.cuda.current_device()).major == 7
    ),
    "Platform does not have BF16 support",
)
class TestBF16Support(TestCase):
    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "Torch-TensorRT Runtime is not available",
    )
    def test_bf16_cpp(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = self.conv(x)
                out = self.relu(out)
                return out

        in_tensor = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.bfloat16)
        mod = MyModule().to(torch.device("cuda")).to(torch.bfloat16)

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            enabled_precisions={torch.float, torch.bfloat16, torch.half},
            min_block_size=1,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            delta=3e-2,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    def test_bf16_py(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                out = self.conv(x)
                out = self.relu(out)
                return out

        in_tensor = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.bfloat16)
        mod = MyModule().to(torch.device("cuda")).to(torch.bfloat16)

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            enabled_precisions={torch.float, torch.bfloat16, torch.half},
            min_block_size=1,
            use_python_runtime=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            delta=3e-2,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    def test_bf16_torch_compile(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x):
                return self.linear(x)

        device = torch.device("cuda", 0)
        mod = MyModule().eval().to(device).bfloat16()
        inputs = [torch.randn((128, 20), dtype=torch.bfloat16, device=device)]

        with torch.inference_mode():
            trt_mod = torch_tensorrt.compile(
                mod,
                ir="torch_compile",
                inputs=inputs,
                enabled_precisions={torch.bfloat16},
                debug=True,
                min_block_size=1,
                device=device,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )

            torch_model_results = mod(*inputs)
            optimized_model_results = trt_mod(*inputs)

            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                delta=3e-2,
                msg=f"Torch outputs and TRT outputs don't match close enough.",
            )
