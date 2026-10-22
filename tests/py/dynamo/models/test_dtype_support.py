# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

import math
import unittest

import pytest
import torch
import torch_tensorrt
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias


@pytest.mark.critical
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
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        with torch_tensorrt.logging.debug():
            optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
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
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
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
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )


def _build_engine(input_dtype, add_layer):
    # Built by hand: the _to_copy converter does not accept a cast to uint8 or FP8, and
    # TensorRT does not let a cast layer read an FP8 input.
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    x = network.add_input("x", input_dtype, (4, 4))
    y = add_layer(network, x).get_output(0)
    y.name = "y"
    network.mark_output(y)
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    assert engine is not None
    return TorchTensorRTModule(bytes(engine), ["x"], ["y"])


def _unit_scale(network):
    return network.add_constant(
        (), trt.Weights(torch.ones((), dtype=torch.float32).numpy())
    ).get_output(0)


@pytest.mark.critical
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT Runtime is not available",
)
@unittest.skipIf(
    torch.cuda.get_device_capability() < (8, 9),
    "FP8 requires compute capability 8.9 or later",
)
class TestFP8Support(TestCase):
    def test_fp8_output(self):
        trt_mod = _build_engine(
            trt.float32,
            lambda network, x: network.add_quantize(
                x, _unit_scale(network), trt.DataType.FP8
            ),
        )
        in_tensor = torch.rand(4, 4, device="cuda") * 10
        out = trt_mod(in_tensor)

        self.assertEqual(out.dtype, torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(out.float(), in_tensor.to(torch.float8_e4m3fn).float())
        )

    def test_fp8_input(self):
        trt_mod = _build_engine(
            trt.fp8,
            lambda network, x: network.add_dequantize(
                x, _unit_scale(network), trt.float32
            ),
        )
        in_tensor = (torch.rand(4, 4, device="cuda") * 10).to(torch.float8_e4m3fn)
        out = trt_mod(in_tensor)

        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, in_tensor.float()))


@pytest.mark.critical
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT Runtime is not available",
)
class TestUInt8Support(TestCase):
    def test_uint8_input(self):
        class ImageModule(nn.Module):
            def forward(self, image):
                return (image.float() / 255).permute(2, 0, 1)

        in_tensor = torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8, device="cuda")
        mod = ImageModule().to("cuda")

        exp_mod = torch.export.export(mod, (in_tensor,))
        trt_mod = torch_tensorrt.dynamo.compile(
            exp_mod,
            inputs=[in_tensor],
            pass_through_build_failures=True,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"Torch outputs and TRT outputs don't match close enough.",
        )

    def test_uint8_output(self):
        trt_mod = _build_engine(
            trt.float32, lambda network, x: network.add_cast(x, trt.uint8)
        )
        in_tensor = torch.randint(0, 256, (4, 4), device="cuda").float()
        out = trt_mod(in_tensor)

        self.assertEqual(out.dtype, torch.uint8)
        self.assertTrue(torch.equal(out, in_tensor.to(torch.uint8)))


@pytest.mark.critical
@unittest.skipIf(
    torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8
    or (
        torch.cuda.get_device_properties(torch.cuda.current_device()).major == 8
        and torch.cuda.get_device_properties(torch.cuda.current_device()).minor == 7
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
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
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
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        torch_model_results = mod(in_tensor)
        optimized_model_results = trt_mod(in_tensor)
        assert torch_model_results.dtype == optimized_model_results.dtype
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
                min_block_size=1,
                device=device,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )

            torch_model_results = mod(*inputs)
            optimized_model_results = trt_mod(*inputs)
            assert torch_model_results.dtype == optimized_model_results.dtype
            max_diff = float(
                torch.max(torch.abs(optimized_model_results - torch_model_results))
            )
            self.assertAlmostEqual(
                max_diff,
                0,
                delta=3e-2,
                msg=f"Torch outputs and TRT outputs don't match close enough.",
            )
