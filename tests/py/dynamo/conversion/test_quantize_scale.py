# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import tensorrt as trt
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestQuantizeScale(TestCase):
    def test_int8_low_precision_amax_gives_float32_scale(self):
        # In float16, amax / 127 rounds a small scale down, or to 0.
        # TensorRT builds an INT8 quantize with a run-time scale only for bfloat16.
        def network(amax_from_input):
            network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            ctx = ConversionContext(network, CompilationSettings())
            dtype = trt.bfloat16 if amax_from_input else trt.float16
            x = network.add_input("x", dtype, (4,))
            amax = torch.tensor(1e-5, dtype=torch.float16)
            if amax_from_input:
                amax = network.add_input("amax", dtype, ())
            impl.quantize.quantize(
                ctx, torch.ops.aten.mul.Tensor, None, "q", x, amax, 8, 0
            )
            # ctx keeps the constant weights alive.
            return ctx, [network.get_layer(i) for i in range(network.num_layers)]

        ctx, layers = network(False)
        (scale,) = [l for l in layers if l.name == "q_scale"]
        scale.__class__ = trt.IConstantLayer
        expected = torch.tensor(1e-5, dtype=torch.float16).item() / 127
        torch.testing.assert_close(scale.weights.item(), expected, rtol=1e-6, atol=0)
        (div,) = [l for l in network(True)[1] if l.type == trt.LayerType.ELEMENTWISE]
        self.assertEqual(div.get_input(0).dtype, trt.float32)

    def test_fp8_run_time_amax_with_float64_default_dtype(self):
        # TensorRT has no float64, so the constants must not follow the default dtype.
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network, CompilationSettings())
        x = network.add_input("x", trt.float32, (4,))
        amax = network.add_input("amax", trt.float32, ())
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            impl.quantize.quantize(
                ctx, torch.ops.aten.mul.Tensor, None, "q", x, amax, 8, 4
            )
        finally:
            torch.set_default_dtype(default_dtype)


if __name__ == "__main__":
    run_tests()
