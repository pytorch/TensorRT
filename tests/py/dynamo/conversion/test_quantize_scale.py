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

    def test_fp8_zero_amax_gives_a_scale_of_one_over_448(self):
        # An FP8 scale of 0 zeroes the input or gives NaN, so an amax of 0 means 1.
        def network(amax_from_input):
            network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            ctx = ConversionContext(network, CompilationSettings())
            x = network.add_input("x", trt.float32, (4,))
            amax = torch.tensor(0.0)
            if amax_from_input:
                amax = network.add_input("amax", trt.float32, ())
            impl.quantize.quantize(
                ctx, torch.ops.aten.mul.Tensor, None, "q", x, amax, 8, 4
            )
            # ctx keeps the constant weights alive.
            return ctx, [network.get_layer(i) for i in range(network.num_layers)]

        ctx, layers = network(False)
        (scale,) = [l for l in layers if l.name == "q_scale"]
        scale.__class__ = trt.IConstantLayer
        torch.testing.assert_close(scale.weights.item(), 1 / 448, rtol=1e-6, atol=0)
        layers = network(True)[1]
        self.assertTrue(any(l.type == trt.LayerType.SELECT for l in layers))

    def _int4_weight(self, weight, amax, narrow_range=True, users=()):
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network, CompilationSettings())
        if users:
            # A graph where the quantize output goes to the given ops, each with an
            # output of the given shape.
            graph = torch.fx.Graph()
            ctx.current_node = graph.call_function(
                torch.ops.aten.mul.Tensor, (graph.placeholder("w"), 1)
            )
            for op, shape in users:
                user = graph.call_function(op, (ctx.current_node, shape))
                user.meta["val"] = torch.empty(shape, device="meta")
        impl.quantize.quantize(
            ctx,
            torch.ops.aten.mul.Tensor,
            None,
            "q",
            weight,
            amax,
            4,
            0,
            narrow_range=narrow_range,
        )
        packed = ctx.weight_refit_map["q_int4_weight CONSTANT"]
        # Even columns are in the low nibble, odd columns in the high nibble.
        qdata = torch.stack(
            [torch.bitwise_left_shift(packed, 4) >> 4, packed >> 4], dim=-1
        ).flatten(-2)
        layers = [network.get_layer(i) for i in range(network.num_layers)]
        (constant,) = [l for l in layers if l.name == "q_int4_weight"]
        (scale,) = [l for l in layers if l.name == "q_scale"]
        scale.__class__ = trt.IConstantLayer
        constant.__class__ = trt.IConstantLayer
        self.int4_shape = tuple(constant.shape)
        self.int4_layers = layers
        return qdata, torch.from_numpy(scale.weights.copy())

    def test_int4_weight_tiny_amax_quantizes_to_0(self):
        # ModelOpt quantizes a block with an amax below 2**-24 to 0. In float32 the
        # scale of this one does not round to 0.
        weight = torch.zeros(2, 128)
        weight[0, 0] = 1e-10
        weight[1, 0] = 1.0
        amax = weight.abs().amax(dim=1)
        qdata, scale = self._int4_weight(weight, amax)
        torch.testing.assert_close(scale.flatten(), torch.tensor([1.0, 1 / 7]))
        self.assertEqual(qdata[:, 0].tolist(), [0, 7])

    def test_int4_weight_clamps_to_the_signed_range(self):
        # Weights past amax clamp to 7, and to -8 only without narrow_range.
        weight = torch.zeros(1, 128)
        weight[0, :4] = torch.tensor([2.0, -2.0, -8 / 7, 1.0])
        amax = torch.tensor([1.0])
        for narrow_range, low in ((True, -7), (False, -8)):
            qdata, _ = self._int4_weight(weight, amax, narrow_range)
            self.assertEqual(qdata[0, :4].tolist(), [7, low, low, 7])

    def test_int4_weight_scale_that_rounds_to_0_gives_a_scale_of_one(self):
        # amax / 7 rounds to 0 in float16, which would be a scale of 0.
        weight = torch.zeros(2, 128, dtype=torch.float16)
        weight[0, 0] = 1e-7
        weight[1, 0] = 1.0
        amax = weight.float().abs().amax(dim=1)
        qdata, scale = self._int4_weight(weight, amax)
        torch.testing.assert_close(
            scale.flatten(), torch.tensor([1.0, 1 / 7], dtype=torch.float16)
        )
        self.assertEqual(qdata[:, 0].tolist(), [0, 7])

    def test_int4_weight_dequantizes_in_the_weight_dtype(self):
        # TensorRT fuses the dequantize into the matmul only when its scale and output
        # have the weight's type, with no cast after it.
        weight = torch.randn(2, 128, dtype=torch.float16)
        _, scale = self._int4_weight(weight, weight.float().abs().amax(dim=1))
        self.assertEqual(scale.dtype, torch.float16)
        (dequantize,) = [
            l for l in self.int4_layers if l.type == trt.LayerType.DEQUANTIZE
        ]
        self.assertEqual(dequantize.get_output(0).dtype, trt.float16)
        self.assertNotIn(trt.LayerType.CAST, [l.type for l in self.int4_layers])

    def test_int4_weight_takes_the_shape_of_the_reshape_after_it(self):
        # ModelOpt passes a (512, 256) weight as 1024 rows of one 128 block each.
        weight = torch.randn(1024, 128)
        amax = weight.abs().amax(dim=1)
        reshape, view = torch.ops.aten.reshape.default, torch.ops.aten.view.default
        self._int4_weight(weight, amax, users=[(reshape, [512, 256])])
        self.assertEqual(self.int4_shape, (512, 256))
        # A layer that runs twice can give the quantize two reshapes.
        self._int4_weight(weight, amax, users=[(reshape, [512, 256])] * 2)
        self.assertEqual(self.int4_shape, (512, 256))
        # A conv weight reshapes back to 4 dims.
        with self.assertRaisesRegex(ValueError, "matmul weights only"):
            self._int4_weight(weight, amax, users=[(view, [64, 2, 8, 128])])
        # Any other user hides the weight's shape.
        with self.assertRaisesRegex(ValueError, "could not find the weight's shape"):
            self._int4_weight(
                weight,
                amax,
                users=[(reshape, [512, 256]), (torch.ops.aten.repeat.default, [2, 1])],
            )

    def test_int4_weight_rejects_an_amax_per_column(self):
        # With a square weight, an amax per column has one value per row too.
        weight = torch.randn(128, 128)
        with self.assertRaisesRegex(ValueError, "one amax per block row"):
            self._int4_weight(weight, weight.abs().amax(dim=0, keepdim=True))

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

    def test_groupwise_int8_outside_int4_range_is_rejected(self):
        # Packing these values into 4 bits would silently change the weights.
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network, CompilationSettings())
        scale = torch.rand(64, 16, dtype=torch.float32)

        def dequantize(*values):
            qdata = torch.zeros(64, 256, dtype=torch.int8)
            qdata.view(-1)[: len(values)] = torch.tensor(values, dtype=torch.int8)
            return impl.quantize.dequantize_affine(
                ctx,
                torch.ops.aten.mul.Tensor,
                None,
                "dq",
                qdata,
                [1, 16],
                scale,
                torch.float32,
            )

        for values in (range(-32, 32), [8], [-9]):
            with self.assertRaisesRegex(RuntimeError, "outside -8..7"):
                dequantize(*values)
        dequantize(-8, 7)

    def test_fp8_dequantize_keeps_its_default_output_dtype(self):
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network, CompilationSettings())
        x = network.add_input("x", trt.fp8, (4,))
        out = impl.quantize.dequantize_affine_float8(
            ctx, torch.ops.aten.mul.Tensor, None, "dq", x, torch.tensor(0.5)
        )
        self.assertEqual(out.dtype, trt.bfloat16)


if __name__ == "__main__":
    run_tests()
