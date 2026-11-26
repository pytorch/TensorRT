# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from unittest import mock

import tensorrt as trt
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl
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

    def _quantize_node(self, users):
        """A quantize node whose output goes to the given ops, each with an output of the
        given shape."""
        graph = torch.fx.Graph()
        node = graph.call_function(
            torch.ops.aten.mul.Tensor, (graph.placeholder("w"), 1)
        )
        for op, shape in users:
            user = graph.call_function(op, (node, shape))
            user.meta["val"] = torch.empty(shape, device="meta")
        return node

    def _int4_weight(
        self, weight, amax, narrow_range=True, users=(), converter=False, **settings
    ):
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network, CompilationSettings(**settings))
        if users:
            ctx.current_node = self._quantize_node(users)
        if converter:
            args = (weight, amax, 4, 0)
            kwargs = {"unsigned": False, "narrow_range": narrow_range}
            if converter == "args":
                args, kwargs = args + tuple(kwargs.values()), {}
            aten_ops_converters.aten_ops_quantize_op(
                ctx, torch.ops.aten.mul.Tensor, args, kwargs, "q"
            )
        else:
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
        constant.__class__ = trt.IConstantLayer
        self.int4_shape = tuple(constant.shape)
        self.int4_layers = layers
        return qdata, ctx.weight_refit_map["q_scale CONSTANT"]

    def test_int4_weight_tiny_amax_quantizes_to_0(self):
        # ModelOpt quantizes a block with an amax below 2**-24 to 0, and one just above
        # it as usual. In float32 neither scale rounds to 0.
        weight = torch.zeros(3, 128)
        weight[:, 0] = torch.tensor([2**-25, 2**-23, 1.0])
        amax = weight.abs().amax(dim=1)
        qdata, scale = self._int4_weight(weight, amax)
        torch.testing.assert_close(
            scale.flatten(), torch.tensor([1.0, 2**-23 / 7, 1 / 7])
        )
        self.assertEqual(qdata[:, 0].tolist(), [0, 7, 7])

    def test_int4_weight_dequantizes_in_float32_in_a_refittable_engine(self):
        # Many refittable engines fail to build with a 16-bit INT4 dequantize.
        weight = torch.randn(2, 128, dtype=torch.float16)
        amax = weight.float().abs().amax(dim=1)
        _, scale = self._int4_weight(weight, amax, immutable_weights=False)
        self.assertEqual(scale.dtype, torch.float32)
        (dequantize,) = [
            l for l in self.int4_layers if l.type == trt.LayerType.DEQUANTIZE
        ]
        self.assertEqual(dequantize.get_output(0).dtype, trt.float32)
        self.assertIn(trt.LayerType.CAST, [l.type for l in self.int4_layers])

    def test_int4_weight_clamps_to_the_signed_range(self):
        # Weights past amax clamp to 7, and to -8 only without narrow_range.
        weight = torch.zeros(1, 128)
        weight[0, :4] = torch.tensor([2.0, -2.0, -8 / 7, 1.0])
        amax = torch.tensor([1.0])
        for narrow_range, low in ((True, -7), (False, -8)):
            qdata, _ = self._int4_weight(weight, amax, narrow_range)
            self.assertEqual(qdata[0, :4].tolist(), [7, low, low, 7])

    def test_int4_weight_converter_passes_narrow_range(self):
        # ModelOpt's INT4 configs quantize with narrow_range=False, which keeps -8.
        # A graph traced without torch.export can pass the flags as keywords.
        if not hasattr(aten_ops_converters, "aten_ops_quantize_op"):
            self.skipTest("The quantize_op converter needs ModelOpt")
        weight = torch.zeros(1, 128)
        weight[0, 0] = -8 / 7
        for converter in ("args", "kwargs"):
            for narrow_range, low in ((True, -7), (False, -8)):
                qdata, _ = self._int4_weight(
                    weight, torch.tensor([1.0]), narrow_range, converter=converter
                )
                self.assertEqual(qdata[0, 0].item(), low)

    def test_int4_weight_rounds_like_modelopt(self):
        # ModelOpt rounds w * (7 / amax). Some ties round the other way in (w * 7) /
        # amax, and normally distributed weights hit them.
        generator = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weight = torch.randn(8192, 128, generator=generator).to(dtype)
                amax = weight.float().abs().amax(dim=1)
                qdata, _ = self._int4_weight(weight, amax)
                scale = torch.tensor(7.0) / amax.reshape(-1, 1)
                expected = (weight.float() * scale).round().to(torch.int8)
                self.assertEqual(qdata, expected)

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
        # A refittable engine dequantizes in float32, which keeps that scale.
        qdata, scale = self._int4_weight(weight, amax, immutable_weights=False)
        torch.testing.assert_close(scale.flatten(), amax / 7)
        self.assertEqual(qdata[:, 0].tolist(), [7, 7])

    def test_int4_weight_near_the_float16_max_dequantizes_in_float32(self):
        # 65504 / 7 rounds up in float16, and 7 times that is infinity.
        weight = torch.zeros(1, 128, dtype=torch.float16)
        weight[0, 0] = 65504
        _, scale = self._int4_weight(weight, torch.tensor([65504.0]))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertIn(trt.LayerType.CAST, [l.type for l in self.int4_layers])

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
        # Reshapes that disagree, or any other user, keep the block-row shape. So
        # does a reshape that splitting the graph puts in another engine, where the
        # quantize's engine passes the weight on as an output.
        for users in (
            [(reshape, [512, 256]), (reshape, [256, 512])],
            [(reshape, [512, 256]), (torch.ops.aten.repeat.default, [2, 1])],
        ):
            self._int4_weight(weight, amax, users=users)
            self.assertEqual(self.int4_shape, (1024, 128))
        node = self._quantize_node([(reshape, [512, 256])])
        node.graph.output(node)
        self.assertEqual(
            impl.quantize.int4_weight_shape(node, weight, amax), (1024, 128)
        )
        # A conv weight, which reshapes back to 4 dims, runs in PyTorch.
        node = self._quantize_node([(view, [64, 2, 8, 128])])
        self.assertIsNone(impl.quantize.int4_weight_shape(node, weight, amax))

    def test_int4_weight_with_an_amax_per_column_runs_in_pytorch(self):
        # With a square weight, an amax per column has one value per row too.
        weight = torch.randn(128, 128)
        amax = weight.abs().amax(dim=0, keepdim=True)
        self.assertIsNone(impl.quantize.int4_weight_shape(None, weight, amax))

    def test_int4_weight_with_an_odd_last_dim_runs_in_pytorch(self):
        # A per-channel amax adds no reshape, so the constant keeps the weight's shape.
        weight = torch.randn(4, 127)
        amax = weight.abs().amax(dim=1, keepdim=True)
        self.assertIsNone(impl.quantize.int4_weight_shape(None, weight, amax))

    def test_int4_quantize_of_a_run_time_tensor_runs_in_pytorch(self):
        # Under torch.compile, ModelOpt's INT4 weight reaches the quantize as a node.
        graph = torch.fx.Graph()
        node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            (graph.placeholder("w"), graph.placeholder("amax"), 4, 0),
        )
        self.assertFalse(aten_ops_converters.quantize_op_validator(node))

    def test_int4_quantize_on_tensorrt_rtx_for_turing_runs_in_pytorch(self):
        module = torch.nn.Module()
        module.register_buffer("weight", torch.randn(4, 128))
        graph = torch.fx.Graph()
        node = graph.call_function(
            torch.ops.aten.mul.Tensor, (graph.get_attr("weight"), torch.ones(4), 4, 0)
        )
        graph.output(node)
        torch.fx.GraphModule(module, graph)
        for turing in (False, True):
            with mock.patch.object(
                aten_ops_converters, "trt_rtx_targets_turing", return_value=turing
            ):
                self.assertEqual(
                    aten_ops_converters.quantize_op_validator(node), not turing
                )

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

    def test_groupwise_int8_outside_int4_range_runs_in_pytorch(self):
        # Packing these values into 4 bits would silently change the weights.
        def supported(*values, block_size=(1, 16)):
            qdata = torch.zeros(64, 256, dtype=torch.int8)
            qdata.view(-1)[: len(values)] = torch.tensor(values, dtype=torch.int8)
            module = torch.nn.Module()
            module.register_buffer("qdata", qdata)
            graph = torch.fx.Graph()
            node = graph.call_function(
                torch.ops.aten.mul.Tensor, (graph.get_attr("qdata"), list(block_size))
            )
            graph.output(node)
            torch.fx.GraphModule(module, graph)
            # torch.compile partitions in fake mode.
            with FakeTensorMode(allow_non_fake_inputs=True):
                return aten_ops_converters.torchao_dequantize_affine_validator(node)

        for values in (range(-32, 32), [8], [-9]):
            self.assertFalse(supported(*values))
        self.assertTrue(supported(-8, 7))
        # Per-row int8 weights are not packed into INT4.
        self.assertTrue(supported(-128, 127, block_size=(1, 256)))
        # TensorRT-RTX cannot dequantize INT4 into a matmul on Turing.
        with mock.patch.object(
            aten_ops_converters, "trt_rtx_targets_turing", return_value=True
        ):
            self.assertFalse(supported(-8, 7))
            self.assertTrue(supported(-128, 127, block_size=(1, 256)))

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
