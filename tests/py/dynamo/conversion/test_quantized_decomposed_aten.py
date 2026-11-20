# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import unittest
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

from .harness import DispatchTestCase, skip_if_trt_rtx_turing

if not impl.quantize.QUANTIZED_DECOMPOSED_QUANTIZE_OPS:
    pytest.skip(
        "torch.ao.quantization.fx._decomposed is not available",
        allow_module_level=True,
    )

qd = torch.ops.quantized_decomposed


def _random_inputs(dtype: torch.dtype, scale: float) -> torch.Tensor:
    # Values past both ends of the range.
    generator = torch.Generator().manual_seed(0)
    return (torch.randn(120, generator=generator) * 200 * scale).to(dtype)


def _halfway_inputs(dtype: torch.dtype, scale: float) -> torch.Tensor:
    # Halfway values, exact for a power-of-two scale.
    return ((torch.arange(-6, 6, dtype=torch.float32) + 0.5) * scale).to(dtype)


def _tie_tolerance(scale: float, out_dtype: torch.dtype = torch.float32) -> float:
    # When a dequantize follows a quantize in the same engine, TensorRT can multiply by the
    # scalar scale one unit in the last place off, which moves a value on a halfway point by
    # one quantization step. A power-of-two scale is exact. The result is also rounded to
    # out_dtype; 8 * eps covers that for the halfway inputs, which reach 7 steps at most.
    if math.frexp(scale)[0] == 0.5:
        return 0.0
    return scale * (1 + 8 * torch.finfo(out_dtype).eps)


class TestQuantizedDecomposedConverter(DispatchTestCase):
    def run_qdq_test(self, module, input_dtype, scale, out_dtype=None):
        # Only values on a halfway point may move by one quantization step.
        self.run_test(
            module,
            [_random_inputs(input_dtype, scale)],
            use_dynamo_tracer=True,
            rtol=0,
            atol=0,
        )
        self.run_test(
            module,
            [_halfway_inputs(input_dtype, scale)],
            use_dynamo_tracer=True,
            rtol=0,
            atol=_tie_tolerance(scale, out_dtype or torch.float32),
        )

    @parameterized.expand(
        [
            ("uint8_asymmetric", torch.uint8, 0.25, 117, 0, 255, torch.float32, None),
            ("int8_symmetric", torch.int8, 0.25, 0, -128, 127, torch.float32, None),
            # A scale that is not a power of two makes float16 and bfloat16 math inexact.
            (
                "float16_in_bfloat16_out",
                torch.uint8,
                0.0371,
                3,
                0,
                255,
                torch.float16,
                torch.bfloat16,
            ),
            (
                "bfloat16_in_float16_out",
                torch.int8,
                0.0371,
                -5,
                -127,
                127,
                torch.bfloat16,
                torch.float16,
            ),
        ]
    )
    def test_per_tensor(
        self, _, dtype, scale, zero_point, lo, hi, input_dtype, out_dtype
    ):
        if torch.bfloat16 in (input_dtype, out_dtype):
            skip_if_trt_rtx_turing(self, "bfloat16")

        class QDQ(nn.Module):
            def forward(self, x):
                q = qd.quantize_per_tensor(x, scale, zero_point, lo, hi, dtype)
                return qd.dequantize_per_tensor(
                    q, scale, zero_point, lo, hi, dtype, out_dtype=out_dtype
                )

        self.run_qdq_test(QDQ(), input_dtype, scale, out_dtype)

    @parameterized.expand(
        [
            ("uint8_float32_in", torch.uint8, 117, 0, 255, torch.float32),
            ("int8_float16_in", torch.int8, -7, -128, 127, torch.float16),
            ("int8_bfloat16_in", torch.int8, -7, -128, 127, torch.bfloat16),
        ]
    )
    def test_quantize_is_exact(self, _, dtype, zero_point, lo, hi, input_dtype):
        # A quantize whose result leaves the engine matches PyTorch exactly, also next to
        # halfway points, where dividing by the scale, or a reciprocal computed in float32
        # arithmetic, would not. The neighbors are float32 values, so only the float32 case
        # checks them: in float16 and bfloat16 they round back onto the halfway points.
        if input_dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")
        scale = 0.0371

        class Q(nn.Module):
            def forward(self, x):
                return qd.quantize_per_tensor(x, scale, zero_point, lo, hi, dtype)

        ties = (torch.arange(-100, 100) + 0.5) / torch.tensor(1.0 / scale)
        x = torch.cat(
            [
                ties,
                torch.nextafter(ties, ties + 1),
                torch.nextafter(ties, ties - 1),
                torch.randn(400) * 3,
            ]
        )
        self.run_test(Q(), [x.to(input_dtype)], use_dynamo_tracer=True, rtol=0, atol=0)

    def test_per_tensor_constant_tensor_qparams(self):
        class QDQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(0.0371))
                self.register_buffer("zero_point", torch.tensor(-7))

            def forward(self, x):
                q = qd.quantize_per_tensor.tensor(
                    x, self.scale, self.zero_point, -128, 127, torch.int8
                )
                return qd.dequantize_per_tensor.tensor(
                    q, self.scale, self.zero_point, -128, 127, torch.int8
                )

        self.run_qdq_test(QDQ(), torch.float32, 0.0371)

    @parameterized.expand(
        [("bfloat16", torch.bfloat16, 300), ("float16", torch.float16, 3000)]
    )
    def test_dequantize_rounds_to_out_dtype_before_scaling(
        self, _, out_dtype, zero_point
    ):
        # q - zero_point is not representable in out_dtype, and PyTorch rounds it there.
        if out_dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        class QDQ(nn.Module):
            def forward(self, x):
                q = qd.quantize_per_tensor(x, 1.0, 0, -128, 127, torch.int8)
                return qd.dequantize_per_tensor(
                    q, 0.0371, zero_point, -128, 127, torch.int8, out_dtype=out_dtype
                )

        self.run_test(
            QDQ(), [torch.arange(-128.0, 128.0)], use_dynamo_tracer=True, rtol=0, atol=0
        )

    @parameterized.expand([("axis_1", 1), ("axis_minus_1", -1)])
    def test_per_channel(self, _, axis):
        class QDQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scales", torch.rand(6) * 0.1 + 0.01)
                self.register_buffer("zero_points", torch.randint(0, 255, (6,)))

            def forward(self, x):
                q = qd.quantize_per_channel(
                    x, self.scales, self.zero_points, axis, 0, 255, torch.uint8
                )
                return qd.dequantize_per_channel(
                    q, self.scales, self.zero_points, axis, 0, 255, torch.uint8
                )

        shape = (2, 6, 6) if axis == 1 else (2, 3, 6)
        self.run_test(
            QDQ(), [torch.randn(shape) * 10], use_dynamo_tracer=True, rtol=0, atol=0
        )

    @parameterized.expand(
        [
            ("float16_out", [300, -3000, 1000, 7], torch.float16),
            ("bfloat16_out", [300, -3000, 1000, 7], torch.bfloat16),
            ("no_zero_point", None, None),
        ]
    )
    def test_dequantize_per_channel(self, _, zero_points, out_dtype):
        # The per-channel op computes in float32 and casts at the end. These zero points put
        # q - zero_point beyond the integers float16 and bfloat16 hold exactly.
        if out_dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        class DQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scales", torch.rand(4) * 0.1 + 0.01)
                self.register_buffer(
                    "zero_points",
                    None if zero_points is None else torch.tensor(zero_points),
                )

            def forward(self, q):
                return qd.dequantize_per_channel(
                    q,
                    self.scales,
                    self.zero_points,
                    1,
                    0,
                    255,
                    torch.uint8,
                    out_dtype=out_dtype,
                )

        self.run_test(
            DQ(),
            [torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)],
            use_dynamo_tracer=True,
            rtol=0,
            atol=0,
        )

    def test_quantize_per_channel_is_exact(self):
        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scales", torch.rand(4) * 0.1 + 0.01)
                self.register_buffer("zero_points", torch.tensor([-7, 0, 3, 100]))

            def forward(self, x):
                return qd.quantize_per_channel(
                    x, self.scales, self.zero_points, 1, -128, 127, torch.int8
                )

        self.run_test(
            Q(), [torch.randn(3, 4, 5) * 5], use_dynamo_tracer=True, rtol=0, atol=0
        )

    @parameterized.expand([("per_tensor", None), ("per_channel", 1)])
    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "TensorRT-RTX has no int8 activations",
    )
    def test_dequantize_int8_input(self, _, axis):
        # TensorRT reads an int8 engine input only through its dequantize layer.
        class DQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scales", torch.rand(4) * 0.1 + 0.01)
                self.register_buffer("zero_points", torch.tensor([-7, 0, 3, 100]))

            def forward(self, q):
                if axis is None:
                    return qd.dequantize_per_tensor(
                        q, 0.0371, -7, -128, 127, torch.int8
                    )
                return qd.dequantize_per_channel(
                    q, self.scales, self.zero_points, axis, -128, 127, torch.int8
                )

        self.run_test(
            DQ(),
            [torch.randint(-128, 128, (3, 4, 5), dtype=torch.int8)],
            use_dynamo_tracer=True,
            rtol=0,
            atol=0,
        )

    def test_int8_engine_input_stays_in_pytorch_on_tensorrt_rtx(self):
        def dequantize(q):
            return qd.dequantize_per_tensor(q, 0.1, 0, -128, 127, torch.int8)

        class DQ(nn.Module):
            def forward(self, q, x):
                single = qd.quantize_per_tensor(x, 0.1, 0, -128, 127, torch.int8)
                # A partitioner may put the users of a shared quantize in other engines.
                shared = qd.quantize_per_tensor(x + 1, 0.1, 0, -128, 127, torch.int8)
                return [dequantize(t) for t in (q, single, shared, shared)]

        inputs = (torch.zeros(4, dtype=torch.int8), torch.randn(4))
        gm = torch.export.export(DQ(), inputs).module()
        dequantizes = [
            n for n in gm.graph.nodes if n.target == qd.dequantize_per_tensor.default
        ]
        DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())
        rtx = aten_ops_converters.ENABLED_FEATURES._replace(tensorrt_rtx=True)
        with mock.patch.object(aten_ops_converters, "ENABLED_FEATURES", rtx):
            self.assertEqual(
                [n in DYNAMO_CONVERTERS for n in dequantizes],
                [False, True, False, False],
            )

    def test_dequantize_constant_weight(self):
        class Weighted(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randint(-127, 128, (16, 32), dtype=torch.int8)
                )
                self.register_buffer("scales", torch.rand(16) * 0.01)
                self.register_buffer("zero_points", torch.zeros(16, dtype=torch.int64))

            def forward(self, x):
                w = qd.dequantize_per_channel(
                    self.weight, self.scales, self.zero_points, 0, -127, 127, torch.int8
                )
                # Elementwise: exact on TensorRT-RTX too, unlike a float32 GEMM.
                return x * w

        self.run_test(
            Weighted(),
            [torch.randn(4, 16, 32)],
            use_dynamo_tracer=True,
            rtol=0,
            atol=0,
        )

    def test_dynamic_shape(self):
        class QDQ(nn.Module):
            def forward(self, x):
                q = qd.quantize_per_tensor(x, 0.1, 128, 0, 255, torch.uint8)
                return qd.dequantize_per_tensor(q, 0.1, 128, 0, 255, torch.uint8)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 8),
                opt_shape=(4, 8),
                max_shape=(8, 8),
            )
        ]
        self.run_test_with_dynamic_shape(
            QDQ(), input_specs, use_dynamo_tracer=True, rtol=0, atol=0
        )

    def test_float32_warning_is_logged_once(self):
        aten_ops_converters._warn_quantized_decomposed_float32.cache_clear()

        class QDQ(nn.Module):
            def forward(self, x):
                q = qd.quantize_per_tensor(x, 0.25, 0, -128, 127, torch.int8)
                return qd.dequantize_per_tensor(q, 0.25, 0, -128, 127, torch.int8)

        with self.assertLogs(aten_ops_converters.__name__, "WARNING") as logs:
            self.run_test(
                QDQ(), [torch.randn(8)], use_dynamo_tracer=True, rtol=0, atol=0
            )
        self.assertEqual(sum("no int8 speed" in line for line in logs.output), 1)

    def test_unsupported_forms_are_left_to_pytorch(self):
        class Forms(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scales", torch.rand(4, dtype=torch.float64) + 0.1)
                self.register_buffer("zero_points", torch.zeros(4, dtype=torch.int64))
                self.register_buffer("float32_scales", torch.rand(4) + 0.1)
                self.register_buffer(
                    "int8_zero_points", torch.ones(4, dtype=torch.int8)
                )
                self.register_buffer(
                    "float_zero_points", torch.tensor([1.5, -2.5, 0.25, 3.0])
                )
                self.register_buffer(
                    "int32_zero_points", torch.tensor([3, -7, 0, 1], dtype=torch.int32)
                )
                self.register_buffer(
                    "one_zero_point", torch.zeros(1, dtype=torch.int64)
                )

            def per_channel(self, x, scales, zero_points, axis=1):
                q = qd.quantize_per_channel(
                    x, scales, zero_points, axis, -128, 127, torch.int8
                )
                return qd.dequantize_per_channel(
                    q, scales, zero_points, axis, -128, 127, torch.int8
                )

            def forward(self, x):
                supported = qd.quantize_per_tensor(x, 0.1, 3, 0, 255, torch.uint8)
                int16 = qd.quantize_per_tensor(x, 0.1, 0, -32768, 32767, torch.int16)
                scale, zero_point = qd.choose_qparams.tensor(
                    x, -128, 127, 1e-5, torch.int8
                )
                runtime = qd.quantize_per_tensor.tensor(
                    x, scale, zero_point, -128, 127, torch.int8
                )
                other_user = qd.quantize_per_tensor(x, 0.1, 0, -128, 127, torch.int8)
                # PyTorch raises for these two at run time; torch.export does not check them.
                beyond_dtype = qd.quantize_per_tensor(x, 0.1, 0, 0, 1000, torch.uint8)
                bad_axis = self.per_channel(
                    x, self.float32_scales, self.zero_points, -3
                )
                float64_out = qd.quantize_per_tensor(x, 0.1, 0, -128, 127, torch.int8)
                return (
                    qd.dequantize_per_tensor(supported, 0.1, 3, 0, 255, torch.uint8),
                    qd.dequantize_per_tensor(int16, 0.1, 0, -32768, 32767, torch.int16),
                    qd.dequantize_per_tensor.tensor(
                        runtime, scale, zero_point, -128, 127, torch.int8
                    ),
                    other_user.to(torch.int32),
                    qd.dequantize_per_tensor(
                        beyond_dtype, 0.1, 0, 0, 1000, torch.uint8
                    ),
                    bad_axis,
                    qd.dequantize_per_tensor(
                        float64_out,
                        0.1,
                        0,
                        -128,
                        127,
                        torch.int8,
                        out_dtype=torch.float64,
                    ),
                    self.per_channel(x, self.scales, self.zero_points),
                    self.per_channel(x, self.float32_scales, self.int8_zero_points),
                    self.per_channel(x, self.float32_scales, self.float_zero_points),
                    self.per_channel(x, self.float32_scales, self.int32_zero_points),
                    self.per_channel(x, self.float32_scales, self.one_zero_point),
                )

        gm = torch.export.export(Forms(), (torch.randn(2, 4),)).module()
        outputs = next(n for n in gm.graph.nodes if n.op == "output").args[0]
        DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())

        def converted_ops(output):
            # Whether each quantized_decomposed op that this output depends on converts.
            ops, stack = {}, [output]
            while stack:
                node = stack.pop()
                if getattr(node.target, "namespace", None) == "quantized_decomposed":
                    ops[node.target.__name__] = node in DYNAMO_CONVERTERS
                stack.extend(node.all_input_nodes)
            return ops

        q, dq = "quantize_per_tensor.default", "dequantize_per_tensor.default"
        qc, dqc = "quantize_per_channel.default", "dequantize_per_channel.default"
        runtime_ops = {
            "choose_qparams.tensor": False,
            "quantize_per_tensor.tensor": False,
            "dequantize_per_tensor.tensor": False,
        }
        expected = {
            "supported": {q: True, dq: True},
            "int16": {q: False, dq: False},
            "runtime": runtime_ops,
            "other_user": {q: False},
            # PyTorch's per-tensor dequantize does not check its range.
            "beyond_dtype": {q: False, dq: True},
            "bad_axis": {qc: False, dqc: False},
            "float64_out": {q: True, dq: False},
            "float64_scales": {qc: False, dqc: False},
            "int8_zero_points": {qc: False, dqc: False},
            "float_zero_points": {qc: False, dqc: False},
            "int32_zero_points": {qc: True, dqc: True},
            # PyTorch raises when the zero points and scales differ in length.
            "one_zero_point": {qc: False, dqc: False},
        }
        self.assertEqual(len(outputs), len(expected))
        for output, (form, ops) in zip(outputs, expected.items()):
            with self.subTest(form):
                self.assertEqual(converted_ops(output), ops)


if __name__ == "__main__":
    run_tests()
