# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from unittest.mock import patch

import pytest
import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.backend import backends
from torch_tensorrt.dynamo.conversion import impl

if not impl.quantize.QUANTIZED_DECOMPOSED_QUANTIZE_OPS:
    pytest.skip(
        "torch.ao.quantization.fx._decomposed is not available",
        allow_module_level=True,
    )

qd = torch.ops.quantized_decomposed
OUTPUT_SCALE = 0.1
# TensorRT-RTX rejects the float32 linear on Turing, so full compilation fails there.
TRT_RTX_TURING = (
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (7, 5)
)


def _quantize_weight(weight: torch.Tensor):
    scales = (weight.abs().amax(dim=list(range(1, weight.dim()))) / 127).float()
    zero_points = torch.zeros_like(scales, dtype=torch.int64)
    return (
        qd.quantize_per_channel(weight, scales, zero_points, 0, -127, 127, torch.int8),
        scales,
        zero_points,
    )


class ConvLinear(torch.nn.Module):
    """The graph convert_pt2e makes: Q/DQ around activations, int8 per-channel weights."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype
        # PT2E quantizers give activations a nonzero zero point, for int8 too.
        self.range = (0, 255, 117) if dtype == torch.uint8 else (-128, 127, 8)
        for name, shape in (("conv", (8, 3, 3, 3)), ("linear", (10, 8 * 6 * 6))):
            weight, scales, zero_points = _quantize_weight(torch.randn(shape))
            self.register_buffer(f"{name}_weight", weight)
            self.register_buffer(f"{name}_scales", scales)
            self.register_buffer(f"{name}_zero_points", zero_points)
        self.conv_bias = torch.nn.Parameter(torch.randn(8) * 0.1)
        self.linear_bias = torch.nn.Parameter(torch.randn(10) * 0.1)

    def qdq(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        lo, hi, zero_point = self.range
        q = qd.quantize_per_tensor(x, scale, zero_point, lo, hi, self.dtype)
        return qd.dequantize_per_tensor(q, scale, zero_point, lo, hi, self.dtype)

    def weight(self, name: str) -> torch.Tensor:
        return qd.dequantize_per_channel(
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_scales"),
            getattr(self, f"{name}_zero_points"),
            0,
            -127,
            127,
            torch.int8,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qdq(x, 0.02)
        x = torch.relu(
            torch.nn.functional.conv2d(x, self.weight("conv"), self.conv_bias)
        )
        x = self.qdq(x, 0.05).flatten(1)
        x = torch.nn.functional.linear(x, self.weight("linear"), self.linear_bias)
        return self.qdq(x, OUTPUT_SCALE)


class ActivationQDQ(torch.nn.Module):
    """Per-channel and .tensor Q/DQ pairs on activations, with parameters in buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scales", torch.rand(4) * 0.1 + 0.01)
        self.register_buffer("zero_points", torch.tensor([-7, 0, 3, 100]))
        self.register_buffer("scale", torch.tensor(OUTPUT_SCALE))
        self.register_buffer("zero_point", torch.tensor(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = (self.scales, self.zero_points, 1, -128, 127, torch.int8)
        x = torch.relu(
            qd.dequantize_per_channel(qd.quantize_per_channel(x, *args), *args)
        )
        args = (self.scale, self.zero_point, -128, 127, torch.int8)
        return qd.dequantize_per_tensor.tensor(
            qd.quantize_per_tensor.tensor(x, *args), *args
        )


def _dequantize(q: torch.Tensor) -> torch.Tensor:
    return qd.dequantize_per_tensor(q, 0.125, 0, -128, 127, torch.int8)


def _pytorch_targets(module: torch.fx.GraphModule) -> set:
    # A module with no TensorRT engine comes back unpartitioned, with its ops at the top.
    graphs = [module] + [
        child for name, child in module.named_children() if "_run_on_acc" not in name
    ]
    return {
        node.target
        for graph_module in graphs
        if isinstance(graph_module, torch.fx.GraphModule)
        for node in graph_module.graph.nodes
    }


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestPT2EQuantized(TestCase):
    def compile_and_compare(self, model, x, **settings):
        exported = torch.export.export(model, (x,))
        compiled = torch_tensorrt.dynamo.compile(
            exported, inputs=[x], min_block_size=1, disable_tf32=True, **settings
        )
        self.assert_close_up_to_one_step(compiled(x), model(x), OUTPUT_SCALE)
        return compiled

    def assert_close_up_to_one_step(self, actual, expected, step):
        # A float result one ulp off can round to the next integer, so allow one output
        # step, but only on the rare values that sit that close to a halfway point.
        torch.testing.assert_close(actual, expected, rtol=0, atol=step * 1.01)
        moved = int(((actual - expected).abs() > step / 2).sum())
        self.assertLessEqual(moved, max(1, actual.numel() // 100))

    @parameterized.expand([("uint8", torch.uint8), ("int8", torch.int8)])
    @unittest.skipIf(
        TRT_RTX_TURING, "FP32 linear is not supported on TensorRT-RTX for Turing"
    )
    def test_one_engine(self, _, dtype):
        torch.manual_seed(0)
        model = ConvLinear(dtype).eval().cuda()
        # Without the quantize and dequantize converters this raises: the model is not fully supported.
        self.compile_and_compare(
            model, torch.randn(2, 3, 8, 8, device="cuda"), require_full_compilation=True
        )

    def test_per_channel_and_tensor_qparams_one_engine(self):
        torch.manual_seed(0)
        compiled = self.compile_and_compare(
            ActivationQDQ().cuda(),
            torch.randn(2, 4, 8, device="cuda") * 5,
            require_full_compilation=True,
        )
        self.assertFalse(
            {
                target
                for target in _pytorch_targets(compiled)
                if getattr(target, "namespace", None) == "quantized_decomposed"
            }
        )

    @parameterized.expand(
        [
            ("uint8_quantize_out", torch.uint8, qd.dequantize_per_tensor.default),
            ("int8_quantize_out", torch.int8, qd.dequantize_per_tensor.default),
            ("uint8_dequantize_in", torch.uint8, qd.quantize_per_tensor.default),
            ("int8_dequantize_in", torch.int8, qd.quantize_per_tensor.default),
        ]
    )
    def test_quantized_values_cross_engine_boundary(self, _, dtype, pytorch_op):
        if (
            torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx
            and dtype == torch.int8
            and pytorch_op == qd.quantize_per_tensor.default
        ):
            self.skipTest("TensorRT-RTX has no int8 activations")
        torch.manual_seed(0)
        model = ConvLinear(dtype).eval().cuda()
        compiled = self.compile_and_compare(
            model,
            torch.randn(2, 3, 8, 8, device="cuda"),
            torch_executed_ops={pytorch_op},
        )
        targets = _pytorch_targets(compiled)
        self.assertIn(pytorch_op, targets)
        self.assertFalse(
            targets
            & {qd.quantize_per_tensor.default, qd.dequantize_per_tensor.default}
            - {pytorch_op}
        )

    @parameterized.expand(
        [
            # Only an engine output casts a quantize's float32 result back to int8, and a
            # branch output is not one.
            (
                "returns_quantize",
                lambda x, q, pred: _dequantize(
                    torch.cond(
                        pred,
                        lambda x, q: qd.quantize_per_tensor(
                            x, 0.125, 0, -128, 127, torch.int8
                        ),
                        lambda x, q: q.clone(),
                        (x, q),
                    )
                ),
            ),
            # TensorRT rejects int8 values that enter a conditional and reach its
            # dequantize layer.
            (
                "dequantizes_int8_input",
                lambda x, q, pred: torch.cond(
                    pred, lambda x, q: _dequantize(q) + x, lambda x, q: x - 1, (x, q)
                )
                * 2,
            ),
        ]
    )
    def test_cond_branch_stays_in_pytorch(self, _, forward):
        class Model(torch.nn.Module):
            def forward(self, x, q, pred):
                return forward(x, q, pred)

        torch.manual_seed(0)
        x = torch.randn(2, 4, device="cuda") * 10
        q = torch.randint(-128, 128, (2, 4), dtype=torch.int8, device="cuda")
        inputs = (x, q, torch.tensor(True, device="cuda"))
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Model(), inputs), inputs=list(inputs), min_block_size=1
        )
        self.assertIn(torch.ops.higher_order.cond, _pytorch_targets(compiled))
        for pred in (True, False):
            args = (x, q, torch.tensor(pred, device="cuda"))
            torch.testing.assert_close(compiled(*args), Model()(*args), rtol=0, atol=0)

    def test_cond_branch_with_quantize_dequantize_converts(self):
        class Model(torch.nn.Module):
            def forward(self, x, pred):
                return torch.cond(
                    pred,
                    lambda x: _dequantize(
                        qd.quantize_per_tensor(x, 0.125, 0, -128, 127, torch.int8)
                    )
                    + 1,
                    lambda x: x - 1,
                    (x,),
                )

        torch.manual_seed(0)
        x = torch.randn(2, 4, device="cuda") * 10
        inputs = (x, torch.tensor(True, device="cuda"))
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Model(), inputs), inputs=list(inputs), min_block_size=1
        )
        self.assertNotIn(torch.ops.higher_order.cond, _pytorch_targets(compiled))
        for pred in (True, False):
            args = (x, torch.tensor(pred, device="cuda"))
            torch.testing.assert_close(compiled(*args), Model()(*args), rtol=0, atol=0)

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "TensorRT-RTX has no int8 activations",
    )
    def test_torch_compile_int8_input(self):
        class Model(torch.nn.Module):
            def forward(self, q):
                return qd.dequantize_per_tensor(q, 0.125, 3, -128, 127, torch.int8)

        q = torch.randint(-128, 128, (2, 8), dtype=torch.int8, device="cuda")
        compiled_graphs = []
        original_compile_module = backends.compile_module

        def capture_compile(*args, **kwargs):
            compiled_graphs.append(original_compile_module(*args, **kwargs))
            return compiled_graphs[-1]

        torch._dynamo.reset()
        try:
            with patch.object(backends, "compile_module", capture_compile):
                # Without pass_through_build_failures, a failed build quietly falls back
                # to PyTorch.
                compiled = torch.compile(
                    Model(),
                    backend="torch_tensorrt",
                    options={"min_block_size": 1, "pass_through_build_failures": True},
                )
                torch.testing.assert_close(compiled(q), Model()(q), rtol=0, atol=0)
        finally:
            torch._dynamo.reset()

        self.assertEqual(len(compiled_graphs), 1)
        gm = compiled_graphs[0]
        self.assertTrue(any("_run_on_acc" in name for name, _ in gm.named_children()))
        self.assertNotIn(qd.dequantize_per_tensor.default, _pytorch_targets(gm))

    @unittest.skipIf(
        TRT_RTX_TURING, "FP32 linear is not supported on TensorRT-RTX for Turing"
    )
    def test_xnnpack_quantizer(self):
        # The quantizer is a private torchao test helper that a release may not ship.
        try:
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e,
                prepare_pt2e,
            )
            from torchao.testing.pt2e._xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
        except ImportError as e:
            self.skipTest(f"torchao PT2E XNNPACKQuantizer not available: {e}")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3)
                self.linear = torch.nn.Linear(8 * 6 * 6, 10)

            def forward(self, x):
                return self.linear(torch.relu(self.conv(x)).flatten(1))

        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, 8, device="cuda")
        model = torch.export.export(Model().eval().cuda(), (x,)).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        model = prepare_pt2e(model, quantizer)
        model(x)
        model = convert_pt2e(model)
        exported = torch.export.export(model, (x,))
        compiled = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[x],
            min_block_size=1,
            disable_tf32=True,
            require_full_compilation=True,
        )
        # The model ends in a dequantize, so allow one step of its scale, as above.
        output_scale = [
            n.args[1]
            for n in exported.graph.nodes
            if n.target == qd.dequantize_per_tensor.default
        ][-1]
        self.assert_close_up_to_one_step(compiled(x), model(x), output_scale)


if __name__ == "__main__":
    run_tests()
