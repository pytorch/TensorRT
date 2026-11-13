# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    UnsupportedOperatorException,
)
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    to_copy_dtype_validator,
)

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestCloneConverter(DispatchTestCase):
    def test_clone_contiguous(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.ops.aten.clone.default(
                    x, memory_format=torch.contiguous_format
                )
                return y + 1

        inputs = [torch.randn((1, 3, 10))]
        self.run_test(
            Clone(),
            inputs,
        )

    def test_clone_regular(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.ops.aten.clone.default(x)
                return y + 1

        inputs = [torch.randn((8, 2, 10))]
        self.run_test(
            Clone(),
            inputs,
        )


class TestToCopyConverter(DispatchTestCase):
    def test_to_copy_half(self):
        class ToCopyHalf(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.half)
                return y

        inputs = [torch.rand((1, 3, 10), dtype=torch.half)]
        self.run_test(
            ToCopyHalf(),
            inputs,
        )

    def test_to_copy_float(self):
        class ToCopyFloat(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.float)
                return y

        inputs = [torch.rand((1, 3, 10), dtype=torch.float)]
        self.run_test(
            ToCopyFloat(),
            inputs,
        )

    def test_to_copy_bfloat16(self):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class ToCopyBFloat16(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.bfloat16)
                y = y**2
                return y

        inputs = [torch.rand((1, 3, 10), dtype=torch.bfloat16)]
        self.run_test(
            ToCopyBFloat16(),
            inputs,
        )

    def test_to_copy_i64b(self):
        class ToCopy64Bit(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.int64)
                return y

        inputs = [torch.randn((1, 3, 10)).int()]

        self.run_test(
            ToCopy64Bit(),
            inputs,
        )

    def test_to_copy_float64_with_truncate_double(self):
        class ToCopyDouble(nn.Module):
            def forward(self, x):
                y = x * 2
                return y, torch.ops.aten._to_copy.default(y, dtype=torch.float64)

        # aten.copy reaches the same cast without the _to_copy validator, and
        # broadcasts its source to the buffer shape.
        class CopyIntoDouble(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("dest", torch.zeros(2, 3, 10, dtype=torch.double))

            def forward(self, x):
                return torch.ops.aten.copy.default(self.dest, x * 2)

        inputs = [torch.rand((1, 3, 10), dtype=torch.float)]
        for model in (ToCopyDouble(), CopyIntoDouble()):
            with self.subTest(type(model).__name__):
                self.run_test(
                    model,
                    inputs,
                    check_dtype=False,
                    use_dynamo_tracer=True,
                    require_full_compilation=True,
                )

    def test_to_copy_float64_dynamic_with_memory_format_after_pytorch_op(self):
        # lgamma stays in PyTorch, so the cast ends up alone on an engine input.
        class LgammaDouble(nn.Module):
            def __init__(self, memory_format):
                super().__init__()
                self.memory_format = memory_format

            def forward(self, x):
                return torch.lgamma(x).to(
                    torch.float64, memory_format=self.memory_format
                )

        for memory_format in (None, torch.contiguous_format, torch.preserve_format):
            with self.subTest(memory_format=memory_format):
                model = LgammaDouble(memory_format).eval().cuda()
                x = torch.rand(3, 5, device="cuda") + 0.5
                batch = torch.export.Dim("batch", min=1, max=8)
                exported = torch.export.export(
                    model, (x,), dynamic_shapes=({0: batch},)
                )
                compiled = torch_tensorrt.dynamo.compile(
                    exported,
                    arg_inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, 5),
                            opt_shape=(3, 5),
                            max_shape=(8, 5),
                            dtype=torch.float32,
                        )
                    ],
                    min_block_size=1,
                    truncate_double=True,
                )
                # Only the cast without a memory_format goes to TensorRT.
                self.assertEqual(
                    sum("_run_on_acc" in name for name, _ in compiled.named_children()),
                    int(memory_format is None),
                )
                for n in (1, 3, 8):
                    x = torch.rand(n, 5, device="cuda") + 0.5
                    torch.testing.assert_close(compiled(x), model(x))

    def test_float64_copies_share_one_truncation_rule(self):
        # The float64 rule for _to_copy applies to aten.copy too: a source float32
        # cannot hold exactly, or a result layout TensorRT cannot produce, stays in
        # PyTorch, and the other cases convert.
        class CopyIntoDouble(nn.Module):
            def __init__(self, memory_format):
                super().__init__()
                dest = torch.zeros(2, 3, 4, 5, dtype=torch.double)
                self.register_buffer("dest", dest.to(memory_format=memory_format))

            def forward(self, x):
                return torch.ops.aten.copy.default(self.dest, x * 2)

        class BoolToDouble(nn.Module):
            def forward(self, x):
                return (x > 0.5).to(torch.float64) * 3

        # Inside the engine the uint8 value is int32, so only the graph's dtype shows
        # that float32 holds it.
        class Uint8ToDouble(nn.Module):
            def forward(self, x):
                return (x * 255).to(torch.uint8).double()

        class CopyUint8IntoDouble(CopyIntoDouble):
            def forward(self, x):
                return torch.ops.aten.copy.default(self.dest, (x * 255).to(torch.uint8))

        cases = (
            (CopyIntoDouble(torch.channels_last), torch.rand(2, 3, 4, 5), True),
            (
                CopyIntoDouble(torch.contiguous_format),
                torch.randint(9, (2, 3, 4, 5)),
                True,
            ),
            (BoolToDouble(), torch.rand(2, 3, 4, 5), False),
            (Uint8ToDouble(), torch.rand(2, 3, 4, 5), False),
            (
                CopyUint8IntoDouble(torch.contiguous_format),
                torch.rand(2, 3, 4, 5),
                False,
            ),
        )
        for model, x, copy_in_pytorch in cases:
            with self.subTest(type(model).__name__, dtype=x.dtype):
                model = model.eval().cuda()
                x = x.cuda()
                compiled = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, (x,)),
                    arg_inputs=[x],
                    min_block_size=1,
                    truncate_double=True,
                )
                names = [name for name, _ in compiled.named_children()]
                self.assertEqual(sum("_run_on_acc" in name for name in names), 1)
                self.assertEqual(
                    any("_run_on_gpu" in name for name in names), copy_in_pytorch
                )
                out, expected = compiled(x), model(x)
                torch.testing.assert_close(out, expected)
                self.assertEqual(out.stride(), expected.stride())

    def test_to_copy_uint8_stays_in_one_engine(self):
        # PyTorch truncates toward zero, then wraps modulo 256. Values outside the int64
        # range, infinity and NaN are unspecified: PyTorch differs on CPU and CUDA.
        class ToUint8(nn.Module):
            def forward(self, x):
                return x.to(torch.uint8), x.to(torch.uint8).float() * 2

        # A cond branch that casts to uint8 can stay in the engine too.
        class CondCast(nn.Module):
            def forward(self, pred, x):
                return torch.cond(
                    pred, lambda v: v.to(torch.uint8).float(), lambda v: v * 2, (x,)
                )

        # NHWC frames, so the uint8 value is not contiguous until the slice is copied.
        class Tiles(nn.Module):
            def forward(self, frames, other):
                tiles = frames.permute(0, 3, 1, 2).round().clamp(0, 255)
                tiles = tiles.to(torch.uint8)[..., 1:3, :].contiguous()
                return torch.cat([tiles, other], dim=-1).float()

        # 3e9 is past the int32 range, where a cast through int32 would saturate.
        cases = (
            (ToUint8(), (torch.tensor([256.7, -1.5, 3e9]),)),
            (ToUint8(), (torch.tensor([300, -1, -257, 2**40 + 300]),)),
            (ToUint8(), (torch.tensor([300, -1, -257]).int(),)),
            (ToUint8(), (torch.tensor([256.7, -1.5, 300.0]).half(),)),
            (ToUint8(), (torch.tensor([256.7, -1.5, 3e9]).bfloat16(),)),
            (ToUint8(), (torch.tensor([True, False]),)),
            (CondCast(), (torch.tensor(True), torch.tensor([1.5, 2.0]))),
            (
                Tiles(),
                (
                    torch.linspace(-20, 300, 2 * 4 * 5 * 3).reshape(2, 4, 5, 3),
                    torch.arange(2 * 3 * 2 * 4, dtype=torch.uint8).reshape(2, 3, 2, 4),
                ),
            ),
        )
        for model, inputs in cases:
            with self.subTest(type(model).__name__, dtype=inputs[-1].dtype):
                if inputs[-1].dtype == torch.bfloat16:
                    skip_if_trt_rtx_turing(self, "bfloat16")
                inputs = [x.cuda() for x in inputs]
                compiled = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                )
                names = [name for name, _ in compiled.named_children()]
                self.assertEqual(names, ["_run_on_acc_0"])
                torch.testing.assert_close(
                    compiled(*inputs), model(*inputs), rtol=0, atol=0
                )

    def test_to_copy_uint8_math_is_refused_without_fallback(self):
        # The int32 values standing for uint8 would not wrap: 100 + 200 would give 300.
        class AddToUint8(nn.Module):
            def forward(self, x):
                return (x.to(torch.uint8) + 200).float()

        x = torch.tensor([100.0], device="cuda")
        with self.assertRaises(UnsupportedOperatorException) as raised:
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                torch.export.export(AddToUint8(), (x,)), inputs=[x], min_block_size=1
            )
        self.assertIn("computes a uint8 result", str(raised.exception.__cause__))

    def test_to_copy_multiple_returns(self):
        class ToCopyReturns(nn.Module):
            def forward(self, x):
                x_1 = x + 1
                y = torch.ops.aten._to_copy.default(x_1, dtype=torch.float)
                z = torch.ops.aten._to_copy.default(x_1, dtype=torch.float)
                return y, z

        inputs = [torch.rand((1, 3, 10), dtype=torch.float)]
        self.run_test(
            ToCopyReturns(),
            inputs,
        )

    def test_to_copy_validator_rejects_int8_to_uint8(self):
        class AddThenToUint8(nn.Module):
            def forward(self, x):
                return torch.ops.aten._to_copy.default(x + 1, dtype=torch.uint8)

        x = torch.zeros(4, dtype=torch.int8)
        exported = torch.export.export(AddThenToUint8(), (x,))
        node = next(
            node
            for node in exported.graph.nodes
            if node.target is torch.ops.aten._to_copy.default
        )
        self.assertFalse(to_copy_dtype_validator(placeholder_only=False)(node))

    def test_to_copy_validator_rejects_device_transfer(self):
        class DeviceAndDtypeCopy(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                y = x + 1
                return torch.ops.aten._to_copy.default(
                    y, device=self.device, dtype=torch.int32
                )

        def to_copy_node(device):
            x = torch.randn(4, device="cuda")
            exported = torch.export.export(DeviceAndDtypeCopy(device), (x,))
            return next(
                node
                for node in exported.graph.nodes
                if node.target is torch.ops.aten._to_copy.default
            )

        validator = to_copy_dtype_validator(placeholder_only=False)
        self.assertFalse(validator(to_copy_node("cpu")))
        self.assertTrue(validator(to_copy_node("cuda")))

    def test_to_copy_validator_float64_keeps_layout_in_pytorch(self):
        class DoubleCopy(nn.Module):
            def __init__(self, memory_format, view):
                super().__init__()
                self.memory_format = memory_format
                self.view = view

            def forward(self, x):
                y = self.view(x + 1)
                return torch.ops.aten._to_copy.default(
                    y, dtype=torch.float64, memory_format=self.memory_format
                )

        def to_copy_node(memory_format, dtype=torch.float32, view=lambda y: y):
            x = torch.ones(2, 3, 4, 5, dtype=dtype)
            exported = torch.export.export(DoubleCopy(memory_format, view), (x,))
            return next(
                node
                for node in exported.graph.nodes
                if node.target is torch.ops.aten._to_copy.default
            )

        validator = to_copy_dtype_validator(placeholder_only=False)
        settings = CompilationSettings(truncate_double=True)
        self.assertTrue(validator(to_copy_node(torch.contiguous_format), settings))
        self.assertFalse(validator(to_copy_node(torch.channels_last), settings))
        # Copying a strided input gives a contiguous result, which TensorRT matches.
        self.assertTrue(
            validator(
                to_copy_node(torch.preserve_format, view=lambda y: y[..., ::2]),
                settings,
            )
        )
        self.assertFalse(
            validator(
                to_copy_node(torch.preserve_format, view=lambda y: y.transpose(1, 2)),
                settings,
            )
        )
        # TensorRT cannot cast float8 to float32.
        self.assertFalse(
            validator(
                to_copy_node(torch.contiguous_format, torch.float8_e4m3fn), settings
            )
        )
        self.assertFalse(
            validator(
                to_copy_node(torch.contiguous_format),
                CompilationSettings(truncate_double=False),
            )
        )


if __name__ == "__main__":
    run_tests()
