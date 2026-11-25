# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture boundaries, TensorRT placement, and serialization of PyTorch regions."""

import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.region_compilation import (
    check_decomposed_regions,
    get_region_records,
    is_region_hop,
    normalize_region_scopes,
    region_hop_records,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region._errors import RegionCaptureError, RegionError
from torch_tensorrt.region._session import RegionCompilationSession

from ._utils import capture_regions, compile_regions


class SharedBlock(torch.nn.Module):
    """All operations have converters; one invocation has an explicit Torch policy."""

    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())

    def forward(self, x):
        h = self.block(x)
        with region.execute_in_torch(name="shared"):
            y = self.block(h)
        return self.block(y) + h


class RepeatedScopes(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        for _ in range(2):
            with region.execute_in_torch(name="repeated"):
                x = torch.relu(x * 1.5)
        return x - 3


class MultipleOutputs(torch.nn.Module):
    def __init__(self, direct_output=False):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))
        self.direct_output = direct_output

    def forward(self, x, y):
        h = x + 1
        with region.execute_in_torch(name="branches"):
            a = torch.relu(h * self.scale)
            b = torch.sigmoid(y - 1)
        return (a, b) if self.direct_output else (a + h, b * 3)


class ComplexRegion(torch.nn.Module):
    def __init__(self, location):
        super().__init__()
        self.location = location

    def forward(self, x):
        if self.location == "parent":
            z = torch.complex(x, x + 1)
            with region.execute_in_torch():
                y = torch.relu(x)
            return y + z.abs()
        with region.execute_in_torch():
            if self.location == "body":
                x = torch.complex(x, x + 1)
            y = x * x
            if self.location == "body":
                y = y.abs()
        return y


class TensorLiteralRegion(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch():
            y = x + torch.tensor([1.0, 2.0], device=x.device)
        return y * 2


class CpuScalarRegion(torch.nn.Module):
    """A literal and a scalar buffer that stay on the CPU, as PyTorch allows."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        with region.execute_in_torch():
            y = torch.relu(x) * self.scale + torch.tensor(0.5)
        return y + 1


class BufferRegion(torch.nn.Module):
    def __init__(self, constant_only=False):
        super().__init__()
        self.register_buffer("bias", torch.randn(8))
        self.constant_only = constant_only

    def forward(self, x):
        folded = self.bias + 2
        with region.execute_in_torch():
            y = torch.relu(self.bias * 2 if self.constant_only else x + folded)
        return x + y if self.constant_only else y * 3


class DeadRegion(torch.nn.Module):
    """Torch-TensorRT decomposes full_like into full(shape), so the region dies."""

    def forward(self, x):
        with region.execute_in_torch(name="shape_only"):
            y = torch.relu(x) * 2
        return torch.full_like(y, 3.0) + x


class ShapeOnlyRegion(DeadRegion):
    """A live region follows the region that decomposition removes."""

    def forward(self, x):
        z = super().forward(x)
        with region.execute_in_torch(name="live"):
            w = torch.sin(z)
        return w + 1


class Fp32Island(torch.nn.Module):
    """An fp16 model that keeps an fp32 RMSNorm in PyTorch."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, x):
        h = x + 1
        with region.execute_in_torch(backend=self.backend):
            f = h.float()
            f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
        return f.half() * 2


class SeededRegion(torch.nn.Module):
    """Operations tagged nondeterministic_seeded that draw no random numbers in eval."""

    def __init__(self, kind, backend="eager"):
        super().__init__()
        self.kind, self.backend = kind, backend
        self.attention = torch.nn.MultiheadAttention(
            8, 2, dropout=0.1, batch_first=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Dropout(0.1),
            torch.nn.Dropout1d(0.1),
            torch.nn.AlphaDropout(0.1),
            torch.nn.FeatureAlphaDropout(0.1),
            torch.nn.RReLU(),
        )
        self.lstm = torch.nn.LSTM(8, 8, batch_first=True)
        self.gru = torch.nn.GRU(8, 8, batch_first=True)
        self.rnn_tanh = torch.nn.RNN(8, 8, batch_first=True)
        self.rnn_relu = torch.nn.RNN(8, 8, nonlinearity="relu", batch_first=True)

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(backend=self.backend):
            if self.kind == "sdpa":
                y = F.scaled_dot_product_attention(x, x, x)
            elif self.kind == "native_dropout":
                # Traced, it returns its input, which a region must not output.
                y = torch.native_dropout(x * 2, 0.1, False)[0]
            elif self.kind == "attention":
                y = self.mlp(self.attention(x, x, x, need_weights=False)[0])
            elif self.kind == "mlp":
                y = self.mlp(x)
            else:
                y = getattr(self, self.kind)(x)[0]
        return y * 2


class TestExecuteInTorchCapture(TestCase):

    @parameterized.expand([("autocast",), ("no_grad",)])
    def test_higher_order_operator_can_read_region_output(self, construct):
        class ReadAfter(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    y = torch.relu(x)
                scope = (
                    torch.autocast("cpu", enabled=False)
                    if construct == "autocast"
                    else torch.no_grad()
                )
                # The block's body copies y's tags onto its input and cast guard.
                with scope:
                    z = y.float().softmax(-1)
                return z + 1

        model, x = ReadAfter().eval(), torch.randn(2, 8)
        with torch.enable_grad():
            gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    @parameterized.expand(
        [
            ("eval_dropout", lambda x: F.dropout(x, 0.1, training=False)),
            ("no_op_casts", lambda x: x.float().to(torch.float32)),
        ]
    )
    def test_ops_returning_their_input_do_not_become_outputs(self, _, op):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)

            def forward(self, x):
                with region.execute_in_torch():
                    h = self.fc(op(x))
                # Export binds these reads of x to the op that returned it.
                return x + h, x

        model, x = Residual().eval(), torch.randn(2, 8)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))

    def test_decomposition_may_drop_dead_regions_but_not_change_live_ones(self):
        with RegionCompilationSession():
            ep = torch.export.export(
                ShapeOnlyRegion().eval(), (torch.randn(2, 8),), strict=True
            )
        ep = normalize_region_scopes(ep)
        captured = ep.graph_module
        decomposed = ep.run_decompositions(get_decompositions()).graph_module
        self.assertEqual(
            [r.id for r in region_hop_records(captured)], ["shape_only#0", "live#1"]
        )
        self.assertEqual([r.id for r in region_hop_records(decomposed)], ["live#1"])
        check_decomposed_regions(captured, decomposed)

        (hop,) = (node for node in decomposed.graph.nodes if is_region_hop(node))
        hop.kwargs = {"hints": {**hop.kwargs["hints"], "backend": "inductor"}}
        with self.assertRaisesRegex(RegionError, "changed"):
            check_decomposed_regions(captured, decomposed)

    @parameterized.expand([("boundary",), ("body",), ("parent",)])
    def test_complex_tensors_anywhere_with_live_regions_are_rejected(self, location):
        dtype = torch.complex64 if location == "boundary" else torch.float32
        with self.assertRaisesRegex(RegionCaptureError, "complex tensors.*regions"):
            capture_regions(ComplexRegion(location), torch.randn(2, dtype=dtype))

    def test_cloned_chunk_outputs_are_supported(self):
        class ClonedOutputs(torch.nn.Module):
            def forward(self, x):
                with region.execute_in_torch():
                    a, b = (x + 1).chunk(2, dim=-1)
                    a, b = a.clone(), b.clone()
                return a, b

        model, x = ClonedOutputs().eval(), torch.randn(2, 8)
        gm = capture_regions(model, x)
        self.assertEqual(len(get_region_records(gm)), 1)
        torch.testing.assert_close(gm(x), model(x))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestExecuteInTorchIntegration(TestCase):
    @staticmethod
    def _compile(model, inputs, use_fast_partitioner=True, **overrides):
        options = dict(disable_tf32=True)
        options.update(overrides)
        return compile_regions(
            model, inputs, use_fast_partitioner=use_fast_partitioner, **options
        )

    @parameterized.expand(
        [
            (f"{name}_{partition}", model, shape, count, fast)
            for name, model, shape, count in [
                ("shared", SharedBlock, (2, 8), 1),
                ("multiple_outputs", MultipleOutputs, (2, 8), 1),
                ("repeated", RepeatedScopes, (2, 8), 2),
                ("folded_state", BufferRegion, (2, 8), 1),
                ("constant_input", lambda: BufferRegion(constant_only=True), (2, 8), 1),
                ("tensor_literal", TensorLiteralRegion, (2,), 1),
            ]
            for partition, fast in [("fast", True), ("global", False)]
            if fast or name in {"shared", "multiple_outputs", "constant_input"}
        ]
    )
    def test_region_survives_compilation(self, _, factory, shape, count, fast):
        model = factory().eval().cuda()
        inputs = [torch.randn(shape, device="cuda")]
        if isinstance(model, MultipleOutputs):
            inputs.append(torch.randn_like(inputs[0]))
        expected = model(*inputs)
        compiled = self._compile(model, inputs, fast)
        engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
        self.assertTrue(engines)
        if isinstance(model, SharedBlock):
            self.assertEqual(len(engines), 2)
        self.assertEqual(len(get_region_records(compiled)), count)
        torch.testing.assert_close(compiled(*inputs), expected, atol=5e-3, rtol=5e-3)

    @parameterized.expand([("fast", True), ("global", False)])
    def test_region_left_dead_by_decomposition_is_a_noop(self, _, fast):
        model = ShapeOnlyRegion().eval().cuda()
        x = torch.randn(2, 8, device="cuda")
        compiled = self._compile(model, (x,), fast)
        self.assertEqual([r.id for r in get_region_records(compiled)], ["live#1"])
        torch.testing.assert_close(compiled(x), model(x))

    def test_cpu_scalars_do_not_make_a_region_span_devices(self):
        x = torch.randn(2, 8, device="cuda")
        model = CpuScalarRegion().eval()  # Keep the 0-dim buffer on the CPU.
        compiled = self._compile(model, [x])
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x))
        # A CPU tensor with dimensions still makes the region span devices.
        with self.assertRaisesRegex(
            RegionCaptureError, "'branches#0' cannot span multiple devices: cpu, cuda"
        ):
            capture_regions(MultipleOutputs().cuda(), x, torch.randn(2, 8))

    @parameterized.expand([("eager",), ("inductor",)])
    def test_eval_attention_region_compiles(self, backend):
        if backend == "inductor" and not has_triton():
            self.skipTest("CUDA Inductor requires Triton support on the device")
        model = SeededRegion("attention", backend).eval().cuda()
        x = torch.randn(1, 4, 8, device="cuda")
        compiled = self._compile(model, (x,))
        self.assertTrue(
            any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
        )
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x), atol=5e-3, rtol=5e-3)

    @parameterized.expand([("eager",), ("inductor",)])
    def test_fp32_island_in_fp16_model_compiles(self, backend):
        if backend == "inductor" and not has_triton():
            self.skipTest("CUDA Inductor requires Triton support on the device")
        model = Fp32Island(backend).eval().cuda()
        x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
        compiled = self._compile(model, (x,))
        self.assertEqual(len(get_region_records(compiled)), 1)
        torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand(
        [
            (f"{setting}_{dryrun=}", setting, dryrun)
            for setting in (
                "require_full_compilation",
                "offload_module_to_cpu",
                "enable_autocast",
                "enable_resource_partitioning",
            )
            for dryrun in (False, True)
        ]
    )
    def test_conflicting_settings_stop_before_unsafe_stages(self, _, setting, dryrun):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        forbidden_stages = (
            "torch_tensorrt.dynamo._compiler.convert_module",
            "torch_tensorrt.dynamo._compiler.deallocate_module",
            "torch_tensorrt.dynamo._compiler.resource_partition",
            "torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass.trace_intermediate_node_outputs",
        )
        with ExitStack() as stack:
            for target in forbidden_stages:
                stack.enter_context(
                    mock.patch(
                        target, side_effect=AssertionError(f"Unexpected {target}")
                    )
                )
            # A dry run rejects the conflict as a compilation does.
            stack.enter_context(
                self.assertRaisesRegex((RuntimeError, ValueError), setting)
            )
            self._compile(model, (x,), dryrun=dryrun, **{setting: True})

    def test_training_model_is_rejected(self):
        model, x = SharedBlock().train().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self._compile(model, (x,))

    def test_engine_only_output_is_rejected(self):
        model, x = SharedBlock().eval().cuda(), torch.randn(2, 8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "serialized-engine-only"):
            torch_tensorrt.convert_method_to_trt_engine(
                model,
                inputs=[x],
                ir="dynamo",
                strict=True,
                offload_module_to_cpu=False,
            )


if __name__ == "__main__":
    run_tests()
