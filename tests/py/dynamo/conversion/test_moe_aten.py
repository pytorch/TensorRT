import contextlib
import importlib.util
from unittest.mock import patch

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    grouped_mm_kernel_validator,
    grouped_mm_validator,
)
from torch_tensorrt.kernels import _triton

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestHistcConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((32,), 8, 0.0, 7.0),
            ((16, 4), 4, -1.0, 2.0),
            ((64,), 16, 0.0, 15.0),
        ]
    )
    def test_histc(self, shape, bins, lo, hi):
        class Histc(nn.Module):
            def forward(self, x):
                return torch.ops.aten.histc.default(x, bins, lo, hi)

        inputs = [torch.rand(*shape) * (hi - lo + 2) + lo - 1]
        self.run_test(Histc(), inputs, use_dynamo_tracer=True)

    def test_histc_integer_ids(self):
        class Histc(nn.Module):
            def forward(self, x):
                return torch.ops.aten.histc.default(x, 8, 0, 7)

        inputs = [torch.randint(0, 8, (64,), dtype=torch.int32)]
        self.run_test(Histc(), inputs, use_dynamo_tracer=True)


def _kernel_unavailable():
    if not torch.cuda.is_available():
        return "it needs a CUDA GPU"
    return aten_ops_converters._grouped_mm_kernel_unavailable()


_KERNEL_ON = CompilationSettings(grouped_mm_kernel=True)


def _decline_cause(name):
    if name == "old_triton":
        return patch.object(
            _triton, "_triton_import", side_effect=ImportError("Triton 3.4")
        )
    if name == "triton_driver":
        return patch.object(
            _triton._triton_import().runtime.driver.active,
            "get_current_target",
            side_effect=RuntimeError("no launcher"),
        )
    if name == "sm75":
        return patch.object(torch.cuda, "get_device_capability", return_value=(7, 5))
    if name == "no_ptx_limit":
        return patch.object(_triton, "_driver_max_ptx_version", return_value=None)
    return contextlib.nullcontext()


def _kernel_spy():
    return patch.object(impl.moe, "grouped_mm_kernel", wraps=impl.moe.grouped_mm_kernel)


class TestGroupedMMConverter(DispatchTestCase):
    # Each case checks which converter took the op.
    @parameterized.expand(
        [
            ("dense", False, 16, 32, 24, (3, 5, 0, 8), torch.bfloat16),
            (
                "dense_8_groups",
                False,
                32,
                64,
                48,
                (3, 5, 0, 4, 6, 0, 5, 9),
                torch.bfloat16,
            ),
            ("kernel", True, 16, 32, 24, (3, 5, 0, 8), torch.bfloat16),
            (
                "kernel_fp16_partial_tiles",
                True,
                300,
                40,
                136,
                (0, 129, 1, 0, 170),
                torch.float16,
            ),
        ]
    )
    def test_grouped_mm(self, _, kernel, rows, k, n, counts, dtype):
        if dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")
        if kernel:
            self._require_kernel()

        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        experts = len(counts)
        offs = torch.cumsum(torch.tensor(counts), 0).to(torch.int32)
        self.assertEqual(offs[-1].item(), rows, "Offsets must cover every output row.")
        x = torch.randn(rows, k, dtype=dtype)
        w = torch.randn(experts, k, n, dtype=dtype)
        # Float16 outputs near 6 are 0.004 apart, so a sum in another order can round
        # one step away from PyTorch.
        tolerance = {"rtol": 1e-2, "atol": 1e-2} if dtype == torch.float16 else {}
        with _kernel_spy() as spy:
            self.run_test(
                GroupedMM(),
                [x, w, offs],
                use_dynamo_tracer=True,
                grouped_mm_kernel=kernel,
                **tolerance,
            )
        self.assertEqual(spy.called, kernel)

    def test_grouped_mm_kwargs_offs(self):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs=offs)

        rows, k, n, experts = 16, 32, 24, 4
        counts = torch.tensor([3, 5, 0, 8])
        offs = torch.cumsum(counts, 0).to(torch.int32)
        self.assertEqual(offs[-1].item(), rows, "Offsets must cover every output row.")
        x = torch.randn(rows, k, dtype=torch.bfloat16)
        w = torch.randn(experts, k, n, dtype=torch.bfloat16)
        self.run_test(
            GroupedMM(),
            [x, w, offs],
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("small", 16, 32, 24, 4, True),
            ("large_experts_decode", 8, 3072, 2560, 128, True),
            ("at_limit", 16384, 1024, 1024, 16, True),
            ("above_limit", 16385, 1024, 1024, 16, False),
            ("large_experts_prefill", 28000, 3072, 2560, 128, False),
            ("one_group", 32768, 4096, 4096, 1, True),
        ]
    )
    def test_grouped_mm_validator_dense_cost(self, _, rows, k, n, experts, expected):
        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        # Only the shapes matter, so meta tensors keep the large cases cheap.
        args = (
            torch.empty(rows, k, device="meta", dtype=torch.bfloat16),
            torch.empty(experts, k, n, device="meta", dtype=torch.bfloat16),
            torch.empty(experts, device="meta", dtype=torch.int32),
        )
        ep = torch.export.export(GroupedMM(), args)
        node = next(
            nd
            for nd in ep.graph.nodes
            if nd.target == torch.ops.aten._grouped_mm.default
        )
        self.assertEqual(grouped_mm_validator(node), expected)
        self.assertEqual(grouped_mm_validator(node, CompilationSettings()), expected)
        full = CompilationSettings(require_full_compilation=True)
        self.assertTrue(grouped_mm_validator(node, full))
        no_limit = CompilationSettings(grouped_mm_dense_max_macs=None)
        self.assertTrue(grouped_mm_validator(node, no_limit))

    @parameterized.expand(
        [("false", False), ("true", True), ("zero", 0), ("negative", -1)]
    )
    def test_grouped_mm_dense_max_macs_rejects_non_positive(self, _, value):
        with self.assertRaises(ValueError):
            CompilationSettings(grouped_mm_dense_max_macs=value)

    def test_settings_saved_before_the_limit_keep_converting(self):
        state = CompilationSettings().__getstate__()
        del state["grouped_mm_dense_max_macs"]
        restored = CompilationSettings.__new__(CompilationSettings)
        restored.__setstate__(state)
        self.assertIsNone(restored.grouped_mm_dense_max_macs)

    @parameterized.expand(
        [
            ("bfloat16", torch.bfloat16, None, True),
            ("float32", torch.float32, None, False),
            ("bias", torch.bfloat16, ("bias", None), False),
            ("out_dtype_same", torch.bfloat16, ("out_dtype", torch.bfloat16), True),
            ("out_dtype_float32", torch.bfloat16, ("out_dtype", torch.float32), False),
            ("mat2_float16", torch.bfloat16, (1, torch.float16), False),
            ("offs_int64", torch.bfloat16, (2, torch.int64), False),
        ]
    )
    def test_grouped_mm_kernel_validator(self, _, dtype, change, expected):
        node = self._grouped_mm_node(dtype=dtype)
        # PyTorch's export rejects most of these, so the node is changed after export.
        if change is not None:
            where, value = change
            if isinstance(where, int):
                arg = node.args[where]
                arg.meta["val"] = arg.meta["val"].to(value)
            else:
                node.kwargs = {**node.kwargs, where: value or node.args[2]}
        self.assertEqual(grouped_mm_kernel_validator(node, _KERNEL_ON), expected)
        # The kernel is opt-in.
        self.assertFalse(grouped_mm_kernel_validator(node, CompilationSettings()))
        self.assertFalse(grouped_mm_kernel_validator(node))

    def _require_kernel(self):
        reason = _kernel_unavailable()
        if reason is not None:
            self.skipTest(f"the grouped GEMM kernel is unavailable: {reason}")

    def _grouped_mm_node(
        self,
        dynamic_shapes=None,
        groups=8,
        dtype=torch.bfloat16,
        rows=512,
        k=256,
        n=128,
    ):
        # Without the kernel's requirements every case declines at the first check,
        # so the negative cases would not test what they are named for.
        self._require_kernel()

        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        # Only the shapes matter, so meta tensors keep large cases cheap.
        args = (
            torch.empty(rows, k, device="meta", dtype=dtype),
            torch.empty(groups, k, n, device="meta", dtype=dtype),
            torch.empty(groups, device="meta", dtype=torch.int32),
        )
        ep = torch.export.export(GroupedMM(), args, dynamic_shapes=dynamic_shapes)
        return next(
            nd
            for nd in ep.graph.nodes
            if nd.target == torch.ops.aten._grouped_mm.default
        )

    def test_grouped_mm_kernel_validator_leaves_one_group_to_the_dense_lowering(self):
        # One group is a plain matrix multiply, which stays a TensorRT matmul.
        node = self._grouped_mm_node(groups=1)
        self.assertFalse(grouped_mm_kernel_validator(node, _KERNEL_ON))
        self.assertTrue(grouped_mm_validator(node, CompilationSettings()))

    @parameterized.expand(
        [
            ("old_triton", {}, "Triton 3.4"),
            ("triton_driver", {}, "no launcher"),
            ("sm75", {}, "8.0"),
            ("no_ptx_limit", {}, "cuda-python"),
            (
                "hardware_compatible",
                {"hardware_compatible": True},
                "hardware_compatible",
            ),
            ("windows", {"enable_cross_compile_for_windows": True}, "Windows"),
        ]
    )
    def test_grouped_mm_kernel_validator_says_why_it_declines(
        self, cause, settings, reason
    ):
        # The op declines with one warning that names the cause, however often the
        # registry asks.
        node = self._grouped_mm_node()
        aten_ops_converters._warn_grouped_mm_kernel_unused.cache_clear()
        self.addCleanup(aten_ops_converters._warn_grouped_mm_kernel_unused.cache_clear)
        on = CompilationSettings(grouped_mm_kernel=True, **settings)
        with (
            _decline_cause(cause),
            self.assertLogs(aten_ops_converters.__name__, level="WARNING") as logs,
        ):
            self.assertFalse(grouped_mm_kernel_validator(node, on))
            self.assertFalse(grouped_mm_kernel_validator(node, on))
        self.assertEqual(len(logs.records), 1)
        self.assertIn(reason, logs.output[0])
        # The dense converter takes the op instead.
        self.assertTrue(grouped_mm_validator(node, CompilationSettings()))

    @parameterized.expand(
        [
            ("unset", torch.bfloat16, {}, True),
            ("set", torch.bfloat16, {"grouped_mm_kernel": True}, False),
            (
                "hardware_compatible",
                torch.bfloat16,
                {"hardware_compatible": True},
                False,
            ),
            ("float32", torch.float32, {}, False),
            ("no_cuda_python", torch.bfloat16, {}, False),
            ("no_gpu", torch.bfloat16, {}, False),
        ]
    )
    def test_dense_warning_suggests_the_kernel_only_where_it_helps(
        self, name, dtype, settings, hint
    ):
        # Above grouped_mm_dense_max_macs, so the dense converter declines and warns.
        node = self._grouped_mm_node(
            dtype=dtype, groups=128, rows=28000, k=3072, n=2560
        )
        aten_ops_converters._warn_grouped_mm_left_to_pytorch.cache_clear()
        self.addCleanup(
            aten_ops_converters._warn_grouped_mm_left_to_pytorch.cache_clear
        )
        real_find_spec = importlib.util.find_spec
        missing = {
            "no_cuda_python": patch.object(
                importlib.util,
                "find_spec",
                side_effect=lambda module: (
                    None if module.startswith("cuda") else real_find_spec(module)
                ),
            ),
            # Without a GPU the dense validator must not raise.
            "no_gpu": patch.object(torch.cuda, "is_available", return_value=False),
        }.get(name, contextlib.nullcontext())
        # The hint must not start Triton or the CUDA driver for users who did not opt in.
        with (
            missing,
            patch.object(
                aten_ops_converters,
                "_grouped_mm_kernel_unavailable",
                side_effect=AssertionError("the hint started Triton"),
            ),
            self.assertLogs(aten_ops_converters.__name__, level="WARNING") as logs,
        ):
            self.assertFalse(
                grouped_mm_validator(node, CompilationSettings(**settings))
            )
        self.assertEqual("grouped_mm_kernel=True" in logs.output[-1], hint)


if __name__ == "__main__":
    run_tests()
