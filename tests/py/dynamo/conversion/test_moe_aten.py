import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import grouped_mm_validator

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


class TestGroupedMMConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (16, 32, 24, (3, 5, 0, 8), torch.bfloat16),
            (32, 64, 48, (3, 5, 0, 4, 6, 0, 5, 9), torch.bfloat16),
        ]
    )
    def test_grouped_mm(self, rows, k, n, counts, dtype):
        if dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        experts = len(counts)
        offs = torch.cumsum(torch.tensor(counts), 0).to(torch.int32)
        self.assertEqual(offs[-1].item(), rows, "Offsets must cover every output row.")
        x = torch.randn(rows, k, dtype=dtype)
        w = torch.randn(experts, k, n, dtype=dtype)
        self.run_test(
            GroupedMM(),
            [x, w, offs],
            use_dynamo_tracer=True,
        )

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


if __name__ == "__main__":
    run_tests()
