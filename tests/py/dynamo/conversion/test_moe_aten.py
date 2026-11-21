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
            ("at_limit", 16384, 1024, 1024, 16, True),
            ("above_limit", 16385, 1024, 1024, 16, False),
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
        limit = CompilationSettings(grouped_mm_dense_max_macs=2**38)
        self.assertEqual(grouped_mm_validator(node, limit), expected)
        full = CompilationSettings(
            require_full_compilation=True, grouped_mm_dense_max_macs=2**38
        )
        self.assertTrue(grouped_mm_validator(node, full))
        # The limit is opt-in: a default compile converts the op as before.
        self.assertTrue(grouped_mm_validator(node))

    def test_grouped_mm_validator_declines_dynamic_rows(self):
        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        args = (
            torch.empty(64, 32, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 32, 24, device="meta", dtype=torch.bfloat16),
            torch.empty(4, device="meta", dtype=torch.int32),
        )
        rows = torch.export.Dim("rows", min=2, max=4096)
        ep = torch.export.export(
            GroupedMM(), args, dynamic_shapes=({0: rows}, None, None)
        )
        node = next(
            nd
            for nd in ep.graph.nodes
            if nd.target == torch.ops.aten._grouped_mm.default
        )
        # The dense lowering would pin the row count, even where the settings would
        # otherwise keep the op in TensorRT.
        settings = CompilationSettings(
            assume_dynamic_shape_support=True, require_full_compilation=True
        )
        # The registry asks more than once per node, and the op leaves the engine, so
        # one warning per node says why.
        with self.assertLogs(
            "torch_tensorrt.dynamo.conversion.aten_ops_converters", level="WARNING"
        ) as logs:
            self.assertFalse(grouped_mm_validator(node, settings))
            self.assertFalse(grouped_mm_validator(node, settings))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("needs static shapes", logs.output[0])

    @parameterized.expand(
        [
            ("true", True),
            ("zero", 0),
            ("negative", -1),
            ("float", 2.5),
        ]
    )
    def test_grouped_mm_dense_max_macs_rejects_non_positive_integers(self, _, value):
        with self.assertRaises(ValueError):
            CompilationSettings(grouped_mm_dense_max_macs=value)


if __name__ == "__main__":
    run_tests()
