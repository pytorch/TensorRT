import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


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
            (16, 32, 24, 4, torch.bfloat16),
            (32, 64, 48, 8, torch.bfloat16),
        ]
    )
    def test_grouped_mm(self, rows, k, n, experts, dtype):
        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        counts = torch.randint(0, max(rows // experts, 1) + 1, (experts,))
        offs = torch.cumsum(counts, 0).clamp(max=rows).to(torch.int32)
        x = torch.randn(rows, k, dtype=dtype)
        w = torch.randn(experts, k, n, dtype=dtype)
        self.run_test(
            GroupedMM(),
            [x, w, offs],
            use_dynamo_tracer=True,
        )

    def test_grouped_mm_kwargs_offs(self):
        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs=offs)

        rows, k, n, experts = 16, 32, 24, 4
        counts = torch.tensor([3, 5, 0, 4])
        offs = torch.cumsum(counts, 0).clamp(max=rows).to(torch.int32)
        x = torch.randn(rows, k, dtype=torch.bfloat16)
        w = torch.randn(experts, k, n, dtype=torch.bfloat16)
        self.run_test(
            GroupedMM(),
            [x, w, offs],
            use_dynamo_tracer=True,
        )

    def test_grouped_mm_validator_keeps_dynamic_rows(self):
        # Checking a dynamic-rows grouped_mm must not specialize the rows
        # (e.g. the token count of a decode graph's MoE layer)
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            grouped_mm_validator,
        )

        class GroupedMM(nn.Module):
            def forward(self, x, w, offs):
                return torch.ops.aten._grouped_mm.default(x, w, offs)

        x = torch.randn(16, 32, dtype=torch.bfloat16)
        w = torch.randn(4, 32, 24, dtype=torch.bfloat16)
        offs = torch.tensor([4, 8, 12, 16], dtype=torch.int32)
        rows = torch.export.Dim("rows", min=2, max=64)
        ep = torch.export.export(
            GroupedMM(), (x, w, offs), dynamic_shapes=({0: rows}, None, None)
        )
        node = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten._grouped_mm.default
        )
        grouped_mm_validator(node)

        sym = node.args[0].meta["val"].shape[0].node
        value_range = sym.shape_env.var_to_range[sym.expr]
        self.assertLess(value_range.lower, value_range.upper)


if __name__ == "__main__":
    run_tests()
