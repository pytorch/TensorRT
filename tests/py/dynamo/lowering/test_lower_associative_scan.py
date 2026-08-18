# type: ignore
"""Tests for ``lower_associative_scan`` (Mamba affine recurrence).

``combine_mode="pointwise"`` keeps ``higher_order.associative_scan`` through
export. The pass matches only the combine ``(a_l*a_r, a_r*b_l + b_r)`` with a
static scan length and replaces it with a Hillis–Steele aten scan.
"""

import math
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering, pre_export_lowering
from torch_tensorrt.dynamo.lowering.passes.lower_associative_scan import (
    _is_mamba_affine_combine,
    lower_associative_scan,
)


def _mamba_scan_module(combine_mode: str) -> torch.nn.Module:
    class MambaScan(torch.nn.Module):
        def __init__(self, mode: str):
            super().__init__()
            self.combine_mode = mode

        def forward(self, discrete_a, delta_b_u, c):
            from torch._higher_order_ops.associative_scan import associative_scan

            def combine_fn(left, right):
                a_left, b_left = left
                a_right, b_right = right
                return (a_left * a_right, a_right * b_left + b_right)

            _, all_h = associative_scan(
                combine_fn,
                (discrete_a, delta_b_u),
                dim=2,
                combine_mode=self.combine_mode,
            )
            return (
                torch.matmul(all_h.permute(0, 2, 1, 3), c.unsqueeze(-1))
                .squeeze(-1)
                .permute(0, 2, 1)
            )

    return MambaScan(combine_mode)


def _has_associative_scan(gm: torch.fx.GraphModule) -> bool:
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        name = str(n.target)
        if "associative_scan" in name and "higher_order" in name:
            return True
        if n.target is getattr(torch.ops.higher_order, "associative_scan", None):
            return True
    return False


def _lower_exported(model, inputs, experimental: bool = False):
    settings = CompilationSettings(min_block_size=1)
    with torch.no_grad():
        ep = torch.export.export(model, inputs)
    ep = pre_export_lowering(ep, settings)
    ep = ep.run_decompositions(get_decompositions(experimental))
    return post_lowering(ep.module(), settings), ep


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestLowerAssociativeScan(TestCase):
    def test_pointwise_scan_removed_from_graph(self):
        b, d, s, n = 1, 4, 8, 16
        model = _mamba_scan_module("pointwise").cuda().eval()
        inputs = (
            torch.rand(b, d, s, n, device="cuda"),
            torch.randn(b, d, s, n, device="cuda"),
            torch.randn(b, s, n, device="cuda"),
        )
        gm, _ = _lower_exported(model, inputs)
        self.assertFalse(
            _has_associative_scan(gm),
            f"associative_scan still present after lowering:\n{gm.graph}",
        )
        # Parallel scan stages should introduce doubling slices/cats.
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.aten.slice.Tensor, targets)
        self.assertIn(torch.ops.aten.cat.default, targets)
        self.assertIn(torch.ops.aten.mul.Tensor, targets)
        self.assertIn(torch.ops.aten.add.Tensor, targets)

    def test_pointwise_scan_numerics_match_eager(self):
        b, d, s, n = 1, 2, 8, 4
        model = _mamba_scan_module("pointwise").cuda().eval()
        inputs = (
            torch.rand(b, d, s, n, device="cuda"),
            torch.randn(b, d, s, n, device="cuda"),
            torch.randn(b, s, n, device="cuda"),
        )
        gm, ep = _lower_exported(model, inputs)
        self.assertFalse(_has_associative_scan(gm))

        ref = model(*[t.clone() for t in inputs])
        # Execute the lowered GraphModule (aten scan, no HOP).
        out = gm(*[t.clone() for t in inputs])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

        compiled = torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(inputs),
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        trt_out = compiled(*[t.clone() for t in inputs])
        torch.testing.assert_close(trt_out, ref, rtol=1e-3, atol=1e-3)

    def test_non_power_of_two_length(self):
        b, d, s, n = 1, 2, 7, 4
        model = _mamba_scan_module("pointwise").cuda().eval()
        inputs = (
            torch.rand(b, d, s, n, device="cuda"),
            torch.randn(b, d, s, n, device="cuda"),
            torch.randn(b, s, n, device="cuda"),
        )
        gm, _ = _lower_exported(model, inputs)
        self.assertFalse(_has_associative_scan(gm))
        torch.testing.assert_close(
            gm(*[t.clone() for t in inputs]),
            model(*[t.clone() for t in inputs]),
            rtol=1e-4,
            atol=1e-4,
        )
        # ceil(log2(7)) == 3 stages
        self.assertEqual(math.ceil(math.log2(s)), 3)

    def test_decline_non_mamba_combine(self):
        """A different associative combine must keep the HOP."""

        class SumScan(torch.nn.Module):
            def forward(self, x):
                from torch._higher_order_ops.associative_scan import associative_scan

                def combine_fn(left, right):
                    return left + right

                return associative_scan(
                    combine_fn, x, dim=0, combine_mode="pointwise"
                )

        model = SumScan().cuda().eval()
        x = torch.randn(8, 4, device="cuda")
        settings = CompilationSettings(min_block_size=1)
        with torch.no_grad():
            ep = torch.export.export(model, (x,))
        ep = pre_export_lowering(ep, settings)
        ep = ep.run_decompositions(get_decompositions(False))
        gm = ep.module()

        # Confirm a scan HOP is present, then that our pass declines it.
        self.assertTrue(_has_associative_scan(gm))
        gm2 = lower_associative_scan(gm, settings)
        self.assertTrue(_has_associative_scan(gm2))

    def test_mamba_combine_matcher_unit(self):
        """Direct unit check of the narrow combine matcher."""

        class Combine(torch.nn.Module):
            def forward(self, a_l, b_l, a_r, b_r):
                return a_l * a_r, a_r * b_l + b_r

        traced = torch.fx.symbolic_trace(Combine())
        self.assertTrue(_is_mamba_affine_combine(traced))

        class BadCombine(torch.nn.Module):
            def forward(self, a_l, b_l, a_r, b_r):
                return a_l + a_r, b_l + b_r

        self.assertFalse(_is_mamba_affine_combine(torch.fx.symbolic_trace(BadCombine())))


if __name__ == "__main__":
    run_tests()
