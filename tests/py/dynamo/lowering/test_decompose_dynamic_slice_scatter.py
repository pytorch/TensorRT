# type: ignore
"""End-to-end tests for the ``decompose_dynamic_slice_scatter`` lowering pass.

When ``slice_scatter``'s start/end/step is a SymInt (e.g. derived from a
dynamic dim), the static converter path doesn't apply. The lowering pass
rewrites the op into ``arange + view + expand + scatter`` so each piece
hits its existing dynamic-shape converter.
"""

import unittest

import torch
import torch_tensorrt
from torch.export import Dim, export
from torch.testing._internal.common_utils import TestCase, run_tests


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestDecomposeDynamicSliceScatter(TestCase):
    def _compile_and_check(self, model, sample_inputs, dynamic_shapes, retry_inputs):
        model = model.cuda().eval()
        sample_inputs = tuple(t.cuda() for t in sample_inputs)
        ep = export(model, sample_inputs, dynamic_shapes=dynamic_shapes)
        compiled = torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=list(sample_inputs),
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        for inputs in [sample_inputs, retry_inputs]:
            inputs = tuple(t.cuda() if t.device.type != "cuda" else t for t in inputs)
            out = compiled(*[t.clone() for t in inputs])
            ref = model(*[t.clone() for t in inputs])
            torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_dynamic_end_from_src_dim(self):
        """end = src.shape[dim] (SymInt). Writes the prefix of cache."""

        class M(torch.nn.Module):
            def forward(self, cache, src):
                L = src.shape[2]
                return torch.slice_scatter(cache, src, dim=2, start=0, end=L)

        cache = torch.zeros(1, 4, 32, 8)
        src = torch.ones(1, 4, 5, 8) * 3.0
        L_dim = Dim("L", min=1, max=16)
        retry = (torch.zeros(1, 4, 32, 8), torch.ones(1, 4, 9, 8) * 2.0)
        self._compile_and_check(
            M(), (cache, src), {"cache": None, "src": {2: L_dim}}, retry
        )

    def test_dynamic_start_and_end(self):
        """Both start and end are SymInts (from independent dynamic dims)."""

        class M(torch.nn.Module):
            def forward(self, cache, offset, src):
                # offset's leading dim provides a SymInt write position;
                # update_len is src.shape[dim]
                start = offset.shape[0]
                end = start + src.shape[2]
                return torch.slice_scatter(cache, src, dim=2, start=start, end=end)

        cache = torch.zeros(1, 4, 32, 8)
        offset = torch.zeros(3)  # start=3
        src = torch.ones(1, 4, 4, 8) * 7.0
        S_dim = Dim("S", min=0, max=20)
        L_dim = Dim("L", min=1, max=8)
        retry = (
            torch.zeros(1, 4, 32, 8),
            torch.zeros(10),
            torch.ones(1, 4, 6, 8) * 1.5,
        )
        self._compile_and_check(
            M(),
            (cache, offset, src),
            {"cache": None, "offset": {0: S_dim}, "src": {2: L_dim}},
            retry,
        )

    def test_dim_zero_dynamic(self):
        """dim=0 with dynamic end."""

        class M(torch.nn.Module):
            def forward(self, cache, src):
                L = src.shape[0]
                return torch.slice_scatter(cache, src, dim=0, start=0, end=L)

        cache = torch.zeros(16, 8)
        src = torch.ones(3, 8) * 9.0
        L_dim = Dim("L", min=1, max=8)
        retry = (torch.zeros(16, 8), torch.ones(7, 8) * 4.0)
        self._compile_and_check(
            M(), (cache, src), {"cache": None, "src": {0: L_dim}}, retry
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestStaticPathPreserved(TestCase):
    """All-int start/end/step still hits the converter directly (KV fast path
    or static scatter fallback), not the lowering decomposition."""

    def test_static_slice_scatter_untouched(self):
        class M(torch.nn.Module):
            def forward(self, cache, src):
                return torch.slice_scatter(cache, src, dim=2, start=3, end=4)

        model = M().cuda().eval()
        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        src = torch.ones(1, 4, 1, 8, device="cuda") * 5.0
        ep = export(model, (cache, src))
        # The lowering pass should leave this node alone — confirm by
        # checking the post-lowering graph still contains slice_scatter.
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
            post_lowering,
        )

        gm = post_lowering(ep.module(), CompilationSettings())
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.slice_scatter.default, targets)

        compiled = torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=[cache, src],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        out = compiled(cache.clone(), src)
        ref = model(cache.clone(), src)
        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    run_tests()
