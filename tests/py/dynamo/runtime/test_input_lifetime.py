"""Regression test for the C++ runtime use-after-free of contiguified
input copies (core/runtime/execute_engine.cpp).

Bug: setup_input_tensors stashed `.contiguous()` copies in a function-
local std::list. After return, those copies were freed; the CUDA caching
allocator recycled their addresses for the engine's output buffer when
shape+dtype matched. TRT's input bindings then aliased onto outputs,
corrupting reads-after-writes inside the engine.

Fix: hoist `formatted_inputs` to a scope that outlives `enqueueV3`.

Trigger conditions reproduced here:
  - Input is non-contiguous → setup_input_tensors calls .contiguous()
    and allocates a fresh CUDA buffer.
  - Output shape+dtype matches the input → caching allocator can
    recycle the freed input buffer for the output.
  - Model has a residual that re-reads the input at the very end →
    intermediate scratch writes between the early and late reads of x
    can corrupt x via the aliased buffer.
"""

import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

B, S, H = 1, 4096, 128
NUM_BLOCKS = 8
NUM_TRIALS = 3


class _ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, 4 * d, bias=False)
        self.lin2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return x + self.lin2(torch.nn.functional.silu(self.lin1(self.norm(x))))


class _Model(nn.Module):
    """Small residual stack. The trailing `+ x` forces TRT's engine to
    keep the input alive across the entire computation; the residual
    blocks add intermediate scratch writes between early and late reads.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock(H) for _ in range(NUM_BLOCKS)])

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        return h + x


def _make_noncontig_input(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(B, H, S, device="cuda", dtype=torch.bfloat16, generator=g)
    x = base.transpose(1, 2)
    assert tuple(x.shape) == (B, S, H)
    assert not x.is_contiguous()
    return x


class TestInputLifetime(TestCase):
    @parameterized.expand(
        [
            ("cpp_runtime", False),
            ("python_runtime", True),
        ]
    )
    def test_noncontig_input_matching_output_shape(self, _name, use_python_runtime):
        torch.manual_seed(0)
        model = _Model().to(device="cuda", dtype=torch.bfloat16).eval()
        x = _make_noncontig_input(seed=1)

        with torch.inference_mode():
            eager_out = model(x)

        torch._dynamo.reset()
        compiled = torch.compile(
            model,
            backend="tensorrt",
            fullgraph=False,
            options={
                "truncate_double": True,
                "enabled_precisions": {torch.float, torch.half, torch.bfloat16},
                "min_block_size": 1,
                "optimization_level": 1,
                "enable_resource_partitioning": True,
                "use_python_runtime": use_python_runtime,
            },
        )

        with torch.inference_mode():
            compiled(x)  # compile / warmup
            for trial in range(NUM_TRIALS):
                trt_out = compiled(x)
                diff = (eager_out.float() - trt_out.float()).abs()
                mean = diff.mean().item()
                mx = diff.max().item()
                frac_gt_1 = (diff > 1.0).float().mean().item()
                # bf16 vs fp32 numerical drift on this small model is well
                # under these bounds; the pre-fix divergence was mean>1.5,
                # max>10, >1.0 over 60% of values. Tight bounds keep the
                # test sensitive while leaving room for legitimate bf16 noise.
                self.assertLess(
                    mean,
                    0.05,
                    f"trial {trial}: mean_abs_diff {mean:.4f} suggests input/output aliasing "
                    f"(use-after-free of contiguified input copy in execute_engine.cpp)",
                )
                self.assertLess(mx, 5.0, f"trial {trial}: max_abs_diff {mx:.4f}")
                self.assertLess(
                    frac_gt_1,
                    0.005,
                    f"trial {trial}: {100*frac_gt_1:.2f}% of outputs differ by >1.0",
                )


if __name__ == "__main__":
    run_tests()
