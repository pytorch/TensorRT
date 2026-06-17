"""Regression coverage for C++ runtime input buffer lifetimes.

The C++ runtime binds TensorRT inputs to raw data pointers. If input setup
creates a temporary contiguous tensor, the runtime must keep that tensor alive
until TensorRT has finished using the pointer. Otherwise the CUDA caching
allocator can recycle the freed input buffer for an output tensor with the same
shape and dtype, causing TensorRT input and output bindings to alias.
"""

import unittest

import torch
import torch.nn as nn
import torch_tensorrt
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
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock(H) for _ in range(NUM_BLOCKS)])

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h + x


def _make_noncontig_input(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(B, H, S, device="cuda", dtype=torch.bfloat16, generator=g)
    x = base.transpose(1, 2)
    assert tuple(x.shape) == (B, S, H)
    assert not x.is_contiguous()
    return x


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestInputLifetime(TestCase):
    def setUp(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("standard",),
            ("cudagraphs",),
            ("output_allocator",),
            ("pre_allocated_outputs",),
            ("pre_allocated_outputs_cudagraphs",),
        ]
    )
    def test_noncontig_input_matching_output_shape_cpp_runtime(self, runtime_mode):
        torch.manual_seed(0)
        model = _Model().to(device="cuda", dtype=torch.bfloat16).eval()
        x = _make_noncontig_input(seed=1)

        with torch.inference_mode():
            eager_out = model(x)

        compiled = torch_tensorrt.compile(
            model,
            "dynamo",
            [torch_tensorrt.Input(shape=(B, S, H), dtype=torch.bfloat16)],
            truncate_double=True,
            enabled_precisions={torch.float, torch.half, torch.bfloat16},
            min_block_size=1,
            optimization_level=1,
            enable_resource_partitioning=True,
            use_python_runtime=False,
        )

        with torch.inference_mode():
            compiled(x)
            for trial in range(NUM_TRIALS):
                if runtime_mode == "cudagraphs":
                    with torch_tensorrt.runtime.enable_cudagraphs(
                        compiled
                    ) as cg_module:
                        trt_out = cg_module(x)
                elif runtime_mode == "output_allocator":
                    with torch_tensorrt.runtime.enable_output_allocator(compiled):
                        trt_out = compiled(x)
                elif runtime_mode == "pre_allocated_outputs":
                    with torch_tensorrt.runtime.enable_pre_allocated_outputs(compiled):
                        trt_out = compiled(x)
                elif runtime_mode == "pre_allocated_outputs_cudagraphs":
                    with torch_tensorrt.runtime.enable_pre_allocated_outputs(compiled):
                        with torch_tensorrt.runtime.enable_cudagraphs(
                            compiled
                        ) as cg_module:
                            trt_out = cg_module(x)
                else:
                    trt_out = compiled(x)

                diff = (eager_out.float() - trt_out.float()).abs()
                mean = diff.mean().item()
                mx = diff.max().item()
                frac_gt_1 = (diff > 1.0).float().mean().item()

                self.assertLess(
                    mean,
                    0.05,
                    f"{runtime_mode} trial {trial}: mean_abs_diff {mean:.4f} suggests input/output aliasing",
                )
                self.assertLess(
                    mx, 5.0, f"{runtime_mode} trial {trial}: max_abs_diff {mx:.4f}"
                )
                self.assertLess(
                    frac_gt_1,
                    0.005,
                    f"{runtime_mode} trial {trial}: {100 * frac_gt_1:.2f}% of outputs differ by >1.0",
                )


if __name__ == "__main__":
    run_tests()
