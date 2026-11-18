# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch_tensorrt as torchtrt


class AffineRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.arange(8, dtype=torch.float32))

    def forward(self, x):
        return torch.relu(x * 2 + self.bias)


@torch.no_grad()
def _compile_and_run():
    model = AffineRelu().eval().cuda()
    scripted = torch.jit.script(model)
    spec = {
        "forward": torchtrt.ts.TensorRTCompileSpec(
            inputs=[torchtrt.Input((2, 8), dtype=torch.float32)],
            enabled_precisions={torch.float32},
            disable_tf32=True,
        )
    }
    lowered = torch._C._jit_to_backend("tensorrt", scripted, spec)
    for offset in (0.0, -4.0):
        sample = torch.randn(2, 8, device="cuda") + offset
        torch.testing.assert_close(lowered(sample), model(sample))
    torch.cuda.synchronize()
    return lowered


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript frontend is unavailable",
)
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
class TestToBackendFactory(unittest.TestCase):
    def test_engine_execution_and_teardown(self):
        # A child process also catches heap corruption during interpreter shutdown.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    for _ in range(2):
        lowered = _compile_and_run()
        del lowered
        gc.collect()
    lowered = _compile_and_run()
