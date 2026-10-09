# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the installed Windows delegate through ExecuTorch's Python bindings."""

import sys

import pytest
import torch
import torch_tensorrt

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows runtime integration"
)


@pytest.mark.unit
def test_windows_export_and_execute(tmp_path):
    # CI's Windows GPU lane must fail rather than pass through an unexercised skip.
    assert (
        torch.cuda.is_available()
    ), "The Windows ExecuTorch lane requires a CUDA device"
    assert torch.version.cuda in {"13.2", "13.4"}

    import torch_tensorrt_executorch_runtime
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    assert runtime.backend_registry.is_available(
        torch_tensorrt_executorch_runtime.BACKEND_NAME
    )
    # These must survive loading an out-of-tree delegate as well.
    assert runtime.backend_registry.is_available("CudaBackend")
    assert runtime.backend_registry.is_available("XnnpackBackend")

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 1)

    model = Model().eval().cuda()
    example = torch.randn(2, 8, device="cuda")
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[example],
        min_block_size=1,
        enabled_precisions={torch.float32},
    )
    path = tmp_path / "windows.pte"
    torch_tensorrt.save(
        compiled, str(path), output_format="executorch", inputs=[example]
    )
    method = runtime.load_program(path).load_method("forward")
    for _ in range(2):
        result = method.execute((example.cpu(),))
        torch.testing.assert_close(result[0].cpu(), model(example).cpu())
