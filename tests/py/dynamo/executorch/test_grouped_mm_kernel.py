# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A ``.pte`` whose TensorRT engine holds the grouped GEMM kernel, run by the ExecuTorch runtime.

The run happens in a child process that imports neither Triton nor Torch-TensorRT, as a deployed
program would.
"""

import importlib.util
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402

_CHILD = textwrap.dedent("""
    import sys

    import torch
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    x, offs, expected = torch.load(sys.argv[2])
    method = Runtime.get().load_program(sys.argv[1]).load_method("forward")
    torch.testing.assert_close(method.execute((x, offs))[0], expected, rtol=1e-2, atol=1e-2)
    assert "triton" not in sys.modules and "torch_tensorrt" not in sys.modules
    """)


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.randn(4, 64, 128, dtype=torch.bfloat16) * 0.05
        )

    def forward(self, x, offs):
        return torch.ops.aten._grouped_mm.default(x, self.w, offs)


def test_the_kernel_runs_in_the_executorch_runtime(tmp_path):
    # Decided at run time, not collection time, as in test_unaligned_io.py.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build a TensorRT engine")
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip("needs the torch_tensorrt_executorch_runtime package")
    import torch_tensorrt
    from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl

    if reason := aten_ops_converters._grouped_mm_kernel_unavailable():
        pytest.skip(f"the grouped GEMM kernel is unavailable: {reason}")
    torch.manual_seed(0)
    model = _Experts().eval().cuda()
    inputs = (
        torch.randn(128, 64, dtype=torch.bfloat16, device="cuda"),
        torch.tensor([10, 10, 100, 128], dtype=torch.int32, device="cuda"),
    )
    spy = mock.patch.object(
        impl.moe, "grouped_mm_kernel", wraps=impl.moe.grouped_mm_kernel
    )
    with torch.no_grad(), spy as kernel:
        trt_gm = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            grouped_mm_kernel=True,
        )
        expected = model(*inputs)
    assert kernel.called
    pte = tmp_path / "model.pte"
    torch_tensorrt.save(
        trt_gm,
        str(pte),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
    )
    data = tmp_path / "data.pt"
    torch.save((*(t.cpu() for t in inputs), expected.cpu()), data)
    subprocess.run([sys.executable, "-c", _CHILD, str(pte), str(data)], check=True)
