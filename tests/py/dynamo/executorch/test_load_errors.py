# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Load real TensorRT programs in ways that have to fail, and check each fails with a clear error."""

import importlib.util
import os
import subprocess
import sys

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402


@pytest.fixture(autouse=True)
def requires_runtime():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build a TensorRT engine")
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip(
            "needs the torch-tensorrt-executorch-runtime package to load the program"
        )


def _save(path, **kwargs):
    import torch_tensorrt

    torch.manual_seed(0)
    inputs = (torch.randn(4, 16, device="cuda"),)
    model = torch.nn.Linear(16, 16).eval().cuda()
    trt_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs), inputs=list(inputs), min_block_size=1
    )
    torch_tensorrt.save(
        trt_gm,
        str(path),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        **kwargs,
    )


def test_loading_with_no_visible_gpu_fails_with_a_clear_error(tmp_path):
    from executorch.exir import ExecutorchBackendConfig

    pte = tmp_path / "model.pte"
    # Planned in host memory, so loading reaches the delegate. A device-planned buffer would stop
    # the load earlier, in ExecuTorch's CUDA allocator, which reports a missing GPU on its own.
    _save(
        pte,
        backend_config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=False),
    )
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "import torch_tensorrt_executorch_runtime\n"
        "from executorch.runtime import Runtime\n"
        "program = Runtime.get().load_program(Path(sys.argv[1]))\n"
        "try:\n"
        "    program.load_method('forward')\n"
        "except RuntimeError as error:\n"
        "    print(error)\n"
        "    sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(pte)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 3, (result.returncode, result.stderr[-2000:])
    assert "CUDA is unavailable" in result.stdout + result.stderr
