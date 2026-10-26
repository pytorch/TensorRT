# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a device-resident .pte on a CUDA input that is not on TensorRT's 256-byte boundary.

A slice such as ``x[1:]`` starts one element past its storage's address. Bound to TensorRT as it is,
the engine's kernel faults and every later CUDA call in the process fails, so each run happens in a
child process and this one only reads what the child reports.
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402

_CHILD = textwrap.dedent("""
    import sys

    import torch
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    pte, data, offset = sys.argv[1], sys.argv[2], int(sys.argv[3])
    x, y, expected = torch.load(data)

    def at_offset(t):
        storage = torch.empty(t.numel() + 256, dtype=t.dtype, device="cuda")
        view = storage[offset : offset + t.numel()].view(t.shape)
        view.copy_(t)
        return view

    inputs = (at_offset(x), at_offset(y))
    assert (inputs[0].data_ptr() % 256 == 0) == (offset == 0), inputs[0].data_ptr()
    method = Runtime.get().load_program(pte).load_method("forward")
    actual = method.execute(inputs)[0]
    torch.cuda.synchronize()
    assert torch.equal(actual.cpu(), expected), (actual.cpu() - expected).abs().max()
    """)


@pytest.fixture(autouse=True)
def requires_runtime():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build and run a TensorRT engine")
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip("needs the torch_tensorrt_executorch_runtime package")


class _AddMul(torch.nn.Module):
    def forward(self, x, y):
        return x * y + x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_an_input_off_the_256_byte_boundary_gives_the_right_answer(tmp_path, dtype):
    import torch_tensorrt
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    torch.manual_seed(0)
    inputs = tuple(torch.randn(1 << 20, device="cuda", dtype=dtype) for _ in range(2))
    with torch.no_grad():
        exported = torch.export.export(_AddMul().eval(), inputs)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=list(inputs), min_block_size=1
        )
        expected = trt_gm(*inputs)
    pte = tmp_path / "model.pte"
    # The caller hands in its own CUDA tensors, which is how an unaligned address reaches the engine.
    torch_tensorrt.save(
        trt_gm,
        str(pte),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        backend_config=ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    data = tmp_path / "io.pt"
    torch.save((*inputs, expected.cpu()), data)

    # Aligned first, so a failure at one element off is the offset and not the program.
    for offset in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, str(pte), str(data), str(offset)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert (
            result.returncode == 0
        ), f"offset {offset}:\n{result.stdout}\n{result.stderr}"
