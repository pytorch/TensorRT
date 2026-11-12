# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TensorRT's 256-byte address boundary, for buffers the caller passes and buffers the program plans.

A slice such as ``x[1:]`` starts one element past its storage's address. Bound to TensorRT as it is,
the engine's kernel faults and every later CUDA call in the process fails, so each run happens in a
child process and this one only reads what the child reports.

The delegate copies such a buffer through an aligned one. Buffers ``save()`` plans must not need
that copy, so their offsets are checked in the saved program.
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
def requires_cuda():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build a TensorRT engine")


class _AddMul(torch.nn.Module):
    def forward(self, x, y):
        return x * y + x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_an_input_off_the_256_byte_boundary_gives_the_right_answer(tmp_path, dtype):
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip("needs the torch_tensorrt_executorch_runtime package")
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


class _TwoEngines(torch.nn.Module):
    # 7 x 100 and 7 x 300 floats are not multiples of 256 bytes, so on ExecuTorch's default 16-byte
    # planning most buffers after the first start off the boundary. The multiply runs outside
    # TensorRT, which splits the graph into two engines with planned tensors between them.
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(100)
        self.fc1 = torch.nn.Linear(100, 300)
        self.fc2 = torch.nn.Linear(300, 100)

    def forward(self, x, gate):
        h = self.ln(x)
        a = torch.nn.functional.gelu(self.fc1(h))
        return self.fc2(a * gate) + h, a


def _planned_delegate_offsets(pte):
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DelegateCall, Tensor

    plan = deserialize_pte_binary(pte.read_bytes()).program.execution_plan[0]
    offsets = []
    for instruction in plan.chains[0].instructions:
        if not isinstance(instruction.instr_args, DelegateCall):
            continue
        for index in instruction.instr_args.args:
            value = plan.values[index].val
            if isinstance(value, Tensor) and value.allocation_info is not None:
                offsets.append(value.allocation_info.memory_offset)
    return offsets


@pytest.mark.parametrize("device_resident", [False, True], ids=["host-io", "device-io"])
def test_save_plans_every_delegate_buffer_on_the_256_byte_boundary(
    tmp_path, device_resident
):
    import torch_tensorrt
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    model = _TwoEngines().eval().cuda()
    inputs = (
        torch.randn(1, 7, 100, device="cuda"),
        torch.randn(1, 7, 300, device="cuda"),
    )
    with torch.no_grad():
        trt_gm = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
        )
    backend_config = None
    if device_resident:
        backend_config = ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    pte = tmp_path / "model.pte"
    torch_tensorrt.save(
        trt_gm,
        str(pte),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        backend_config=backend_config,
    )

    offsets = _planned_delegate_offsets(pte)
    assert len(offsets) >= 5, offsets
    assert all(offset % 256 == 0 for offset in offsets), offsets
    if backend_config is not None:
        # save() aligns a copy of the caller's planner, not the planner itself.
        assert (
            backend_config.memory_planning_pass.alignment
            == MemoryPlanningPass().alignment
        )
