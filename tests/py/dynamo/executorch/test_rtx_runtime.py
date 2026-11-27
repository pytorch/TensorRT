# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run with Torch-TensorRT-RTX and its separately built ExecuTorch companion."""

import subprocess
import sys

import pytest
import torch
import torch_tensorrt

if not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
    pytest.skip("Requires the RTX compiler", allow_module_level=True)
pytest.importorskip("executorch.exir")
pytest.importorskip("torch_tensorrt_executorch_rtx_runtime")

from executorch.runtime import Runtime  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_rtx_pte_inference(tmp_path, dynamic, cuda_graphs):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 1)

    model = Model().eval().cuda()
    inputs = [
        (
            torch_tensorrt.Input(min_shape=(1, 3), opt_shape=(2, 3), max_shape=(4, 3))
            if dynamic
            else torch_tensorrt.Input((2, 3))
        )
    ]
    with torch.no_grad():
        compiled = torch_tensorrt.compile(model, inputs=inputs, min_block_size=1)
        path = tmp_path / "model.pte"
        torch_tensorrt.save(
            compiled,
            str(path),
            output_format="executorch",
            arg_inputs=inputs,
            use_cuda_graphs=cuda_graphs,
        )
        runtime = Runtime.get()
        assert runtime.backend_registry.is_available("TensorRTRTXBackend")
        program = runtime.load_program(path)
        method = program.load_method("forward")
        for batch in ([1, 4, 2, 1] if dynamic else [2] * 4):
            # The exported method has a host interface; its device-copy nodes
            # stage the inputs and outputs around the CUDA delegate.
            x = torch.randn(batch, 3)
            for _ in range(3):
                actual = method.execute((x,))[0]
                torch.testing.assert_close(actual, torch.relu(x + 1))

        if not dynamic and not cuda_graphs:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
from pathlib import Path
import torch
import torch_tensorrt_executorch_rtx_runtime
from executorch.runtime import Runtime
program = Runtime.get().load_program(Path(sys.argv[1]))
x = torch.ones(2, 3)
output = program.load_method("forward").execute((x,))[0]
torch.testing.assert_close(output, torch.relu(x + 1))
assert not any(n == "torch_tensorrt" or n.startswith("torch_tensorrt.") for n in sys.modules)
""",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr


def test_rtx_with_cuda_fallback(tmp_path):
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir._serialize._program import deserialize_pte_binary

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cos(torch.erfinv(torch.tanh(x)))

    model = Model().eval().cuda()
    inputs = [torch.randn(64, 64, device="cuda") * 0.3]
    with torch.no_grad():
        compiled = torch_tensorrt.compile(model, inputs=inputs, min_block_size=1)
        path = tmp_path / "mixed.pte"
        torch_tensorrt.save(
            compiled,
            str(path),
            output_format="executorch",
            arg_inputs=inputs,
            partitioners=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
        )
        serialized = deserialize_pte_binary(path.read_bytes()).program
        delegates = {d.id for plan in serialized.execution_plan for d in plan.delegates}
        assert {"TensorRTRTXBackend", "CudaBackend"} <= delegates
        data_files = list(tmp_path.glob("*.ptd"))
        assert len(data_files) <= 1
        program = Runtime.get().load_program(
            path, data_path=data_files[0] if data_files else None
        )
        x = inputs[0].cpu()
        actual = program.load_method("forward").execute((x,))[0]
        torch.testing.assert_close(
            actual, torch.cos(torch.erfinv(torch.tanh(x))), rtol=1e-4, atol=1e-5
        )
