# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a ModelOpt-quantized model to a .pte, then load and run it with the ExecuTorch runtime."""

import importlib.util

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402

# Each precision's Model Optimizer config, and the lowest compute capability that has its kernels. TensorRT
# runs int8 on every GPU it supports.
_PRECISIONS = {"int8": ("INT8_DEFAULT_CFG", (0, 0)), "fp8": ("FP8_DEFAULT_CFG", (8, 9))}


@pytest.fixture(autouse=True)
def requires_runtime():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build and run a TensorRT engine")
    for package in ("torch_tensorrt_executorch_runtime", "modelopt"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"needs the {package} package")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(1024, 1024)
        self.down = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _save(path, inputs, precision=None):
    import modelopt.torch.quantization as mtq
    import torch_tensorrt
    from modelopt.torch.quantization.utils import export_torch_mode

    torch.manual_seed(0)
    model = _Model().eval().cuda()
    with torch.no_grad():
        if precision is not None:
            config = getattr(mtq, _PRECISIONS[precision][0])
            mtq.quantize(model, config, forward_loop=lambda m: m(*inputs))
        with export_torch_mode():
            exported = torch.export.export(model, inputs, strict=False)
            trt_gm = torch_tensorrt.dynamo.compile(
                exported, inputs=list(inputs), min_block_size=1
            )
    torch_tensorrt.save(
        trt_gm,
        str(path),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
    )
    return trt_gm


def _operators_and_delegates(pte_path):
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte_path.read_bytes()).program
    operators = {op.name for plan in program.execution_plan for op in plan.operators}
    delegates = {d.id for plan in program.execution_plan for d in plan.delegates}
    return operators, delegates


@pytest.mark.parametrize("precision", list(_PRECISIONS))
def test_a_quantized_engine_runs_in_executorch_like_its_compiled_module(
    tmp_path, precision
):
    """A quantized engine runs in ExecuTorch exactly as its compiled module does.

    The quantize and dequantize ops become TensorRT layers, so only TensorRTBackend runs the model
    (besides the host and device copies), and the engine keeps its weights in 8 bits.
    """
    if torch.cuda.get_device_capability() < _PRECISIONS[precision][1]:
        pytest.skip(f"TensorRT builds {precision} only on newer GPUs")
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    inputs = (torch.randn(16, 1024, device="cuda"),)
    quantized_pte, float_pte = tmp_path / "quantized.pte", tmp_path / "float.pte"
    trt_gm = _save(quantized_pte, inputs, precision)
    _save(float_pte, inputs)

    operators, delegates = _operators_and_delegates(quantized_pte)
    assert delegates == {"TensorRTBackend"}
    # Only the copies that move host inputs and outputs to and from the GPU run outside the engine.
    assert operators <= {"et_copy::_h2d_copy", "et_copy::_d2h_copy"}, operators
    # Two 1024x1024 float32 weights are 8 MiB, and 2 MiB in 8 bits. Half leaves room for the rest of the
    # engine, which differs between GPUs, and still fails if the weights were kept in float.
    assert quantized_pte.stat().st_size < float_pte.stat().st_size / 2

    # The program plans its inputs in host memory, so the caller passes a host tensor.
    method = Runtime.get().load_program(quantized_pte).load_method("forward")
    actual = method.execute((inputs[0].cpu(),))[0]
    with torch.no_grad():
        expected = trt_gm(*inputs)
    # The same engine bytes run in both, so the outputs match bit for bit.
    assert torch.equal(actual.cpu(), expected.cpu())
