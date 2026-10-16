# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a real TensorRT engine to a .pte plus a .ptd data file, then load and run it."""

import importlib.util

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402


@pytest.fixture(autouse=True)
def requires_runtime():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build and run a TensorRT engine")
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip(
            "needs the torch-tensorrt-executorch-runtime package to load the program"
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(self.linear(x))


def _compile(inputs):
    import torch_tensorrt

    torch.manual_seed(0)
    model = _Model().eval().cuda()
    exported = torch.export.export(model, inputs)
    return torch_tensorrt.dynamo.compile(
        exported, inputs=list(inputs), min_block_size=1
    )


def _engine_blobs(pte_path):
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte_path.read_bytes()).program
    return [
        program.backend_delegate_data[delegate.processed.index].data
        for plan in program.execution_plan
        for delegate in plan.delegates
        if delegate.id == "TensorRTBackend"
    ]


def test_an_engine_in_a_data_file_runs_like_the_embedded_engine(tmp_path):
    """One compiled engine, saved twice: embedded, and in engines.ptd.

    Both programs carry the same engine bytes, so they must produce the same output bit for bit.
    Comparing against a second, separately built engine would not be valid: two builds may choose
    different kernels.
    """
    import torch_tensorrt
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime
    from torch_tensorrt.executorch.serialization import deserialize_engine

    inputs = (torch.randn(8, 64, device="cuda"),)
    trt_gm = _compile(inputs)
    embedded_pte = tmp_path / "embedded" / "model.pte"
    external_pte = tmp_path / "external" / "model.pte"
    embedded_pte.parent.mkdir()
    external_pte.parent.mkdir()
    for path, tag in ((embedded_pte, None), (external_pte, "engines")):
        torch_tensorrt.save(
            trt_gm,
            str(path),
            output_format="executorch",
            retrace=False,
            arg_inputs=list(inputs),
            external_engine_data=tag,
        )

    ptd = external_pte.parent / "engines.ptd"
    assert ptd.is_file()
    assert not list(embedded_pte.parent.glob("*.ptd"))
    (embedded_blob,) = _engine_blobs(embedded_pte)
    (external_blob,) = _engine_blobs(external_pte)
    engine, metadata = deserialize_engine(external_blob)
    assert engine == b"" and metadata.engine_key
    embedded_engine, _ = deserialize_engine(embedded_blob)
    assert embedded_engine in ptd.read_bytes()
    assert (
        external_pte.stat().st_size
        < embedded_pte.stat().st_size - len(embedded_engine) // 2
    )

    runtime = Runtime.get()
    # The program plans its inputs in host memory, so the caller passes a host tensor.
    x = torch.randn(8, 64)
    expected = (
        runtime.load_program(embedded_pte).load_method("forward").execute((x,))[0]
    )
    actual = (
        runtime.load_program(external_pte, data_path=ptd)
        .load_method("forward")
        .execute((x,))[0]
    )
    assert torch.equal(actual.cpu(), expected.cpu())


def test_a_program_whose_engine_is_in_a_data_file_fails_to_load_without_it(tmp_path):
    import torch_tensorrt
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    inputs = (torch.randn(8, 64, device="cuda"),)
    pte = tmp_path / "model.pte"
    torch_tensorrt.save(
        _compile(inputs),
        str(pte),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        external_engine_data="engines",
    )
    program = Runtime.get().load_program(pte)
    # 0x24 is ExecuTorch's InvalidExternalData, the error the backend returns for a missing .ptd.
    with pytest.raises(RuntimeError, match="0x:?24"):
        program.load_method("forward")
