import importlib.util
import os
import shutil

import pytest

executorch = pytest.importorskip("executorch.exir")

import torch  # noqa: E402


def _cuda_backend_available() -> bool:
    # find_spec() imports the parent packages of a dotted path and raises
    # ModuleNotFoundError if a parent is absent (it only returns None when the
    # parent exists but the leaf module doesn't). Guard against that so collection
    # still succeeds where executorch.backends.cuda isn't packaged.
    try:
        spec = importlib.util.find_spec("executorch.backends.cuda.cuda_partitioner")
    except ModuleNotFoundError:
        return False
    return spec is not None


CUDA_BACKEND_AVAILABLE = _cuda_backend_available()


def _nvcc_available() -> bool:
    # The AOTInductor compile in the CUDA backend needs nvcc. It's on PATH in a
    # standard CUDA install, but some environments ship the toolkit via CUDA_HOME
    # and the compile resolves nvcc from there rather than PATH -- accept either.
    if shutil.which("nvcc") is not None:
        return True
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    for home in (CUDA_HOME, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if home and os.path.exists(os.path.join(home, "bin", "nvcc")):
            return True
    return False


def _skip_reason():
    # Building a coalesced TRT + CUDA .pte needs a GPU, ExecuTorch's CUDA backend,
    # and a CUDA toolkit (nvcc/ptxas) for the AOTInductor compile run during export.
    if not torch.cuda.is_available():
        return "CUDA GPU required for the CUDA backend"
    if not CUDA_BACKEND_AVAILABLE:
        return "executorch.backends.cuda not installed"
    if not _nvcc_available():
        return "nvcc required for the CUDA (AOTInductor) backend compile"
    return None


@pytest.fixture(autouse=True)
def requires_cuda_backend():
    # Evaluate the GPU/backend/nvcc gate at TEST RUNTIME, not at import. A
    # module-level pytest.mark.skipif is resolved during collection; on remote-GPU
    # runners collection happens off the GPU host, so the skip would be baked in
    # and the runner would never attach a GPU for the (pre-skipped) test.
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _cuda_partitioner():
    """A ``CudaPartitioner`` catch-all so ops TensorRT does not take route to the
    ExecuTorch CUDA (AOTInductor) backend. This is the supported, flag-free way to
    build a coalesced TensorRT + CUDA ``.pte`` via ``save(partitioners=[...])``."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    return CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])


def _deserialize_program(pte_path):
    from executorch.exir._serialize._program import deserialize_pte_binary

    return deserialize_pte_binary(pte_path.read_bytes()).program


def _delegate_ids(pte_path):
    """Backend ids of every delegate in the serialized program, in order."""
    program = _deserialize_program(pte_path)
    return [
        delegate.id for plan in program.execution_plan for delegate in plan.delegates
    ]


# NOTE: these tests are COMPOSITION-ONLY. They assert the serialized .pte carries
# both a TensorRTBackend and a CudaBackend delegate (and, below, that external
# CUDA weights are persisted as a .ptd). They do NOT load or run the program: a
# coalesced ATen-mode .pte cannot be loaded yet because memory-planned CUDA
# buffers get a CPU data pointer, so Method::init fails the CUDA backend's device
# check (tensor_parser_aten hardcodes CPU). A load-run-allclose test should be
# added once that runtime fix lands (follow-up).


def test_erfinv_routes_to_cuda_backend(tmp_path):
    """A genuinely TRT-unsupported op (erfinv) is routed to the CUDA backend via a
    CudaPartitioner catch-all -- no torch_executed_ops pin needed."""
    import torch_tensorrt

    # tanh keeps values in erfinv's (-1, 1) domain.
    class ErfinvModel(torch.nn.Module):
        def forward(self, x):
            return torch.cos(torch.erfinv(torch.tanh(x)))

    model = ErfinvModel().eval().to("cuda")
    inputs = (torch.randn(64, 64, device="cuda"),)
    exported = torch.export.export(model, inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        inputs=list(inputs),
        min_block_size=1,
        truncate_double=True,
    )

    out = tmp_path / "erfinv_cuda.pte"
    torch_tensorrt.save(
        trt_gm,
        str(out),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        partitioners=[_cuda_partitioner()],
    )

    delegate_ids = _delegate_ids(out)
    assert (
        "CudaBackend" in delegate_ids
    ), f"erfinv did not route to the CUDA backend; delegates={delegate_ids}"
    assert (
        "TensorRTBackend" in delegate_ids
    ), f"tanh/cos were not delegated to TensorRT; delegates={delegate_ids}"


def test_weighted_partition_persists_external_data(tmp_path):
    """A CUDA partition that carries weights must have its external data persisted
    next to the .pte.

    The CUDA (AOTInductor) backend emits weights as external named data
    (save_data_externally). If save() only writes the .pte and not the .ptd data
    file, the blob is lost and the program cannot load. Here mm(x, w) carries a
    weight and is pinned out of TensorRT via torch_executed_ops, so it lands on the
    CudaPartitioner and its weight becomes external CUDA data; tanh/cos stay on
    TensorRT. We assert a .ptd is written alongside the .pte."""
    import torch_tensorrt

    class WeightedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(64, 64))

        def forward(self, x):
            x = torch.mm(x, self.w)
            x = torch.tanh(x)
            return torch.cos(x)

    model = WeightedModel().eval().to("cuda")
    inputs = (torch.randn(64, 64, device="cuda"),)
    exported = torch.export.export(model, inputs)
    # Pin mm out of TensorRT so its weight lands on the CudaPartitioner catch-all
    # and is emitted as external CUDA data; tanh/cos still go to TensorRT.
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        inputs=list(inputs),
        min_block_size=1,
        truncate_double=True,
        torch_executed_ops={"torch.ops.aten.mm.default"},
    )

    out = tmp_path / "weighted_cuda.pte"
    torch_tensorrt.save(
        trt_gm,
        str(out),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        partitioners=[_cuda_partitioner()],
    )

    assert out.exists()
    delegate_ids = _delegate_ids(out)
    assert (
        "TensorRTBackend" in delegate_ids
    ), f"tanh/cos were not delegated to TensorRT; delegates={delegate_ids}"
    assert (
        "CudaBackend" in delegate_ids
    ), f"the pinned mm did not route to the CUDA backend; delegates={delegate_ids}"
    # Regression guard for dropped external CUDA weights: the .ptd data file must
    # be written next to the .pte.
    ptd_files = list(tmp_path.glob("*.ptd"))
    assert ptd_files, f"no external .ptd data file was written next to {out}"
    assert all(
        p.stat().st_size > 0 for p in ptd_files
    ), f"external .ptd data file is empty: {ptd_files}"


def test_multimethod_trt_export_preserves_methods(tmp_path):
    """Independent TRT programs remain separate delegated methods after lowering."""
    import torch_tensorrt

    class Prefill(torch.nn.Module):
        def forward(self, x):
            return torch.cos(x)

    class Decode(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    inputs = (torch.randn(16, 16, device="cuda"),)
    programs = {}
    source_snapshots = {}
    for name, model in (
        ("prefill", Prefill()),
        ("decode", Decode()),
    ):
        exported = torch.export.export(model.eval().to("cuda"), inputs)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported,
            inputs=list(inputs),
            min_block_size=1,
            truncate_double=True,
        )
        program = torch_tensorrt.dynamo._exporter.export(
            trt_gm,
            arg_inputs=inputs,
            use_legacy_exporter=False,
        )
        programs[name] = program
        source_snapshots[name] = (
            program.graph_module.code,
            tuple(program.state_dict),
            tuple(program.constants),
            tuple(node.target for node in program.graph.nodes),
        )

    edge = torch_tensorrt.executorch.export(programs)
    out = tmp_path / "multimethod_trt.pte"
    with out.open("wb") as output:
        edge.to_executorch().write_to_file(output)

    execution_plans = _deserialize_program(out).execution_plan
    assert {plan.name for plan in execution_plans} == {"prefill", "decode"}
    assert all(
        any(delegate.id == "TensorRTBackend" for delegate in plan.delegates)
        for plan in execution_plans
    )
    for name, program in programs.items():
        assert source_snapshots[name] == (
            program.graph_module.code,
            tuple(program.state_dict),
            tuple(program.constants),
            tuple(node.target for node in program.graph.nodes),
        )
        assert not any(key.startswith("_trt_engine_") for key in program.state_dict)


def test_graph_module_export_preserves_source(tmp_path):
    import torch_tensorrt

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cos(torch.tanh(x))

    inputs = (torch.randn(16, 16, device="cuda"),)
    exported = torch.export.export(Model().eval().to("cuda"), inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        inputs=list(inputs),
        min_block_size=1,
        truncate_double=True,
    )
    source_code = trt_gm.code
    source_state = tuple(trt_gm.state_dict())
    source_attrs = tuple(name for name, _ in trt_gm.named_modules())

    edge = torch_tensorrt.executorch.export(
        trt_gm,
        arg_inputs=inputs,
        retrace=False,
    )
    out = tmp_path / "graph_module_trt.pte"
    with out.open("wb") as output:
        edge.to_executorch().write_to_file(output)

    assert "TensorRTBackend" in _delegate_ids(out)
    assert trt_gm.code == source_code
    assert tuple(trt_gm.state_dict()) == source_state
    assert tuple(name for name, _ in trt_gm.named_modules()) == source_attrs
    assert not any(
        name.startswith("_trt_engine_") for name, _ in trt_gm.named_buffers()
    )


def test_trt_only_writes_no_ptd(tmp_path):
    """A TRT-only program (no CudaPartitioner) carries no external data, so no
    .ptd is written -- exercises the real write_to_file / _tensor_data path."""
    import torch_tensorrt

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cos(torch.tanh(x))

    model = Model().eval().to("cuda")
    inputs = (torch.randn(64, 64, device="cuda"),)
    exported = torch.export.export(model, inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        inputs=list(inputs),
        min_block_size=1,
        truncate_double=True,
    )

    out = tmp_path / "trt_only.pte"
    torch_tensorrt.save(
        trt_gm,
        str(out),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
    )

    assert out.exists()
    delegate_ids = _delegate_ids(out)
    assert (
        "CudaBackend" not in delegate_ids
    ), f"unexpected CUDA delegate for a TRT-only model; delegates={delegate_ids}"
    assert not list(
        tmp_path.glob("*.ptd")
    ), "TRT-only program must not write an external .ptd"
