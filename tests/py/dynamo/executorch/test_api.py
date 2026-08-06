import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensor
from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj, lift


@pytest.mark.unit
def test_lazy_import_error_when_executorch_missing(monkeypatch):
    original_module = sys.modules.pop("torch_tensorrt.executorch", None)
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    module = importlib.import_module("torch_tensorrt.executorch")

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        _ = module.TensorRTBackend

    sys.modules.pop("torch_tensorrt.executorch", None)
    if original_module is not None:
        sys.modules["torch_tensorrt.executorch"] = original_module


@pytest.mark.unit
def test_save_executorch_error_when_executorch_missing(monkeypatch, tmp_path):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    from torch_tensorrt._compile import save

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
        )


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SETUP_PY = _REPO_ROOT / "setup.py"


def _setup_tree():
    return ast.parse(_SETUP_PY.read_text(encoding="utf-8"))


def _assignment_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"Could not find assignment for {name}")


def _function_def(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


@pytest.mark.unit
def test_packaging_declares_executorch_extra():
    tree = _setup_tree()
    extras = _assignment_value(tree, "EXTRAS_REQUIRE")
    assert isinstance(extras, ast.Dict)

    extras_by_name = {
        key.value: value
        for key, value in zip(extras.keys, extras.values)
        if isinstance(key, ast.Constant)
    }
    for extra_name in ("executorch", "all"):
        assert extra_name in extras_by_name
        requirements = extras_by_name[extra_name]
        assert isinstance(requirements, ast.List)
        assert any(
            isinstance(requirement, ast.Name)
            and requirement.id == "EXECUTORCH_REQUIREMENT"
            for requirement in requirements.elts
        )

    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    extras_keyword = next(
        (keyword for keyword in setup_call.keywords if keyword.arg == "extras_require"),
        None,
    )
    assert extras_keyword is not None
    assert isinstance(extras_keyword.value, ast.Name)
    assert extras_keyword.value.id == "EXTRAS_REQUIRE"


@pytest.mark.unit
def test_executorch_is_not_base_install_requirement():
    tree = _setup_tree()
    for function_name in (
        "get_jetpack_requirements",
        "get_sbsa_requirements",
        "get_x86_64_requirements",
        "get_requirements",
    ):
        function = _function_def(tree, function_name)
        assert not any(
            isinstance(node, ast.Name) and node.id == "EXECUTORCH_REQUIREMENT"
            for node in ast.walk(function)
        )


@pytest.mark.unit
def test_executorch_headers_are_not_dlfw_gated():
    tree = _setup_tree()
    header_package_data = _assignment_value(tree, "executorch_header_package_data")
    assert isinstance(header_package_data, ast.List)
    assert not any(
        isinstance(node, ast.Name) and node.id == "IS_DLFW_CI"
        for node in ast.walk(header_package_data)
    )


def _stub_node(name, target=None):
    return types.SimpleNamespace(name=name, target=name if target is None else target)


def _stub_exported_program(constants, name_to_fqn=None):
    sig = (
        None
        if name_to_fqn is None
        else types.SimpleNamespace(inputs_to_lifted_custom_objs=name_to_fqn)
    )
    return types.SimpleNamespace(constants=constants, graph_signature=sig)


@pytest.mark.unit
def test_resolve_lifted_custom_obj_via_signature_fqn():
    # Modern torch.export: placeholder name differs from the constants FQN key.
    sentinel = object()
    ep = _stub_exported_program({"engine_fqn": sentinel}, {"obj_engine": "engine_fqn"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("obj_engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_legacy_fallback():
    # No signature mapping: fall back to a direct name/target lookup.
    sentinel = object()
    ep = _stub_exported_program({"engine": sentinel}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_signature_present_name_absent_is_none():
    # A present-but-incomplete mapping must not bind a different object by name.
    ep = _stub_exported_program({"engine": object()}, {"some_other_obj": "x"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_missing_is_none():
    ep = _stub_exported_program({}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("missing")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_unwraps_fake_script_object():
    class _Real:
        pass

    fake = FakeScriptObject(object(), "Engine", _Real())
    ep = _stub_exported_program({"engine_fqn": fake}, {"obj_engine": "engine_fqn"})
    resolved = _resolve_lifted_custom_obj(ep, _stub_node("obj_engine"))
    assert not isinstance(resolved, FakeScriptObject)
    assert isinstance(resolved, _Real)


# --- per-partition target_device (TensorRTPartitioner) -----------------------
# These exercise the partitioner directly, so they need ExecuTorch installed;
# they run in the dedicated executorch CI job and skip elsewhere.


@pytest.mark.unit
def test_resolve_target_device_uses_partition_engine(monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    engine_node = object()
    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [engine_node])
    info = ["0"] * (DEVICE_IDX + 1)
    info[DEVICE_IDX] = "2"
    monkeypatch.setattr(P, "_get_engine_info_for_node", lambda ep, n: info)

    partition = types.SimpleNamespace(id=0, nodes=[engine_node])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:2"


@pytest.mark.unit
def test_resolve_target_device_falls_back_when_not_one_engine(monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    partition = types.SimpleNamespace(id=1, nodes=[])

    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:0"

    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [object(), object()])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:0"


@pytest.mark.unit
def test_per_partition_distinct_target_devices(monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    # Each partition's engine node carries its own device id as its value.
    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [nodes[0]])

    def fake_info(ep, node):
        info = ["0"] * (DEVICE_IDX + 1)
        info[DEVICE_IDX] = str(node)
        return info

    monkeypatch.setattr(P, "_get_engine_info_for_node", fake_info)
    d0 = part._resolve_target_device_for_partition(
        object(), types.SimpleNamespace(id=0, nodes=["0"])
    )
    d1 = part._resolve_target_device_for_partition(
        object(), types.SimpleNamespace(id=1, nodes=["1"])
    )
    assert d0 == b"cuda:0"
    assert d1 == b"cuda:1"
    assert d0 != d1


# --- lift() preserves constant dtype/device ----------------------------------
# lift() replaces get_attr constants with placeholders and copies a fake meta
# "val". It must fakify the source constant through the graph's own fake_mode
# (fake_mode.from_tensor), preserving dtype, device and stride; otherwise a
# non-fp32 or non-CPU lifted constant silently gets fp32/cpu meta.


def _traced_gm_with_parameter(dtype, device):
    """A symbolically-traced GraphModule with one get_attr parameter (`c`) of the
    given dtype/device, plus a stub graph_signature lift() can mutate."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Parameter(
                torch.zeros(3, 3, dtype=dtype, device=device), requires_grad=False
            )

        def forward(self, x):
            return x + self.c

    gm = torch.fx.symbolic_trace(M())
    # lift() calls detect_fake_mode over the placeholder "val" metas, so the user
    # input needs a FakeTensor meta.
    fake_mode = FakeTensorMode()
    with fake_mode:
        fake_x = torch.empty(3, 3)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = fake_x
    sig = types.SimpleNamespace(input_specs=[], output_specs=[], user_inputs=["x"])
    return gm, sig


def _lifted_constant_meta(gm, sig):
    lifted_gm, _, _, _ = lift(gm, sig)
    lifted = [
        n for n in lifted_gm.graph.nodes if n.op == "placeholder" and n.name != "x"
    ]
    assert len(lifted) == 1, f"expected 1 lifted constant, got {len(lifted)}"
    return lifted[0].meta["val"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, device",
    [
        (torch.bfloat16, "cpu"),
        (torch.float32, "cuda"),
    ],
)
def test_lift_preserves_constant_dtype_device(dtype, device):
    # Runtime gate (not a module-level skipif, which resolves at collection time
    # and is fragile on remote-GPU runners): skip the CUDA case only when no GPU.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gm, sig = _traced_gm_with_parameter(dtype, device)
    val = _lifted_constant_meta(gm, sig)
    assert isinstance(val, FakeTensor)
    assert val.dtype == dtype
    assert val.device.type == device
    # The source constant is a contiguous 3x3, so from_tensor must carry its real
    # (3, 1) stride onto the meta, not the old all-ones synthetic stride.
    assert val.stride() == torch.zeros(3, 3).stride()


# --- save(output_format="executorch") forwards ExecuTorch lowering kwargs -------
# to_edge_transform_and_lower accepts transform_passes / constant_methods /
# compile_config / generate_etrecord; save() should forward each. compile_config
# defaults to _check_ir_validity=False (the TRT engine placeholder graph fails edge
# IR validation) when omitted, but a caller-supplied config is forwarded verbatim.
# generate_etrecord persists a "<base>_etrecord.bin" next to the .pte.


def _patch_executorch_lowering(monkeypatch, captured):
    """Stub the ExecuTorch lowering + TRT-specific pre/post steps in _save_as_executorch
    so the test exercises only kwarg forwarding. Returns nothing; fills `captured`."""
    import executorch.exir as exir
    import torch_tensorrt._compile as tc

    class _FakeETRecord:
        def save(self, path):
            with open(path, "wb") as fh:
                fh.write(b"etrecord")

    class _FakeExec:
        def write_to_file(self, f):
            f.write(b"")

        def get_etrecord(self):
            return _FakeETRecord()

    class _FakeEdge:
        def to_executorch(self, config=None):
            captured["backend_config"] = config
            return _FakeExec()

    def _fake_lower(exp_program, **kw):
        captured.update(kw)
        return _FakeEdge()

    monkeypatch.setattr(exir, "to_edge_transform_and_lower", _fake_lower)
    monkeypatch.setattr(tc, "_count_executorch_engine_nodes", lambda ep: 0)
    monkeypatch.setattr(tc, "_replace_execute_engine_for_executorch", lambda ep: ep)
    monkeypatch.setattr(tc, "_write_external_tensor_data", lambda prog, path: None)
    # ENABLED_FEATURES is an immutable namedtuple; swap the whole module attribute.
    monkeypatch.setattr(
        tc, "ENABLED_FEATURES", types.SimpleNamespace(torch_tensorrt_runtime=True)
    )


@pytest.mark.unit
def test_save_executorch_forwards_lowering_kwargs(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt._compile as tc
    from executorch.exir import EdgeCompileConfig

    captured = {}
    _patch_executorch_lowering(monkeypatch, captured)

    sentinel_passes = [object()]
    sentinel_methods = {"get_max_seq_len": 128}
    caller_cfg = EdgeCompileConfig(_check_ir_validity=True)
    out = str(tmp_path / "model.pte")

    tc._save_as_executorch(
        object(),
        out,
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=sentinel_methods,
        transform_passes=sentinel_passes,
        compile_config=caller_cfg,
        generate_etrecord=True,
    )

    assert captured["transform_passes"] is sentinel_passes
    assert captured["constant_methods"] is sentinel_methods
    assert captured["generate_etrecord"] is True
    # A caller-supplied compile_config is forwarded verbatim (explicit override
    # respected, not overridden even though it sets _check_ir_validity=True).
    assert captured["compile_config"] is caller_cfg
    assert captured["compile_config"]._check_ir_validity is True
    # ETRecord persisted next to the .pte per ET's "<base>_etrecord.bin" convention.
    assert (tmp_path / "model_etrecord.bin").exists()


@pytest.mark.unit
def test_save_executorch_defaults_when_lowering_kwargs_omitted(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt._compile as tc

    captured = {}
    _patch_executorch_lowering(monkeypatch, captured)

    out = str(tmp_path / "model.pte")
    tc._save_as_executorch(object(), out)

    # Falls back to get_edge_compile_config() (also _check_ir_validity=False).
    assert captured["compile_config"]._check_ir_validity is False
    assert captured["transform_passes"] is None
    assert captured["generate_etrecord"] is False
    # No etrecord written when generate_etrecord is falsy.
    assert not (tmp_path / "model_etrecord.bin").exists()


# --- the same lowering kwargs flow through the *public* torch_tensorrt.save() -----
# save() pops the ExecuTorch-only kwargs and forwards them to _save_as_executorch
# from three dispatch branches: an ExportedProgram input, a GraphModule with
# retrace=True (re-exported here), and a GraphModule with retrace=False (routed
# through the dynamo exporter). Each must extract the options and forward them.


class _AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _stub_save_as_executorch(monkeypatch):
    """Capture the (module, file_path, kwargs) that save() forwards to
    _save_as_executorch without running the real lowering."""
    import torch_tensorrt._compile as tc

    calls = []

    def _fake(module, file_path, **kw):
        calls.append({"module": module, "file_path": file_path, "kwargs": kw})

    monkeypatch.setattr(tc, "_save_as_executorch", _fake)
    return calls


def _assert_lowering_kwargs_forwarded(kw, methods, passes, cfg):
    assert kw["constant_methods"] is methods
    assert kw["transform_passes"] is passes
    assert kw["compile_config"] is cfg
    assert kw["generate_etrecord"] is True
    assert kw["partitioners"] == []
    assert kw["compile_specs"] == []
    assert kw["backend_config"] is None


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_exported_program(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "ep.pte")

    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    torch_tensorrt.save(
        ep,
        out,
        output_format="executorch",
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    assert calls[0]["module"] is ep
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_graphmodule_retrace(
    monkeypatch, tmp_path
):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "gm_retrace.pte")

    gm = torch.export.export(_AddOne(), (torch.randn(2, 2),)).module()
    torch_tensorrt.save(
        gm,
        out,
        output_format="executorch",
        retrace=True,
        arg_inputs=(torch.randn(2, 2),),
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_graphmodule_no_retrace(
    monkeypatch, tmp_path
):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    import torch_tensorrt.dynamo._exporter as _exporter
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    # retrace=False routes through the dynamo exporter (TRT-specific graph surgery);
    # stub it so the test isolates save()'s option extraction + forwarding.
    sentinel_ep = object()
    monkeypatch.setattr(_exporter, "export", lambda *a, **k: sentinel_ep)

    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "gm_no_retrace.pte")

    gm = torch.export.export(_AddOne(), (torch.randn(2, 2),)).module()
    torch_tensorrt.save(
        gm,
        out,
        output_format="executorch",
        retrace=False,
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    assert calls[0]["module"] is sentinel_ep
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_save_executorch_real_etrecord_is_inspector_consumable(tmp_path):
    """A real lowering with generate_etrecord=True writes a sidecar that ExecuTorch's
    devtools can parse back into an Inspector-consumable ETRecord."""
    pytest.importorskip("executorch.exir")
    pytest.importorskip("executorch.devtools")
    import torch_tensorrt
    from executorch.devtools.etrecord import parse_etrecord
    from torch_tensorrt._features import ENABLED_FEATURES

    if not ENABLED_FEATURES.torch_tensorrt_runtime:
        pytest.skip("output_format='executorch' requires the torch_tensorrt runtime")

    ep = torch.export.export(_AddOne(), (torch.randn(4, 4),))
    out = str(tmp_path / "tiny.pte")
    torch_tensorrt.save(ep, out, output_format="executorch", generate_etrecord=True)

    etrecord_path = tmp_path / "tiny_etrecord.bin"
    assert etrecord_path.exists()

    record = parse_etrecord(str(etrecord_path))
    assert record is not None
    # The parsed record carries the edge-dialect program the Inspector correlates
    # runtime events against.
    assert getattr(record, "edge_dialect_program", None) is not None
