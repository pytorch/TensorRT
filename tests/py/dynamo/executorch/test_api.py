import ast
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj, lift


@pytest.mark.unit
def test_the_python_loader_uses_the_api_that_backs_device_arenas():
    """Loading must go through the Module API, not the program loader.

    A delegated program is exported with device-tagged memory-planned arenas. Only the Module API
    allocates device memory for those; the program loader behind ``executorch.runtime`` plans every
    arena on the host, so the device copy the exporter inserts is handed a host destination and
    ``cudaMemcpy`` fails with ``invalid argument``. Asserting the call rather than trusting a
    comment, since the two APIs differ by one function name and swapping back would be silent until
    someone ran a delegated model on a GPU.
    """
    source = _RUNTIME_PY.read_text(encoding="utf-8")
    assert "_load_for_executorch_from_buffer" in source, (
        "the runtime no longer loads through the Module API, so device-planned arenas would be "
        "planned on the host and every delegate boundary copy would fail"
    )
    assert (
        "load_program" not in source
    ), "the runtime still references the program loader, which does not back device-tagged arenas"


@pytest.mark.unit
def test_every_lane_that_builds_the_cuda_shims_filters_the_arch_list():
    """Both ExecuTorch lanes compile the shims, so both need the architecture floor.

    The build lane builds the wheel and the reference runner; the test lane builds the runner
    again, whose CMakeLists forces EXECUTORCH_BUILD_CUDA=ON. Both source the channel's build
    environment, so filtering in only one of them moves the cu126 failure from that job into the
    other rather than removing it. Checked by parsing, because the ordering matters: the filter has
    to run before whatever compiles.
    """
    import yaml

    for name in ("executorch-build-linux.yml", "executorch-test-linux.yml"):
        workflow = yaml.safe_load(
            (_REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        )
        scripts = [
            (job.get("with") or {}).get("script", "")
            for job in workflow.get("jobs", {}).values()
        ]
        script = next(
            (s for s in scripts if "verify-executorch-reference-runner" in s), None
        )
        assert script is not None, f"{name} no longer builds the reference runner"

        lines = script.splitlines()
        filtered = [
            i for i, line in enumerate(lines) if "filter-executorch-cuda-arches" in line
        ]
        assert filtered, (
            f"{name} builds the CUDA shims without narrowing TORCH_CUDA_ARCH_LIST, so a channel "
            "asking for an architecture without __dp4a fails to compile"
        )
        for marker in ("verify-executorch-reference-runner", "pip wheel"):
            compiled = [i for i, line in enumerate(lines) if marker in line]
            if compiled:
                assert filtered[0] < compiled[0], (
                    f"{name} filters the architecture list after {marker!r}, so the value in "
                    "effect while compiling is the unfiltered one"
                )


@pytest.mark.unit
def test_the_cuda_arch_filter_drops_only_what_executorch_cannot_compile():
    """The delegate build must not be asked for an architecture without ``__dp4a``.

    ExecuTorch's CUDA shims call that intrinsic, which nvcc does not declare before
    sm_61, so a list carrying 5.0 or 6.0 fails to compile at all. The cu126 build
    environment asks for exactly that while the CUDA 13 ones start higher, which is why
    only cu126 broke. The filter narrows the request rather than pinning a list, so this
    checks both that the unbuildable entries go and that everything else survives.
    """
    script = _REPO_ROOT / ".github/scripts/filter-executorch-cuda-arches.py"

    def run(requested):
        return subprocess.run(
            [sys.executable, str(script), requested],
            capture_output=True,
            text=True,
        )

    # The list the cu126 environment actually exports.
    result = run("5.0;6.0;7.0;7.5;8.0;8.6;9.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "7.0;7.5;8.0;8.6;9.0"

    # A CUDA 13 list is already buildable and must come back untouched, including
    # architectures newer than anything named in the filter.
    unchanged = "7.5;8.0;8.6;9.0;10.0;12.0"
    assert run(unchanged).stdout.strip() == unchanged

    # A "+PTX" suffix sorts by its number and keeps its suffix.
    assert run("5.0;6.0;8.6+PTX").stdout.strip() == "8.6+PTX"

    # PyTorch also accepts named architectures and expands each before building its gencode
    # flags, so a name has to be expanded here too. Pascal is 6.0;6.1+PTX, and passing the name
    # through untouched would carry 6.0 into the build, which is the target this filter exists to
    # drop. The longer keys go first, or "Maxwell+Tegra" leaves a "+Tegra" tail behind.
    assert run("Pascal").stdout.strip() == "6.1+PTX"
    assert run("Ampere").stdout.strip() == "8.0;8.6+PTX"
    assert run("Pascal;7.5").stdout.strip() == "6.1+PTX;7.5"
    for name in ("Maxwell", "Maxwell+Tegra", "Kepler"):
        rejected = run(name)
        assert rejected.returncode != 0, f"{name} has no member at or above sm_61"
        assert "sm_61" in rejected.stderr

    # Space separated is the other spelling torch accepts, and it normalises spaces the same way
    # (torch/utils/cpp_extension.py does _arch_list.replace(" ", ";")). Filtering only on
    # semicolons let a space separated list through untouched, which is silent rather than loud.
    assert run("5.0 6.0 7.5 8.6").stdout.strip() == "7.5;8.6"
    assert run("5.0, 6.0, 7.5").stdout.strip() == "7.5"

    # Nothing buildable must fail loudly. Printing an empty list would leave nvcc to
    # pick its own default and ship a wheel for architectures nobody asked for.
    empty = run("5.0;6.0")
    assert empty.returncode != 0
    assert "sm_61" in empty.stderr

    # An unset list is not this script's problem to invent.
    assert run("").returncode == 0


def _is_importable_module(name: str) -> bool:
    """Whether ``name`` is a module, so reaching through it needs its own import.

    find_spec raises rather than returning None for a dotted name whose parent is a module but
    whose child is an ordinary attribute: ``sys.modules`` gives ModuleNotFoundError, "__path__
    attribute not found on 'sys'". Treating that as "not a module" is the point, since an attribute
    is reachable without a second import.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (AttributeError, ImportError, ValueError):
        return False


@pytest.mark.unit
def test_the_runtime_package_imports_every_submodule_it_reaches_through():
    """Every dotted module the runtime package uses must be imported, not assumed.

    ``import importlib`` does not bind ``importlib.util``: the attribute exists only once something
    else in the process has imported that submodule. The package used ``importlib.util.find_spec``
    on the strength of a bare ``import importlib``, which worked whenever torch was imported first,
    since torch pulls the submodule in, and raised AttributeError when the runtime wheel was
    imported on its own. Compiling the module here is not enough to catch it, because the attribute
    is only read at call time, so match the source against what it imports.
    """
    source = _RUNTIME_INIT_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    parts = alias.name.split(".")
                    for index in range(len(parts)):
                        imported.add(".".join(parts[: index + 1]))
                else:
                    imported.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                imported.add(f"{node.module}.{alias.name}")
                imported.add(alias.asname or alias.name)

    used: set[str] = set()
    for node in ast.walk(tree):
        # Only two-level attribute reads, so importlib.util.find_spec yields importlib.util rather
        # than the function. A deeper chain would report a name that is never importable.
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
        ):
            used.add(f"{node.value.value.id}.{node.value.attr}")

    # Only names that are importable modules matter. sys.modules is an attribute of an imported
    # module, not a submodule, so it is reachable without a second import and is not a finding.
    missing = sorted(
        name
        for name in used
        if name.split(".")[0] in imported
        and name not in imported
        and _is_importable_module(name)
    )
    assert not missing, (
        "the runtime package reaches through a submodule it never imports, so the attribute exists "
        f"only when another import happens to have loaded it first: {missing}. Import it explicitly."
    )


@pytest.mark.unit
def test_lazy_import_error_when_executorch_missing(monkeypatch):
    import torch_tensorrt

    original_module = sys.modules.pop("torch_tensorrt.executorch", None)
    original_attribute = getattr(torch_tensorrt, "executorch", None)
    if hasattr(torch_tensorrt, "executorch"):
        delattr(torch_tensorrt, "executorch")
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
    if original_attribute is not None:
        torch_tensorrt.executorch = original_attribute
    elif hasattr(torch_tensorrt, "executorch"):
        delattr(torch_tensorrt, "executorch")


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
def test_load_executorch_error_when_delegate_missing(monkeypatch):
    from torch_tensorrt import _compile

    monkeypatch.setattr(_compile, "_has_executorch_runtime", lambda: False)

    with pytest.raises(ImportError, match=r"torch-tensorrt-executorch-runtime"):
        _compile.load("model.pte", format="executorch")


@pytest.mark.unit
def test_load_executorch_dispatches_to_delegate(monkeypatch):
    from torch_tensorrt import _compile

    delegate = types.ModuleType("torch_tensorrt_executorch_runtime")
    delegate.__path__ = []
    runtime = types.ModuleType("torch_tensorrt_executorch_runtime.runtime")
    sentinel = object()
    runtime.load = lambda path: (sentinel, path)
    monkeypatch.setitem(sys.modules, delegate.__name__, delegate)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(_compile, "_has_executorch_runtime", lambda: True)

    assert _compile.load("model.pte", format="executorch") == (
        sentinel,
        "model.pte",
    )


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__
    assert "export" in module.__all__
    assert "Program" not in module.__all__
    assert "load" not in module.__all__
    assert "to_executorch" not in module.__all__


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SETUP_PY = _REPO_ROOT / "setup.py"
_RUNTIME_SETUP_PY = _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/setup.py"
_RUNTIME_PY = (
    _REPO_ROOT
    / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/runtime.py"
)
_RUNTIME_INIT_PY = (
    _REPO_ROOT
    / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/__init__.py"
)


@pytest.mark.unit
def test_runtime_implementation_is_owned_by_runtime_package():
    assert not (_REPO_ROOT / "py/torch_tensorrt/executorch/runtime.py").exists()
    assert (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime/runtime.py"
    ).is_file()


@pytest.mark.unit
def test_runtime_extension_has_dependency_wheel_rpaths():
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    assert "BUILD_WITH_INSTALL_RPATH ON" in cmake
    assert "$ORIGIN/../torch/lib" in cmake
    assert "$ORIGIN/../tensorrt_libs" in cmake
    assert "$ORIGIN/../nvidia/cuda_runtime/lib" in cmake
    assert "$ORIGIN/../nvidia/cu12/lib" in cmake
    assert "$ORIGIN/../nvidia/cu13/lib" in cmake
    assert "CUDAToolkit_VERSION_MAJOR EQUAL 12" in cmake
    assert "CUDAToolkit_VERSION_MAJOR EQUAL 13" in cmake
    assert "-Wl,-Bsymbolic" not in cmake
    assert "set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON" in cmake
    assert "set(EXECUTORCH_BUILD_XNNPACK ON" in cmake


@pytest.mark.unit
def test_the_install_script_puts_a_cuda_runtime_on_the_library_path_for_both_majors():
    """Every CUDA row gets a CUDA runtime directory, not just the CUDA 13 ones.

    The two majors ship different layouts: nvidia-cuda-runtime-cu12 installs
    nvidia/cuda_runtime/lib/libcudart.so.12 while the CUDA 13 line installs
    nvidia/cu13/lib/libcudart.so.13. Naming only the cu13 path left every CUDA 12 row with no
    CUDA runtime on the search path, and the ExecuTorch reference runner died at startup with
    "libcudart.so.12: cannot open shared object file" while the package was installed all along.
    The runtime wheel's own RPATH already lists both directories; this keeps the CI install
    script consistent with it.
    """
    script = (_REPO_ROOT / ".github/scripts/install-torch-tensorrt.sh").read_text(
        encoding="utf-8"
    )
    for directory in ("nvidia/cuda_runtime/lib", "nvidia/cu13/lib"):
        assert directory in script, (
            f"{directory} is not on LD_LIBRARY_PATH, so a row of that CUDA major cannot resolve "
            "libcudart at run time"
        )
    # Matched on the major, so a future cu128 or cu134 row is covered without another edit.
    for pattern in ("cu12*)", "cu13*)"):
        assert (
            pattern in script
        ), f"{pattern} is gone, so a CUDA row of that major would fall through to no directory"


@pytest.mark.unit
def test_runtime_extension_does_not_require_an_embeddable_python():
    """Development.Embed must stay optional, or the release build cannot configure.

    ExecuTorch declares its pybind modules SHARED, so CMake requires the
    Python::Python target and suggests asking for Development.Embed. Taking that
    suggestion breaks the build: the release image's CPython ships no libpython, so
    the component cannot be satisfied and the whole find_package fails. The
    component is therefore requested optionally, matching pybind11, and the target
    is stood in for when it is absent.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "REQUIRED COMPONENTS Interpreter Development.Module" in cmake
    assert "if(NOT TARGET Python::Python)" in cmake

    # Every mention of the component in actual code, comments excluded, must be an
    # optional one. A required request is what fails on an image without libpython.
    code = [line for line in cmake.splitlines() if not line.lstrip().startswith("#")]
    embed_lines = [line for line in code if "Development.Embed" in line]
    assert embed_lines, "Development.Embed should be requested, optionally"
    for line in embed_lines:
        assert "OPTIONAL_COMPONENTS" in line, (
            "Development.Embed must stay optional; the release image has no "
            f"libpython: {line.strip()!r}"
        )


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
def test_runtime_wheel_selects_its_tensorrt_requirement_per_cuda_major():
    """The TensorRT requirement must match the CUDA major used to build the wheel."""
    setup_source = _RUNTIME_SETUP_PY.read_text(encoding="utf-8")
    tree = ast.parse(setup_source)
    function = _function_def(tree, "get_tensorrt_requirement")
    fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None))
    namespace: dict = {"torch": fake_torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<setup.py>", "exec"),
        namespace,
    )

    for cuda_version, requirement in (
        ("12.6", "tensorrt-cu12>=11.1.0,<11.2"),
        ("12.9", "tensorrt-cu12>=11.1.0,<11.2"),
        ("13.0", "tensorrt-cu13>=11.1.0,<11.2"),
        ("13.2", "tensorrt-cu13>=11.1.0,<11.2"),
    ):
        fake_torch.version.cuda = cuda_version
        assert (
            namespace["get_tensorrt_requirement"]() == requirement
        ), f"CUDA {cuda_version} resolves the wrong TensorRT requirement"

    # A CUDA-less torch and an unsupported major are refused rather than guessed at.
    for cuda_version in (None, "11.8"):
        fake_torch.version.cuda = cuda_version
        with pytest.raises(RuntimeError):
            namespace["get_tensorrt_requirement"]()


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
        assert len(requirements.elts) == 1
        assert isinstance(requirements.elts[0], ast.Name)
        assert requirements.elts[0].id == "EXECUTORCH_REQUIREMENT"

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
def test_main_wheel_does_not_package_executorch_delegate_headers():
    assert "include/torch_tensorrt/executorch/*.h" not in _SETUP_PY.read_text(
        encoding="utf-8"
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
    """The device comes from the partition's own engine node, read as metadata only.

    DEVICE_IDX is the only slot wanted, and this runs once per partition on every
    export that does not pin ``target_device`` -- the default -- so reading the full
    record would re-serialize each engine to look at one string.
    """
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    engine_node = object()
    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [engine_node])
    info = ["0"] * (DEVICE_IDX + 1)
    info[DEVICE_IDX] = "2"
    calls = []

    def _spy(ep, n, **kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr(P, "_get_engine_info_for_node", _spy)

    partition = types.SimpleNamespace(id=0, nodes=[engine_node])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:2"
    assert calls == [{"metadata_only": True}]


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

    def fake_info(ep, node, **kwargs):
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


def _traced_gm_with_parameter(dtype, device, requires_grad=False):
    """A symbolically-traced GraphModule with one get_attr parameter (`c`) of the
    given dtype/device, plus a stub graph_signature lift() can mutate."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Parameter(
                torch.zeros(3, 3, dtype=dtype, device=device),
                requires_grad=requires_grad,
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


# --- lift() preserves parameter kind and requires_grad -----------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, requires_grad",
    [
        (torch.uint8, False),
        (torch.int8, False),
        (torch.float32, False),
        # A trainable weight, so hardcoding either flag value fails the test.
        (torch.float32, True),
    ],
)
def test_lift_keeps_parameter_for_any_dtype(dtype, requires_grad):
    gm, sig = _traced_gm_with_parameter(dtype, "cpu", requires_grad)
    _, graph_signature, state_dict, constants = lift(gm, sig)

    kinds = [spec.kind for spec in graph_signature.input_specs]
    assert InputKind.PARAMETER in kinds, f"expected a PARAMETER, got {kinds}"

    # The weight belongs in the state dict as a parameter, not in constants, or a
    # caller can no longer load a checkpoint into the exported module.
    assert "c" in state_dict, f"weight missing from state_dict: {list(state_dict)}"
    assert isinstance(state_dict["c"], torch.nn.Parameter)
    assert state_dict["c"].dtype == dtype
    assert "c" not in constants

    assert state_dict["c"].requires_grad is requires_grad


# --- save(output_format="executorch") forwards ExecuTorch lowering kwargs -------
# After #4440, _save_as_executorch no longer runs the lowering itself: it delegates to
# torch_tensorrt.executorch.export() (which owns the TRT graph surgery + lowering and
# defaults compile_config to get_edge_compile_config()) and then calls
# edge_program.to_executorch(config=backend_config). save() must forward each of
# transform_passes / constant_methods / compile_config / generate_etrecord /
# partitioners / compile_specs into export(), let export() apply
# get_edge_compile_config() when compile_config is omitted (_check_ir_validity=False,
# since the TRT execute_engine placeholder graph fails edge IR validation), and route
# backend_config to to_executorch(). generate_etrecord persists a "<base>_etrecord.bin"
# next to the .pte.


def _patch_executorch_lowering(monkeypatch, captured):
    """Spy the seams _save_as_executorch delegates to after #4440, without running a
    real TRT lowering. Records the kwargs forwarded into
    torch_tensorrt.executorch.export() (captured["export_kwargs"]); lets the real
    export() run against an engine-free program while stubbing only the innermost
    ExecuTorch lowering, so the kwargs export() finally hands that lowering --
    including the get_edge_compile_config() it substitutes for an omitted
    compile_config -- land in `captured`; and records backend_config from
    to_executorch(). Fills `captured`; returns nothing."""
    import executorch.exir as exir
    import torch_tensorrt._compile as tc
    import torch_tensorrt.executorch as tte

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

    # Innermost real lowering seam, called inside export(): capture the kwargs it
    # receives, notably compile_config already resolved to its default when omitted.
    def _fake_lower(exp_program, **kw):
        captured.update(kw)
        return _FakeEdge()

    monkeypatch.setattr(exir, "to_edge_transform_and_lower", _fake_lower)

    # Record what _save_as_executorch forwards into export(), then run the real
    # export() so its default-application and graph staging still execute.
    real_export = tte.export

    def _spy_export(source, **kw):
        captured["export_kwargs"] = kw
        return real_export(source, **kw)

    monkeypatch.setattr(tte, "export", _spy_export)
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

    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    tc._save_as_executorch(
        ep,
        out,
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=sentinel_methods,
        transform_passes=sentinel_passes,
        compile_config=caller_cfg,
        generate_etrecord=True,
    )

    # Every ExecuTorch lowering kwarg is forwarded verbatim into export().
    export_kwargs = captured["export_kwargs"]
    assert export_kwargs["transform_passes"] is sentinel_passes
    assert export_kwargs["constant_methods"] is sentinel_methods
    assert export_kwargs["compile_config"] is caller_cfg
    assert export_kwargs["generate_etrecord"] is True
    assert export_kwargs["partitioners"] == []
    assert export_kwargs["compile_specs"] == []
    # A caller-supplied compile_config reaches the lowering verbatim (explicit override
    # respected, not replaced with the default even though it sets
    # _check_ir_validity=True).
    assert captured["compile_config"] is caller_cfg
    assert captured["compile_config"]._check_ir_validity is True
    # backend_config flows to edge_program.to_executorch(config=...).
    assert captured["backend_config"] is None
    # ETRecord persisted next to the .pte per ET's "<base>_etrecord.bin" convention.
    assert (tmp_path / "model_etrecord.bin").exists()


@pytest.mark.unit
def test_save_executorch_defaults_when_lowering_kwargs_omitted(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt._compile as tc

    captured = {}
    _patch_executorch_lowering(monkeypatch, captured)

    out = str(tmp_path / "model.pte")
    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    tc._save_as_executorch(ep, out)

    # _save_as_executorch forwards None for every omitted lowering kwarg, delegating
    # the defaults to export().
    export_kwargs = captured["export_kwargs"]
    assert export_kwargs["compile_config"] is None
    assert export_kwargs["transform_passes"] is None
    assert export_kwargs["constant_methods"] is None
    assert export_kwargs["partitioners"] is None
    assert export_kwargs["compile_specs"] is None
    assert export_kwargs["generate_etrecord"] is False
    # export() substitutes get_edge_compile_config() for the omitted compile_config
    # (_check_ir_validity=False) and that config reaches the lowering.
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
    # stub it so the test isolates save()'s option extraction + forwarding. The stub
    # returns a real ExportedProgram because save() runs
    # _declare_aliased_kv_mutations_on_ep over the exporter's result before forwarding
    # it, and that pass reads .graph_module / .graph_signature. This program has no
    # engine nodes, so the pass returns it unchanged and the identity assertion below
    # still pins exactly what the exporter produced.
    stub_ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    monkeypatch.setattr(_exporter, "export", lambda *a, **k: stub_ep)

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
    assert calls[0]["module"] is stub_ep
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
