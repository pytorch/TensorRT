"""Exercise the real loader and compatibility wrapper with a controlled native boundary."""

import importlib.util
import inspect
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[4]
PORTABLE = "executorch.extension.pybindings.portable_lib"
DELEGATE = "torch_tensorrt_executorch_runtime"


@pytest.fixture
def compiler(monkeypatch):
    # Isolate the full loader module from compiler frontends that require a GPU.
    for name in list(sys.modules):
        if name == "torch_tensorrt" or name.startswith("torch_tensorrt."):
            monkeypatch.delitem(sys.modules, name)
    modules = {
        "torch_tensorrt": {},
        "torch_tensorrt._enums": {"dtype": object},
        "torch_tensorrt._features": {
            "ENABLED_FEATURES": types.SimpleNamespace(
                fx_frontend=False,
                torchscript_frontend=False,
                dynamo_frontend=False,
                torch_tensorrt_runtime=True,
            ),
            "needs_cross_compile": lambda fn: fn,
        },
        "torch_tensorrt._Input": {"Input": type("Input", (), {})},
        "torch_tensorrt._utils": {"executorch_install_command": lambda: "unused"},
        "torch_tensorrt.dynamo": {},
        "torch_tensorrt.dynamo.runtime": {},
        "torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule": {
            "CudaGraphsTorchTensorRTModule": type(
                "CudaGraphsTorchTensorRTModule", (), {}
            )
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        module.__path__ = [str(ROOT / "py" / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, module)
    # Track these entries before Python imports them so monkeypatch restores them too.
    for name in ("torch_tensorrt._compile", "torch_tensorrt._executorch_compat"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location(
        "torch_tensorrt._compile", ROOT / "py/torch_tensorrt/_compile.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    events = []
    state = types.SimpleNamespace(events=events, output=None)

    class NativeModule:
        def method_names(self):
            return ["forward", "add", "constant", "empty"]

        def run_method(self, name, inputs):
            events.append((name, inputs))
            if name == "forward":
                state.output = [inputs[0] + 1]
            elif name == "add":
                state.output = [inputs[0] + inputs[1], inputs[0]]
            elif name == "constant":
                state.output = [17, None, "constant"]
            elif name == "empty":
                state.output = []
            else:
                raise AssertionError("Unknown method reached the native runtime")
            return state.output

    def register():
        events.append("register")

    def load(data):
        assert events[-1] == "register"
        assert data == b"controlled native program"
        state.data_id = id(data)
        events.append("load")
        return state.native

    state.native = NativeModule()
    for name in (
        "executorch",
        "executorch.extension",
        "executorch.extension.pybindings",
        PORTABLE,
        DELEGATE,
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    state.portable = sys.modules[PORTABLE]
    state.portable._load_for_executorch_from_buffer = load
    state.delegate = sys.modules[DELEGATE]
    state.delegate.register = register
    # The host Program loader and removed private companion runtime must not be used.
    monkeypatch.setitem(sys.modules, "executorch.runtime", None)
    monkeypatch.setitem(sys.modules, DELEGATE + ".runtime", None)
    state.path = tmp_path / "model.pte"
    state.path.write_bytes(b"controlled native program")
    return state


def load_legacy(compiler, path, **kwargs):
    with pytest.warns(DeprecationWarning, match="format='executorch'"):
        return compiler.load(path, format="executorch", **kwargs)


@pytest.mark.parametrize("as_string", [False, True])
def test_released_program_interface(compiler, boundary, as_string):
    path = str(boundary.path) if as_string else boundary.path
    program = load_legacy(compiler, path)
    assert not callable(program)
    assert callable(program.forward) and callable(program.run)
    assert program.method_names == ["forward", "add", "constant", "empty"]
    assert not callable(program.method_names)
    assert boundary.events == ["register", "load"]
    assert id(program._data) == boundary.data_id
    assert sys.modules[PORTABLE] is boundary.portable
    assert sys.modules[DELEGATE] is boundary.delegate
    x = torch.tensor([2.0, 4.0])
    for result in (program.forward(x), program.run([x])):
        torch.testing.assert_close(result[0], x + 1)
    result = program.run((x, 3), method="add")
    assert result is boundary.output
    torch.testing.assert_close(result[0], x + 3)
    assert result[1] is x
    assert program.run([], "constant") == [17, None, "constant"]
    assert program.run([], "empty") == []
    with pytest.raises(TypeError):
        program(x)
    with pytest.raises(TypeError):
        program.forward(x=x)


def test_warning_identifies_caller_and_deprecation_period(compiler, boundary):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        compiler.load(boundary.path, format="executorch")
    assert len(caught) == 1
    warning = caught[0]
    assert warning.category is DeprecationWarning
    assert warning.filename == __file__ and warning.lineno == line
    assert "six months" in str(warning.message)
    assert "_load_for_executorch" in str(warning.message)


def test_legacy_options_remain_ignored(compiler, boundary):
    extra_files = {"metadata": "unchanged"}
    load_legacy(
        compiler,
        boundary.path,
        extra_files=extra_files,
        data_path="not-forwarded.ptd",
        enable_etdump=True,
        map_location="cuda:0",
        unknown_option=object(),
    )
    assert boundary.events == ["register", "load"]
    assert extra_files == {"metadata": "unchanged"}


def test_cuda_normalization_preserves_other_inputs(compiler, boundary, monkeypatch):
    """The CUDA tensor is simulated on CPU; no GPU execution is claimed."""
    cpu = torch.tensor([3.0])
    copies = []

    class SimulatedCudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        def cpu(self):
            copies.append(self)
            return cpu

    class NonTensor:
        is_cuda = True

        def cpu(self):
            raise AssertionError("Only CUDA torch.Tensor inputs are copied")

    cuda = cpu.as_subclass(SimulatedCudaTensor)
    other = NonTensor()
    inputs = [cuda, cpu, other, [cuda], None]
    received = []

    def run_method(name, values):
        received.append((name, values))
        return values

    monkeypatch.setattr(boundary.native, "run_method", run_method)
    program = load_legacy(compiler, boundary.path)
    result = program.forward(*inputs)
    assert len(copies) == 1 and copies[0] is cuda
    assert isinstance(received[0][1], tuple)
    assert received[0][0] == "forward"
    assert result[0] is cpu and result[1] is cpu
    assert result[2] is other and result[3] is inputs[3] and result[4] is None
    assert inputs[0] is cuda


def test_unknown_method_fails_before_native_dispatch(compiler, boundary):
    program = load_legacy(compiler, boundary.path)
    with pytest.raises(ValueError) as raised:
        program.run([], "missing")
    assert str(raised.value) == (
        "Unknown method 'missing'; available methods: ['add', 'constant', 'empty', 'forward']"
    )
    assert boundary.events == ["register", "load"]


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_missing_model_has_released_error(compiler, boundary, kind):
    path = (
        boundary.path.parent
        if kind == "directory"
        else boundary.path.parent / "missing.pte"
    )
    with pytest.raises(FileNotFoundError, match="ExecuTorch model not found"):
        load_legacy(compiler, path)
    assert "load" not in boundary.events


@pytest.mark.parametrize("stage", ["register", "load", "run"])
def test_native_failures_propagate_without_fallback(
    compiler, boundary, monkeypatch, stage
):
    failure = RuntimeError("controlled native failure")

    def fail(*args, **kwargs):
        raise failure

    def unexpected(*args, **kwargs):
        pytest.fail("ExecuTorch errors must not reach another format's loader")

    monkeypatch.setattr(torch.export, "load", unexpected)
    monkeypatch.setattr(torch.jit, "load", unexpected)
    target, attribute = {
        "register": (boundary.delegate, "register"),
        "load": (boundary.portable, "_load_for_executorch_from_buffer"),
        "run": (boundary.native, "run_method"),
    }[stage]
    monkeypatch.setattr(target, attribute, fail)
    with pytest.raises(RuntimeError) as raised:
        program = load_legacy(compiler, boundary.path)
        program.forward(torch.tensor(1))
    assert raised.value is failure


def test_deferred_module_validation_is_preserved(compiler, boundary, monkeypatch):
    failure = RuntimeError("Failed to get method names: invalid program")

    def fail():
        raise failure

    monkeypatch.setattr(boundary.native, "method_names", fail)
    program = load_legacy(compiler, boundary.path)
    with pytest.raises(RuntimeError) as raised:
        program.forward(torch.tensor(1))
    assert raised.value is failure
    assert boundary.events == ["register", "load"]


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_broken_delegate_import_keeps_native_diagnostic(
    compiler, boundary, monkeypatch, error_type
):
    import builtins

    original = builtins.__import__
    failure = error_type("libexecutorch.so: undefined symbol")

    def import_module(name, *args, **kwargs):
        if name == DELEGATE:
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(error_type) as raised:
        load_legacy(compiler, boundary.path)
    assert raised.value is failure
    assert boundary.events == []


def test_standard_path_registers_python_engine_ops(compiler, monkeypatch):
    import builtins

    original = builtins.__import__
    events = []
    name = "torch_tensorrt.dynamo.runtime._TRTEngine"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(compiler.ENABLED_FEATURES, "torch_tensorrt_runtime", False)

    def import_module(module_name, *args, **kwargs):
        if module_name == name:
            events.append("register")
        return original(module_name, *args, **kwargs)

    def load(path, extra_files=None):
        events.append("load")
        return "exported program"

    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(torch.export, "load", load)
    assert compiler.load("standard.pt2", format=None) == "exported program"
    assert events == ["register", "load"]


@pytest.mark.parametrize("missing", [DELEGATE, "transitive_dependency"])
def test_missing_dependency_is_not_misdiagnosed(
    compiler, boundary, monkeypatch, missing
):
    import builtins

    original = builtins.__import__
    error = ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    def import_module(name, *args, **kwargs):
        if name == DELEGATE:
            raise error
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ImportError) as raised:
        load_legacy(compiler, boundary.path)
    if missing == DELEGATE:
        assert "torch_tensorrt_executorch_runtime" in str(raised.value)
        assert raised.value.__cause__ is error
    else:
        assert raised.value is error
    assert boundary.events == []


@pytest.mark.parametrize("format", ["torchscript", "exported_program", "", False, 1])
def test_unsupported_format_preserves_value_error(compiler, boundary, format):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Unsupported format"):
            compiler.load(boundary.path, format=format)
    assert not caught
    assert boundary.events == []


@pytest.mark.parametrize("options", [{}, {"format": None}])
@pytest.mark.parametrize("fallback", [False, True])
def test_standard_dispatch_and_kwargs(
    compiler, boundary, monkeypatch, caplog, options, fallback
):
    extra_files = {"metadata": ""}
    calls = []
    expected = object()

    def export_load(path, extra_files):
        calls.append(("export", path, extra_files))
        if fallback:
            raise RuntimeError("not an exported program")
        extra_files["metadata"] = "export data"
        return expected

    def jit_load(path, map_location=None, _extra_files=None):
        calls.append(("jit", path, map_location, _extra_files))
        _extra_files["metadata"] = "jit data"
        return expected

    monkeypatch.setattr(torch.export, "load", export_load)
    monkeypatch.setattr(torch.jit, "load", jit_load)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compiler.load(
            boundary.path,
            extra_files,
            map_location="cpu",
            unknown_option=True,
            **options,
        )
    assert result is expected
    assert calls[0] == ("export", boundary.path, extra_files)
    assert len(calls) == (2 if fallback else 1)
    if fallback:
        assert calls[1] == ("jit", boundary.path, "cpu", extra_files)
    assert extra_files["metadata"] == ("jit data" if fallback else "export data")
    assert "Keyword argument unknown_option" in caplog.text
    assert not caught
    assert boundary.events == []


@pytest.mark.parametrize("options", [{}, {"format": None}])
@pytest.mark.parametrize("format", ["export", "jit"])
def test_standard_formats_round_trip_real_cpu_torch(
    compiler, boundary, options, format
):
    model = torch.nn.Linear(2, 2).eval()
    x = torch.tensor([[1.0, 2.0]])
    path = str(boundary.path.parent / "standard.pt2")
    extra_files = {"metadata": ""}
    if format == "export":
        torch.export.save(
            torch.export.export(model, (x,)), path, extra_files={"metadata": "kept"}
        )
    else:
        torch.jit.save(
            torch.jit.trace(model, (x,)), path, _extra_files={"metadata": "kept"}
        )
    loaded = compiler.load(path, extra_files=extra_files, **options)
    actual = loaded.module()(x) if format == "export" else loaded(x)
    torch.testing.assert_close(actual, model(x))
    assert extra_files["metadata"] == ("kept" if format == "export" else b"kept")
    assert boundary.events == []


def test_standard_failure_preserves_value_error(compiler, boundary):
    with pytest.raises(ValueError, match="valid Torchscript module or ExportedProgram"):
        compiler.load(str(boundary.path), format=None)
    assert boundary.events == []


def test_format_remains_keyword_only(compiler, boundary):
    signature = inspect.signature(compiler.load)
    assert signature.parameters["format"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["format"].default is None
    with pytest.raises(TypeError):
        compiler.load(boundary.path, None, "executorch")
    assert boundary.events == []
