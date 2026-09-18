# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the real loader and compatibility wrapper with a controlled native boundary."""

import ast
import importlib.util
import inspect
import os
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
    # The replacement it points at has to be the public runtime API, not the underscore-prefixed
    # loader in pybindings, because that one is private and can change without notice.
    assert "executorch.runtime" in str(warning.message)
    assert "load_program" in str(warning.message)
    assert "_load_for_executorch" not in str(warning.message)


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


@pytest.mark.unit
def test_the_published_main_wheel_can_still_reach_the_loader_it_imports() -> None:
    """The released main wheel does ``from ...runtime import load``, by name.

    Upgrading this package on its own must not break that call, so the submodule and the name both
    have to survive as long as a released main wheel reaches for them. Asserting the import path the
    way the published wheel writes it is what makes a deletion visible here rather than in a user's
    traceback.
    """
    package = (
        Path(__file__).resolve().parents[4]
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime"
    )
    module_path = package / "runtime.py"
    assert module_path.exists(), "the published main wheel imports this submodule"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    exported = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "load" in exported, sorted(exported)
    # Its one argument is the path, which is how the main wheel calls it.
    load = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load"
    )
    assert [a.arg for a in load.args.args] == ["file_path"], [
        a.arg for a in load.args.args
    ]
    # Parsing alone cannot see a module that raises while being imported, and the published wheel
    # imports this one, so run its top level. Everything it needs at that point is standard library;
    # the import of the main wheel's loader sits inside the function and is not reached here.
    namespace: dict[str, object] = {
        "__name__": "torch_tensorrt_executorch_runtime.runtime"
    }
    exec(compile(tree, str(module_path), "exec"), namespace)
    assert callable(namespace.get("load")), sorted(namespace)


@pytest.mark.unit
def test_an_old_companion_gets_advice_it_can_act_on(
    compiler, monkeypatch, tmp_path
) -> None:
    """An older companion refuses once ExecuTorch's own bindings are loaded, and says to import it
    earlier. A caller cannot do that: the colliding import happens inside this library. So the
    message has to name the thing that does work, which is upgrading the companion."""
    module = types.ModuleType("torch_tensorrt_executorch_runtime")

    def activate():
        raise ImportError("import torch_tensorrt_executorch_runtime before executorch")

    module.activate = activate
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", module)
    program = tmp_path / "m.pte"
    program.write_bytes(b"unused")
    with pytest.raises(ImportError, match="too old to register"):
        compiler.load(str(program), format="executorch")


@pytest.mark.unit
def test_the_loader_calls_activate_on_a_companion_that_has_no_register(
    compiler, monkeypatch, tmp_path
) -> None:
    """Upgrading the main wheel alone leaves an older companion installed, and that one exposes
    activate() rather than register(). Checking the source for the word proves nothing: the
    fallback can be commented out and the word stays. So drive it and see which one is called.
    """
    called = []
    module = types.ModuleType("torch_tensorrt_executorch_runtime")
    module.activate = lambda: called.append("activate")
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", module)
    program = tmp_path / "m.pte"
    program.write_bytes(b"unused")
    # The load itself cannot finish without a real runtime; reaching it is the point.
    with pytest.raises(Exception):
        compiler.load(str(program), format="executorch")
    assert called == ["activate"], called


@pytest.mark.unit
def test_the_forwarder_returns_the_shape_the_published_api_returned() -> None:
    """Restoring the file was not enough; it has to return what callers already use.

    The API this replaces returned an object carrying run() and forward(), and raised
    FileNotFoundError for a missing path. Forwarding to ExecuTorch's own loader returns neither, so a
    caller of the published API would fail on the return value rather than on the import, which is
    the same breakage one step later.
    """
    source = (
        ROOT
        / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    load = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load"
    )
    returns = [
        ast.unparse(node.value)
        for node in ast.walk(load)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert returns, "the forwarder returns nothing"
    assert any(
        "_load" in returned for returned in returns
    ), f"the forwarder does not delegate to the compatibility loader: {returns}"
    assert (
        "_load_for_executorch" not in source
    ), "forwarding to ExecuTorch's loader returns the wrong object"
    # The loader it forwards to is the one carrying the original interface.
    compat = (ROOT / "py/torch_tensorrt/_executorch_compat.py").read_text(
        encoding="utf-8"
    )
    for member in ("def run(", "def forward(", "FileNotFoundError"):
        assert member in compat, f"the compatibility loader lost {member}"


@pytest.mark.parametrize(
    "search_path,expected",
    [
        (None, "loader could not find it, not that it is incompatible"),
        ("/opt/cuda/lib64:", "empty entry"),
        (":", "empty entry"),
        ("/opt/cuda/lib64", "loader could not find it, not that it is incompatible"),
    ],
)
@pytest.mark.unit
def test_a_library_the_loader_cannot_find_is_not_reported_as_an_abi_mismatch(
    search_path, expected
) -> None:
    """A library the loader could not find is a different problem from one it could not use.

    Only the second is an ABI mismatch. A correct installation failed to import from some working
    directories and not others, because an empty entry in the search path is read as the working
    directory and stops this package's own origin-relative entries resolving. Blaming the ABI sent
    the reader to rebuild a stack that already matched.
    """
    source = (
        ROOT
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime/__init__.py"
    ).read_text(encoding="utf-8")
    assert "cannot open shared object file" in source, source[:200]
    assert "LD_LIBRARY_PATH" in source, "the empty entry trap is not mentioned"
    # The classification the source performs, applied to the text the loader really produces.
    text = (
        "libexecutorch_extension_cuda.so: cannot open shared object file: No such file"
    )
    missing = "cannot open shared object file" in text
    empty = search_path is not None and "" in search_path.split(os.pathsep)
    assert missing, "this case is meant to be a not-found error"
    assert empty == (expected == "empty entry"), (search_path, empty)


@pytest.mark.unit
def test_the_forwarder_says_which_side_is_too_old() -> None:
    """The loader this forwards to belongs to the main wheel and is new in this change.

    A main wheel old enough to import this submodule by name does not carry it, so the forward would
    have raised a bare missing-module error naming something the reader never asked for. That pairing
    should not arise, because this package requires the main wheel of its own build exactly, but an
    install that skipped dependency resolution can produce it.
    """
    source = (
        ROOT
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime/runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    load = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load"
    )
    guarded = [
        node
        for node in ast.walk(load)
        if isinstance(node, ast.Try)
        and any(
            isinstance(h.type, ast.Name) and h.type.id == "ImportError"
            for h in node.handlers
        )
    ]
    assert guarded, "the forward into the main wheel is not guarded"
    assert "older than the one this package was built against" in source, source[-600:]


@pytest.mark.parametrize(
    "api,expect_rewrite", [("activate", True), ("register", False)]
)
@pytest.mark.unit
def test_the_age_rewrite_follows_the_api_not_the_error_name(
    compiler, boundary, monkeypatch, api, expect_rewrite
):
    """Both companions define a class of the same name, so the name cannot tell them apart.

    The rewrite exists for a companion published before registration became a single call. Deciding
    that from the raised error's class name matched the current companion too, so the rewrite never
    fired for the one it was written for. Which registration function the companion exposes is the
    thing that actually differs.
    """

    class DelegateCompatibilityError(ImportError):
        pass

    def refuse():
        raise DelegateCompatibilityError("already loaded, import this package earlier")

    module = types.ModuleType("torch_tensorrt_executorch_runtime")
    setattr(module, api, refuse)
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", module)
    with pytest.raises(ImportError) as raised:
        load_legacy(compiler, boundary.path)
    rewritten = "too old to register" in str(raised.value)
    assert rewritten is expect_rewrite, str(raised.value)


@pytest.mark.unit
def test_the_forwarder_actually_forwards_and_warns(monkeypatch) -> None:
    """Reading the file cannot see a forwarder that forwards nowhere.

    Four separate ways of breaking it left the suite green: dropping the import, dropping the
    deprecation warning, naming the loader without calling it, and returning nothing at all. So it is
    called here, against a stub standing in for the main wheel's loader.

    Called three ways, because the published signature is part of what this module preserves: a
    positional string, the same path by keyword, and a Path object. Renaming the parameter or
    narrowing it to str keeps the positional call working and breaks the other two.
    """
    sentinel = object()
    calls: list[str] = []
    compat = types.ModuleType("torch_tensorrt._executorch_compat")
    compat.load = lambda path: calls.append(path) or sentinel
    parent = types.ModuleType("torch_tensorrt")
    parent._executorch_compat = compat
    monkeypatch.setitem(sys.modules, "torch_tensorrt", parent)
    monkeypatch.setitem(sys.modules, "torch_tensorrt._executorch_compat", compat)
    namespace: dict[str, object] = {
        "__name__": "torch_tensorrt_executorch_runtime.runtime"
    }
    source = (
        ROOT
        / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/runtime.py"
    ).read_text(encoding="utf-8")
    exec(compile(source, "runtime.py", "exec"), namespace)
    with pytest.warns(DeprecationWarning):
        returned = namespace["load"]("some/model.pte")
    assert returned is sentinel, returned
    with pytest.warns(DeprecationWarning):
        assert namespace["load"](path="some/model.pte") is sentinel
    with pytest.warns(DeprecationWarning):
        assert namespace["load"](Path("some/model.pte")) is sentinel
    assert calls == ["some/model.pte", "some/model.pte", Path("some/model.pte")], calls
    signature = inspect.signature(namespace["load"])
    assert list(signature.parameters) == ["path"], signature


@pytest.mark.unit
def test_a_missing_file_is_reported_before_a_missing_delegate(
    compiler, monkeypatch, tmp_path
):
    """Both wrong at once used to report the install, which is true but not what the caller got wrong.

    Someone who mistyped a file name and happens not to have the delegate installed should hear about
    the file name.
    """
    monkeypatch.delitem(sys.modules, "torch_tensorrt_executorch_runtime", raising=False)

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "torch_tensorrt_executorch_runtime":
                raise ModuleNotFoundError(
                    "No module named 'torch_tensorrt_executorch_runtime'",
                    name="torch_tensorrt_executorch_runtime",
                )
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    with pytest.raises(FileNotFoundError, match="not found"):
        load_legacy(compiler, tmp_path / "typo.pte")
