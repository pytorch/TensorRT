# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import ctypes
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

DELEGATE_PATH = (
    Path(__file__).parents[4]
    / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/__init__.py"
)
SETUP_PATH = Path(__file__).parents[4] / "py/torch-tensorrt-executorch-runtime/setup.py"
SKIP_ENV = "TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION"
COMPANION_ROOT = Path(__file__).parents[4] / "py/torch-tensorrt-executorch-runtime"
PACKAGE = "torch_tensorrt_executorch_runtime"


@pytest.mark.parametrize("device_resident", [False, True])
@pytest.mark.parametrize("has_forward", [False, True])
def test_examples_use_the_module_loader(
    monkeypatch, tmp_path, device_resident, has_forward
):
    """Execute the examples without CUDA and verify the Module API receives the original inputs."""
    import runpy

    calls = []
    expected = object()

    class Tensor:
        is_cuda = device_resident
        device = "cuda:0" if device_resident else "cpu"
        shape = (64, 64) if device_resident else (2, 3, 4, 4)

        def cpu(self):
            return self

        def __add__(self, value):
            assert value == 1
            return expected

    tensor = Tensor()
    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    def ones(shape, dtype, device="cpu"):
        assert tuple(shape) == tensor.shape
        assert (device == "cuda") is device_resident
        return tensor

    torch.ones = ones
    torch.tanh = torch.erfinv = lambda value: value
    torch.cos = lambda value: expected
    torch.testing = types.SimpleNamespace(
        assert_close=lambda actual, wanted: calls.append((actual, wanted))
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    _fake_executorch(monkeypatch, set())
    model = tmp_path / "model.pte"
    methods = ["forward"] if has_forward else []

    def execute(inputs):
        assert inputs == (tensor,)
        calls.append("run")
        return [tensor]

    def load_method(name):
        assert name == "forward"
        return types.SimpleNamespace(execute=execute)

    def load_program(path):
        # The examples hand a Path, not a string, because that is what the public loader takes.
        assert str(path) == str(model)
        calls.append("load")
        return types.SimpleNamespace(method_names=methods, load_method=load_method)

    runtime_mod = types.ModuleType("executorch.runtime")
    runtime_mod.Runtime = types.SimpleNamespace(
        get=lambda: types.SimpleNamespace(load_program=load_program)
    )
    monkeypatch.setitem(sys.modules, "executorch.runtime", runtime_mod)
    # Record the import rather than pre-inserting a module. A module already in sys.modules makes
    # "import x" a no-op with nothing to observe, so deleting that import from the example left every
    # case green even though the delegate would never register.
    imported: list[str] = []

    class _Loader:
        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            return None

    class _Recorder:
        def find_spec(self, name, path=None, target=None):
            if name == "torch_tensorrt_executorch_runtime":
                imported.append(name)
                return importlib.util.spec_from_loader(name, loader=_Loader())
            return None

    monkeypatch.delitem(sys.modules, "torch_tensorrt_executorch_runtime", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Recorder(), *sys.meta_path])
    filename = "load_model_device_resident.py" if device_resident else "load_model.py"
    source = (
        Path(__file__).parents[4] / "examples/executorch_reference_runner" / filename
    )
    monkeypatch.setattr(
        sys, "argv", [str(source), "--model_path", str(model), "--num_runs", "2"]
    )
    if not has_forward:
        with pytest.raises(RuntimeError, match="has no 'forward' method"):
            runpy.run_path(str(source), run_name="__main__")
        assert calls == ["load"]
    else:
        runpy.run_path(str(source), run_name="__main__")
        assert calls == ["load", "run", "run", (tensor, expected)]
    # The example has to import the delegate package, because that import is what registers the
    # backend. Nothing here can run without it in a real process.
    assert imported == ["torch_tensorrt_executorch_runtime"], imported


def load_delegate_module(*, register_on_import: bool = False):
    """Import the delegate module from source, side effect suppressed by default.

    Importing the real package registers the backend, which is the whole contract. Every test below
    that drives a failure branch has to install its fakes BEFORE anything loads, so it needs the
    module without that side effect; the opt-out the package documents is exactly the hook for it.
    ``register_on_import=True`` is for the two tests that assert the side effect itself.
    """
    previous = os.environ.get(SKIP_ENV)
    if register_on_import:
        os.environ.pop(SKIP_ENV, None)
    else:
        os.environ[SKIP_ENV] = "1"
    try:
        spec = importlib.util.spec_from_file_location(
            "torchtrt_et_delegate_test", DELEGATE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop(SKIP_ENV, None)
        else:
            os.environ[SKIP_ENV] = previous


def _fake_executorch(monkeypatch, registered):
    """Stand in for the installed ExecuTorch, whose registry the delegate registers into.

    ``registered`` is the live set the fake ``CDLL`` mutates, which is how these tests model
    the one thing that actually matters: the backend appears only as a side effect of loading
    the library.
    """
    portable_lib = types.ModuleType("executorch.extension.pybindings.portable_lib")
    portable_lib._get_registered_backend_names = lambda: sorted(registered)
    pybindings = types.ModuleType("executorch.extension.pybindings")
    pybindings.portable_lib = portable_lib
    extension = types.ModuleType("executorch.extension")
    extension.pybindings = pybindings
    executorch = types.ModuleType("executorch")
    executorch.extension = extension
    for name, module in {
        "executorch": executorch,
        "executorch.extension": extension,
        "executorch.extension.pybindings": pybindings,
        "executorch.extension.pybindings.portable_lib": portable_lib,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _delegate_handle(owns_registration=True):
    return types.SimpleNamespace(
        torch_tensorrt_owns_executorch_registration=lambda: owns_registration
    )


@pytest.mark.unit
def test_the_missing_registry_query_names_the_build_to_install(monkeypatch):
    """An ExecuTorch that does not export the private name must not surface as a bare ImportError.

    Nothing covered the conversion, so deleting the guidance left every test green.
    """
    delegate = load_delegate_module()
    _fake_executorch(monkeypatch, set())
    monkeypatch.delattr(
        sys.modules["executorch.extension.pybindings.portable_lib"],
        "_get_registered_backend_names",
    )
    with pytest.raises(
        delegate.DelegateCompatibilityError, match="build this package pins"
    ):
        delegate._registered_backend_names()


@pytest.mark.unit
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("owns_registration", [False, True])
def test_register_checks_the_loaded_handles_ownership(
    monkeypatch, preloaded, owns_registration
):
    delegate = load_delegate_module()
    registered = {delegate.BACKEND_NAME} if preloaded else set()
    _fake_executorch(monkeypatch, registered)
    handle = _delegate_handle(owns_registration)
    loads = []

    def load(path, mode):
        loads.append(mode)
        registered.add(delegate.BACKEND_NAME)
        return handle

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", load)
    if owns_registration:
        delegate.register()
        delegate.register()
        assert delegate._delegate is handle
        assert len(loads) == 1
        query = handle.torch_tensorrt_owns_executorch_registration
        assert query.argtypes == []
        assert query.restype is ctypes.c_bool
    else:
        for _ in range(2):
            with pytest.raises(
                delegate.DelegateCompatibilityError, match="does not own"
            ):
                delegate.register()
            assert delegate._delegate is None
        assert len(loads) == 2


@pytest.mark.unit
@pytest.mark.parametrize("preloaded", [False, True])
def test_ownership_regression_rejects_presence_only_acceptance(
    monkeypatch, tmp_path, preloaded
):
    tree = ast.parse(DELEGATE_PATH.read_text())
    checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Call)
        and isinstance(node.test.operand.func, ast.Name)
        and node.test.operand.func.id == "query_ownership"
    ]
    assert len(checks) == 1
    checks[0].test = ast.parse(
        "BACKEND_NAME not in _registered_backend_names()", mode="eval"
    ).body
    source = tmp_path / "presence_only.py"
    source.write_text(ast.unparse(tree))
    monkeypatch.setitem(globals(), "DELEGATE_PATH", source)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_register_checks_the_loaded_handles_ownership(monkeypatch, preloaded, False)


@pytest.mark.unit
@pytest.mark.parametrize("preloaded", [False, True])
def test_register_rejects_a_library_without_an_ownership_query(monkeypatch, preloaded):
    delegate = load_delegate_module()
    registered = {delegate.BACKEND_NAME} if preloaded else set()
    _fake_executorch(monkeypatch, registered)

    def load(path, mode):
        registered.add(delegate.BACKEND_NAME)
        return types.SimpleNamespace()

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", load)
    with pytest.raises(delegate.DelegateCompatibilityError, match="ownership query"):
        delegate.register()
    assert delegate._delegate is None


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["missing_noload", "unloaded"])
def test_register_rejects_a_foreign_registration(monkeypatch, failure):
    delegate = load_delegate_module()
    _fake_executorch(monkeypatch, {delegate.BACKEND_NAME})
    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    if failure == "missing_noload":
        monkeypatch.delattr(delegate.os, "RTLD_NOLOAD", raising=False)

    def load(path, mode):
        raise OSError("not loaded")

    monkeypatch.setattr(delegate.ctypes, "CDLL", load)
    with pytest.raises(delegate.DelegateCompatibilityError, match="already registered"):
        delegate.register()
    assert delegate._delegate is None


@pytest.mark.unit
def test_register_retries_after_load_failure(monkeypatch):
    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    handle = _delegate_handle()
    attempts = 0

    def load(path, mode):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("dependency temporarily unavailable")
        registered.add(delegate.BACKEND_NAME)
        return handle

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", load)
    with pytest.raises(delegate.DelegateCompatibilityError):
        delegate.register()
    assert delegate._delegate is None
    delegate.register()
    assert delegate._delegate is handle
    assert attempts == 2


@pytest.mark.unit
def test_concurrent_registration_loads_and_checks_once(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    start = Barrier(8)
    loads = []
    queries = []

    def owns():
        queries.append(True)
        return True

    handle = types.SimpleNamespace(torch_tensorrt_owns_executorch_registration=owns)

    def load(path, mode):
        loads.append(mode)
        registered.add(delegate.BACKEND_NAME)
        time.sleep(0.02)
        return handle

    def register(_):
        start.wait(timeout=5)
        delegate.register()

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", load)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(register, range(8)))
    assert len(loads) == len(queries) == 1
    assert delegate._delegate is handle


def test_importing_the_package_registers_the_backend(monkeypatch):
    """The whole contract of this wheel: the import is the registration.

    ExecuTorch's own delegates register because they are linked into its pybindings extension, so
    loading that extension pulls them in. A delegate in a separate wheel cannot join that link, so
    this package does the equivalent at import time. Nothing here calls ``register()``: the fakes go
    in first, then the module is imported with the side effect ENABLED, and the backend has to appear
    purely as a consequence of that import.
    """
    registered = set()
    _fake_executorch(monkeypatch, registered)
    loaded = []

    def fake_cdll(path, mode):
        loaded.append(path)
        registered.add("TensorRTBackend")
        return _delegate_handle()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(os.path, "isfile", lambda path: True)

    delegate = load_delegate_module(register_on_import=True)

    assert loaded, "importing the package did not load the delegate library"
    assert delegate.BACKEND_NAME in registered
    # And the import left it fully done, not half done: a later call is a no-op rather than a
    # second load, which is what a defensive caller re-asserting registration would hit.
    delegate.register()
    assert len(loaded) == 1


def test_the_opt_out_env_var_suppresses_the_import_side_effect(monkeypatch):
    """The escape hatch the tests themselves depend on, so it needs its own coverage.

    Every failure-branch test below imports the module with the side effect suppressed in order to
    install its fakes first. If the opt-out silently stopped working, those tests would start
    exercising a real load against the machine's own ExecuTorch and their results would mean
    something else entirely.
    """
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: loaded.append(path))

    delegate = load_delegate_module()

    assert not loaded, "the delegate was loaded despite the registration opt-out"
    assert delegate._delegate is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "value,skip",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("off", False),
        ("other", False),
        (" true ", False),
        ("1", True),
        ("true", True),
        ("TrUe", True),
        ("yes", True),
        ("YES", True),
        ("on", True),
        ("ON", True),
    ],
)
def test_registration_opt_out_boolean_values(monkeypatch, value, skip):
    registered = set()
    _fake_executorch(monkeypatch, registered)
    handle = _delegate_handle()
    loads = []

    def load(path, mode):
        loads.append(path)
        registered.add("TensorRTBackend")
        return handle

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(os.path, "isfile", lambda path: True)
    if value is None:
        monkeypatch.delenv(SKIP_ENV, raising=False)
    else:
        monkeypatch.setenv(SKIP_ENV, value)
    spec = importlib.util.spec_from_file_location(
        "registration_flag_test", DELEGATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(loads) == (0 if skip else 1)
    assert module._delegate is (None if skip else handle)
    assert registered == (set() if skip else {"TensorRTBackend"})


@pytest.mark.unit
def test_registration_opt_out_rejects_missing_on_control(monkeypatch, tmp_path):
    tree = ast.parse(DELEGATE_PATH.read_text())
    checks = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    ]
    assert len(checks) == 1
    values = checks[0].test.comparators[0].elts
    assert sum(value.value == "on" for value in values) == 1
    checks[0].test.comparators[0].elts = [
        value for value in values if value.value != "on"
    ]
    path = tmp_path / "without_on.py"
    path.write_text(ast.unparse(tree))
    monkeypatch.setitem(globals(), "DELEGATE_PATH", path)
    with pytest.raises(AssertionError):
        test_registration_opt_out_boolean_values(monkeypatch, "ON", True)


def test_register_loads_the_delegate_and_registers_the_backend(monkeypatch):
    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    loaded = []

    def fake_cdll(path, mode):
        loaded.append((path, mode))
        registered.add(delegate.BACKEND_NAME)
        return _delegate_handle()

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", fake_cdll)

    assert delegate.register() is None

    assert [path for path, _ in loaded] == ["/fake/delegate.so"]
    # Resolve imports eagerly without adding the delegate's exports to the global namespace.
    assert loaded[0][1] == os.RTLD_NOW | os.RTLD_LOCAL


def test_register_twice_loads_the_delegate_once(monkeypatch):
    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    loads = []

    def fake_cdll(path, mode):
        loads.append(path)
        registered.add(delegate.BACKEND_NAME)
        return _delegate_handle()

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", fake_cdll)

    delegate.register()
    delegate.register()

    assert loads == ["/fake/delegate.so"]


def test_register_reports_a_delegate_that_registers_nothing(monkeypatch):
    """A delegate can load cleanly and still not register, which must not pass silently.

    This is the failure mode of a delegate built against a different runtime: the library
    loads, its initializer runs, and the backend lands in a registry nobody queries. Reporting
    it here is the difference between a clear error and an unavailable-backend mystery later.
    """
    delegate = load_delegate_module()
    _fake_executorch(monkeypatch, set())

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(
        delegate.ctypes, "CDLL", lambda path, mode: types.SimpleNamespace()
    )

    with pytest.raises(delegate.DelegateCompatibilityError, match="did not register"):
        delegate.register()


def test_register_reports_a_missing_executorch(monkeypatch):
    # Genuine absence, where the interpreter sets .name to the root package. A blocked or broken
    # submodule is a different diagnosis (its .name is the full dotted path), covered by
    # test_an_unloadable_executorch_is_not_reported_as_absent, so simulate the root going missing
    # rather than None-blocking the chain, which encodes the broken-install signature instead.
    delegate = load_delegate_module()

    class Boom:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("executorch"):
                raise ModuleNotFoundError(
                    "No module named 'executorch'", name="executorch"
                )
            return None

    for name in [n for n in sys.modules if n.startswith("executorch")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Boom(), *sys.meta_path])

    with pytest.raises(
        delegate.DelegateCompatibilityError, match="ExecuTorch must be installed"
    ):
        delegate.register()


def test_register_reports_an_unloadable_delegate(monkeypatch):
    """A load failure that is not the CPU-wheel case keeps the loader's own message.

    Every OSError used to be answered with "install a CUDA build of executorch", which is the
    wrong instruction for a missing TensorRT, a missing CUDA runtime, or a libstdc++ too old
    for the delegate, and sends the reader after the wrong thing.
    """
    delegate = load_delegate_module()
    _fake_executorch(monkeypatch, set())

    def fail(path, mode):
        raise OSError("libnvinfer.so.11: cannot open shared object file")

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", fail)

    with pytest.raises(delegate.DelegateCompatibilityError) as failure:
        delegate.register()

    # The concrete cause survives, and the misleading advice is absent.
    assert "libnvinfer.so.11" in str(failure.value)
    assert "requires a CUDA build of executorch" not in str(failure.value)


def test_register_reports_a_cpu_executorch_wheel(monkeypatch):
    """The one failure the CPU-wheel diagnosis actually fits.

    This package's pin names no local version label, and a specifier written that way admits any
    label, so a +cpu wheel satisfies it and then cannot resolve
    libexecutorch_extension_cuda.so, which only the CUDA wheels ship.
    """
    delegate = load_delegate_module()
    _fake_executorch(monkeypatch, set())

    def fail(path, mode):
        raise OSError(
            "libexecutorch_extension_cuda.so: cannot open shared object file: "
            "No such file or directory"
        )

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", fail)

    with pytest.raises(
        delegate.DelegateCompatibilityError, match="requires a CUDA build of executorch"
    ):
        delegate.register()


def test_the_delegate_library_is_absent_from_an_unbuilt_package(monkeypatch, tmp_path):
    """Use an unbuilt location because the checkout may contain editable build outputs."""
    delegate = load_delegate_module()
    monkeypatch.setattr(delegate, "__file__", str(tmp_path / "__init__.py"))

    with pytest.raises(delegate.DelegateCompatibilityError, match="missing"):
        delegate._delegate_path()


def test_the_delegate_is_named_the_way_executorch_names_its_own(tmp_path, monkeypatch):
    """The delegate must ship as libexecutorch_backend_<name>.so, like ExecuTorch's own.

    ExecuTorch ships libexecutorch_backend_{cuda,xnnpack,qnn,openvino}.so, so a consumer
    looking for a delegate expects that shape. This is worth pinning because the wheel used to
    declare the library as a setuptools Extension, which renamed it to
    _executorch_backend_tensorrt.<abi>.so: a name that hides what the file is and implies a
    Python ABI the library does not have. It exports no PyInit_ and references no Python
    C-API, so the ABI tag was never meaningful.
    """
    delegate = load_delegate_module()

    assert delegate._DELEGATE_LIBRARY == "libexecutorch_backend_tensorrt.so"

    # setup.py holds its own copy, which CI reads to check the wheel. If only one of the two
    # changed, CI would accept a wheel the runtime cannot load, so pin them to each other.
    # Parsed rather than imported: importing setup.py would run setup().
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    (packaged_name,) = [
        node.value.value
        for node in ast.parse(setup_source).body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "DELEGATE_LIBRARY" for target in node.targets
        )
    ]
    assert packaged_name == delegate._DELEGATE_LIBRARY, (
        "setup.py ships a different filename than the runtime looks for: "
        f"{packaged_name} vs {delegate._DELEGATE_LIBRARY}"
    )

    # The real lookup, against a directory laid out the way the wheel installs. Under lib/, the
    # same place ExecuTorch keeps its own backends, which is also where the shipped CMake package
    # searches, so the Python loader and a C++ consumer resolve one file.
    package = tmp_path / "torch_tensorrt_executorch_runtime"
    (package / "lib").mkdir(parents=True)
    (package / "lib" / delegate._DELEGATE_LIBRARY).write_bytes(b"")
    monkeypatch.setattr(
        delegate.os.path, "abspath", lambda _: str(package / "__init__.py")
    )
    assert delegate._delegate_path() == str(
        package / "lib" / delegate._DELEGATE_LIBRARY
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "layout,expected",
    [
        ("absent", False),
        ("path_attribute", True),
        ("file_attribute_only", True),
        ("no_location", False),
        ("wrong_subdirectory", False),
    ],
)
def test_the_cuda_extension_probe_reads_the_installed_executorch(
    monkeypatch, tmp_path, layout, expected
):
    """Decide the CPU-wheel diagnosis on what is on disk, not on what the error names.

    An ABI failure inside a present libexecutorch_extension_cuda.so names it in the message too,
    so the probe is what keeps that user from being told to reinstall the CUDA wheel they already
    have. Parametrised over the module shapes because the previous version read only __path__,
    which types.ModuleType does not define, so under the fakes these tests use it always answered
    False and the branch it guards was unreachable.
    """
    # Load checkout code with controlled dependencies, regardless of the installed companion.
    delegate = load_delegate_module()

    root = tmp_path / "executorch"
    (root / "lib").mkdir(parents=True)
    if layout != "absent":
        directory = root / ("libs" if layout == "wrong_subdirectory" else "lib")
        directory.mkdir(exist_ok=True)
        (directory / delegate._EXTENSION_CUDA_LIBRARY).write_bytes(b"\x7fELF")

    module = types.ModuleType("executorch")
    if layout in {"absent", "path_attribute", "wrong_subdirectory"}:
        module.__path__ = [str(root)]
    elif layout == "file_attribute_only":
        module.__file__ = str(root / "__init__.py")
    monkeypatch.setitem(sys.modules, "executorch", module)

    assert delegate._extension_cuda_present() is expected


@pytest.mark.unit
def test_the_cuda_extension_probe_survives_no_executorch(monkeypatch):
    # Import failure is not an ABI failure: with no ExecuTorch at all the library is absent, so
    # the CPU-wheel advice is correct and the probe must not raise on the way to saying so.
    monkeypatch.setitem(sys.modules, "executorch", None)
    # Load checkout code with controlled dependencies, regardless of the installed companion.
    delegate = load_delegate_module()
    assert delegate._extension_cuda_present() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "extension_on_disk,expect_cpu_advice",
    [(False, True), (True, False)],
)
def test_a_present_but_broken_cuda_extension_is_not_diagnosed_as_a_cpu_wheel(
    monkeypatch, tmp_path, extension_on_disk, expect_cpu_advice
):
    # The whole point of the probe: the loader names the same library in both cases, so only
    # what is on disk distinguishes "you installed the CPU wheel" from "your CUDA wheel is
    # broken". Deleting the probe from the branch makes both cases give the CPU advice.
    # Load checkout code with controlled dependencies, regardless of the installed companion.
    delegate = load_delegate_module()

    # The full submodule chain, because register() imports the registry before it loads the
    # delegate; a bare ModuleType stops it earlier with a different error.
    _fake_executorch(monkeypatch, set())
    root = tmp_path / "executorch"
    (root / "lib").mkdir(parents=True)
    if extension_on_disk:
        (root / "lib" / delegate._EXTENSION_CUDA_LIBRARY).write_bytes(b"\x7fELF")
    sys.modules["executorch"].__path__ = [str(root)]

    monkeypatch.setattr(delegate, "_delegate_path", lambda: str(tmp_path / "d.so"))
    monkeypatch.setattr(
        delegate.ctypes,
        "CDLL",
        lambda *a, **k: (_ for _ in ()).throw(
            OSError("libexecutorch_extension_cuda.so: cannot open shared object file")
        ),
    )

    with pytest.raises(delegate.DelegateCompatibilityError) as raised:
        delegate.register()

    says_cpu = "a CPU build satisfies the version pin" in str(raised.value)
    assert says_cpu is expect_cpu_advice, (
        "the CPU-wheel advice fired for a present extension"
        if says_cpu
        else "the CPU-wheel advice did not fire for a genuinely absent extension"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "message,expect_install_advice",
    [
        ("No module named 'executorch'", True),
        # A dependency of an installed ExecuTorch going missing is also a ModuleNotFoundError, but
        # its name is that dependency, and telling this user to install ExecuTorch is wrong.
        ("No module named 'flatbuffers'", False),
        # A submodule of an installed ExecuTorch that is absent or blocked: CPython sets .name to
        # the full dotted path, not the root, so this is the broken-install diagnosis rather than
        # the absent-package one. Comparing only the first dotted segment misreported it as
        # ExecuTorch being uninstalled.
        ("No module named 'executorch.extension.pybindings.portable_lib'", False),
        # A blocked sys.modules entry means the package was found and something inside it failed,
        # which is the broken-install diagnosis rather than the absent-package one.
        ("import of executorch.extension halted; None in sys.modules", False),
        ("libexecutorch.so: version 'CXXABI_1.3.15' not found", False),
        ("libcudart.so.13: cannot open shared object file", False),
    ],
)
def test_an_unloadable_executorch_is_not_reported_as_absent(
    monkeypatch, message, expect_install_advice
):
    # An ABI mismatch reaches the same except clause as a missing package but needs the opposite
    # repair. Answering both with "install executorch" told the user to reinstall what they had.
    # Load checkout code with controlled dependencies, regardless of the installed companion.
    delegate = load_delegate_module()

    # A finder, because the code under test uses a plain `import` statement rather than
    # importlib.import_module, so patching that function would not be reached.
    class Boom:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("executorch"):
                raise (
                    # name= as the interpreter sets it, since the diagnosis reads it to tell a
                    # genuinely absent ExecuTorch from a missing transitive dependency. The
                    # message names whichever module was not found, so derive it from there.
                    ModuleNotFoundError(
                        message, name=message.split("'")[1] if "'" in message else name
                    )
                    if message.startswith("No module named")
                    else ImportError(message)
                )
            return None

    for name in [n for n in sys.modules if n.startswith("executorch")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Boom(), *sys.meta_path])

    with pytest.raises(delegate.DelegateCompatibilityError) as raised:
        delegate.register()

    advises_install = "must be installed" in str(raised.value)
    assert (
        advises_install is expect_install_advice
    ), f"for {message!r} the diagnosis was: {raised.value}"


@pytest.mark.parametrize(
    "search_path,expected",
    [
        (None, "could not find it, not that it is incompatible"),
        ("/opt/cuda/lib64:", "empty entry"),
        (":", "empty entry"),
        ("/opt/cuda/lib64", "could not find it, not that it is incompatible"),
    ],
)
def test_a_library_the_loader_cannot_find_is_not_reported_as_an_abi_mismatch(
    monkeypatch, search_path, expected
):
    # An empty entry in the search path is read as the working directory and stops this package's
    # own origin-relative entries resolving, so a correct install fails from some directories only.
    delegate = load_delegate_module()

    class Boom:
        def find_spec(self, name, path=None, target=None):
            if name.startswith("executorch"):
                raise ImportError(
                    "libexecutorch_extension_cuda.so: cannot open shared object file: "
                    "No such file or directory"
                )
            return None

    for name in [n for n in sys.modules if n.startswith("executorch")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Boom(), *sys.meta_path])
    if search_path is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", search_path)

    with pytest.raises(delegate.DelegateCompatibilityError) as raised:
        delegate.register()
    assert expected in str(raised.value), raised.value


def _documented_path_recipes() -> list[tuple[str, bool, str]]:
    """Every documented command that imports this package only to print a path.

    The recipes are what a consumer copies, and the import in them registers the delegate, so one
    written without the opt-out fails wherever the delegate cannot load. Returns the file it came
    from, whether it sets the opt-out, and the code it runs.
    """
    recipes = []
    for relative in ("README.md", "cmake/executorch_backend_tensorrt-config.cmake"):
        text = (COMPANION_ROOT / relative).read_text(encoding="utf-8")
        for line in text.splitlines():
            if (
                "python -c " not in line
                or "torch_tensorrt_executorch_runtime" not in line
            ):
                continue
            before, _, rest = line.partition("python -c ")
            quote = rest[0]
            recipes.append(
                (
                    relative,
                    before.endswith(f"{SKIP_ENV}=1 "),
                    rest[1 : rest.index(quote, 1)],
                )
            )
    return recipes


def test_the_documented_path_recipes_run_without_a_loadable_delegate(tmp_path):
    """Run each documented path query where the delegate cannot load, which is where they are used.

    A consumer runs these before anything is built, and CMake runs one of them from a configure
    step. Nothing here is stubbed but the ExecuTorch distribution metadata one of them reads: the
    package comes from the checkout, which carries no built delegate, so a plain import raises.
    Each recipe is then run again with the opt-out removed, so the variable is shown to be what
    makes the recipe work rather than decoration.
    """
    site = tmp_path / "site"
    (site / "executorch").mkdir(parents=True)
    (site / "executorch/__init__.py").touch()
    metadata = site / "executorch-1.6.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: executorch\nVersion: 1.6.0\n", encoding="utf-8"
    )
    recipes = _documented_path_recipes()
    assert len(recipes) == 3, recipes
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in (SKIP_ENV, "PYTHONPATH")
    }
    environment["PYTHONPATH"] = os.pathsep.join([str(site), str(COMPANION_ROOT)])
    for relative, sets_skip, code in recipes:
        assert (
            sets_skip
        ), f"{relative} imports the package for a path without {SKIP_ENV}"
        opted_out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**environment, SKIP_ENV: "1"},
            timeout=60,
        )
        assert opted_out.returncode == 0, opted_out.stderr
        assert str(COMPANION_ROOT / PACKAGE) in opted_out.stdout, opted_out.stdout
        plain = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        assert plain.returncode != 0, plain.stdout
        assert "DelegateCompatibilityError" in plain.stderr, plain.stderr
