import ast
import ctypes
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

DELEGATE_PATH = (
    Path(__file__).parents[4]
    / "py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/__init__.py"
)
SETUP_PATH = Path(__file__).parents[4] / "py/torch-tensorrt-executorch-runtime/setup.py"
SKIP_ENV = "TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION"


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
        return types.SimpleNamespace()

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


def test_register_loads_the_delegate_and_registers_the_backend(monkeypatch):
    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    loaded = []

    def fake_cdll(path, mode):
        loaded.append((path, mode))
        registered.add(delegate.BACKEND_NAME)
        return types.SimpleNamespace()

    monkeypatch.setattr(delegate, "_delegate_path", lambda: "/fake/delegate.so")
    monkeypatch.setattr(delegate.ctypes, "CDLL", fake_cdll)

    assert delegate.register() is None

    assert [path for path, _ in loaded] == ["/fake/delegate.so"]
    # RTLD_NOW so a missing symbol surfaces here instead of mid-execution, and RTLD_LOCAL
    # because the delegate resolves its own imports and exports nothing others need.
    assert loaded[0][1] == os.RTLD_NOW | os.RTLD_LOCAL


def test_register_twice_loads_the_delegate_once(monkeypatch):
    delegate = load_delegate_module()
    registered = set()
    _fake_executorch(monkeypatch, registered)
    loads = []

    def fake_cdll(path, mode):
        loads.append(path)
        registered.add(delegate.BACKEND_NAME)
        return types.SimpleNamespace()

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


def test_the_delegate_library_is_absent_from_a_source_checkout():
    """The delegate is a build artifact, so locating it must fail cleanly when it is missing.

    Run from a checkout, nothing has been built, so this exercises the real lookup rather than
    a stubbed one and pins the error users see when they import the package without installing
    the wheel.
    """
    delegate = load_delegate_module()

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
    # By file path, like every other test here. import_module needs the package installed, and
    # this lane installs ExecuTorch but not the delegate wheel, which is built by a separate job.
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
    # By file path, like every other test here. import_module needs the package installed, and
    # this lane installs ExecuTorch but not the delegate wheel, which is built by a separate job.
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
    # By file path, like every other test here. import_module needs the package installed, and
    # this lane installs ExecuTorch but not the delegate wheel, which is built by a separate job.
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
    # By file path, like every other test here. import_module needs the package installed, and
    # this lane installs ExecuTorch but not the delegate wheel, which is built by a separate job.
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
