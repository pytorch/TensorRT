"""Register the Torch-TensorRT delegate with the installed ExecuTorch runtime.

Importing this package is all it takes. ExecuTorch's own delegates register because they are
linked into its pybindings extension, so loading that extension pulls them in and their static
initializers run. A delegate shipped in a separate wheel cannot join that link, and ExecuTorch has
no discovery hook for out-of-tree backends, so this package performs the equivalent step itself at
import time.

There is deliberately no runtime API here. Once the backend is registered, everything else belongs
to ExecuTorch:

    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program("model.pte")
    outputs = program.load_method("forward").execute((tensor,))

Set ``TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1`` to import the module without loading the
delegate. That is for tooling that wants the metadata only; a normal consumer never needs it.
"""

from __future__ import annotations

import ctypes
import os
import threading

BACKEND_NAME = "TensorRTBackend"
# The same name ExecuTorch gives its own delegates, and the exact filename the wheel ships.
_DELEGATE_LIBRARY = "libexecutorch_backend_tensorrt.so"

_delegate: ctypes.CDLL | None = None
# Registration happens in the delegate's static initializer, so it takes effect inside dlopen,
# before ctypes.CDLL returns and before _delegate is assigned. Every check below therefore has to
# sit inside one critical section: a second thread that squeezed between the load and the
# assignment would see the backend registered with _delegate still None, which is exactly what a
# foreign delegate owning the name looks like.
_registration_lock = threading.Lock()


class DelegateCompatibilityError(ImportError):
    """The delegate could not be loaded against the installed ExecuTorch runtime."""


_EXTENSION_CUDA_LIBRARY = "libexecutorch_extension_cuda.so"


def _extension_cuda_present() -> bool:
    """Whether the installed ExecuTorch actually ships the CUDA extension.

    Resolved from the imported package rather than a hardcoded path, so it follows the
    distribution the loader would have used. ``__file__`` as well as ``__path__``, because a
    namespace-style or synthesised module may carry only one of them, and treating "no location
    at all" as "the file is missing" would send a user with a working CUDA wheel off to
    reinstall it.
    """
    try:
        import executorch
    except ImportError:
        return False
    roots = list(getattr(executorch, "__path__", None) or ())
    location = getattr(executorch, "__file__", None)
    if location:
        roots.append(os.path.dirname(os.path.abspath(location)))
    return any(
        os.path.isfile(os.path.join(root, "lib", _EXTENSION_CUDA_LIBRARY))
        for root in roots
    )


def _delegate_path() -> str:
    # Resolved next to this file rather than through the import system, so it also works
    # before the package is importable. A fixed filename now: the delegate is shipped as
    # package data under its real name, not renamed by setuptools.
    #
    # ``lib/`` rather than the package root, matching where ExecuTorch keeps its own backends
    # (``executorch/lib/libexecutorch_backend_cuda.so`` and friends). A C++ consumer finds this
    # library through the CMake package in ``lib/cmake``, which searches the same directory, so
    # the two consumers agree on one location.
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    path = os.path.join(directory, _DELEGATE_LIBRARY)
    if not os.path.isfile(path):
        raise DelegateCompatibilityError(
            f"The Torch-TensorRT ExecuTorch delegate library is missing from {directory}. "
            "This package must be installed from a wheel; a source checkout contains no "
            "built delegate."
        )
    return path


def register() -> None:
    """Load the delegate so ExecuTorch can execute TensorRT-delegated programs.

    Called once when this package is imported, so a user never has to. It stays public and named
    because the import side effect is the whole point of the package and a reader needs somewhere to
    look, and because a caller that imported the package defensively can re-assert registration
    without reaching into a private name.

    Registration happens in the delegate's own static initializer, which calls into the backend
    registry that lives in the ExecuTorch runtime. Importing ExecuTorch first is what puts that
    runtime in the process; the delegate then binds to the same copy through its DT_NEEDED rather
    than bringing one of its own.

    Idempotent, and safe to call after ``executorch.runtime`` has already been imported. That used
    to be an error, because the delegate arrived as a substitute for ExecuTorch's own Python
    extension and had to get in first. It no longer substitutes anything.

    Thread safe: the whole first registration is serialized, because it lands during ``dlopen``
    while ``_delegate`` is assigned after it, and a caller observing that gap could not tell this
    package's own load from a foreign delegate holding the name.
    """
    if _delegate is not None:
        return

    with _registration_lock:
        # Re-checked under the lock: a caller that queued here while the winner was loading would
        # otherwise redo the whole registration and then reject itself for the registration the
        # winner just made.
        if _delegate is not None:
            return
        _register_locked()


def _register_locked() -> None:
    global _delegate

    try:
        import executorch.extension.pybindings.portable_lib  # noqa: F401
    except ImportError as error:
        # "Not installed" and "installed but unloadable" need different repairs, and the second
        # is what an ABI mismatch looks like: the module is found, its extension fails to load.
        # Answering both with "install executorch" sends that user to reinstall what they have.
        # ModuleNotFoundError covers a genuinely absent package; a bare ImportError whose message
        # names no shared object is the same thing seen through a blocked sys.modules entry. An
        # ABI failure, by contrast, always names the library that would not load.
        # ModuleNotFoundError naming executorch itself. Testing the type alone was wrong: if an
        # installed ExecuTorch fails to import because one of its own transitive dependencies is
        # missing, the exception is also a ModuleNotFoundError, and its .name is that dependency.
        # That user was told to install ExecuTorch, which they already have.
        # Exact match, not the top-level segment: CPython sets .name to the full dotted path when a
        # submodule such as executorch.extension.pybindings.portable_lib is the thing that is
        # absent or blocked, and to the bare "executorch" only when the root package itself is
        # missing. Splitting on "." and comparing the first segment reported a blocked submodule as
        # ExecuTorch being uninstalled, which is the ABI case this branch exists to separate out.
        absent = (
            isinstance(error, ModuleNotFoundError)
            and (error.name or "") == "executorch"
        )
        if absent:
            raise DelegateCompatibilityError(
                "ExecuTorch must be installed to load the Torch-TensorRT delegate. Install "
                "executorch from the same release matrix as this package. The import failed "
                f"with: {error}"
            ) from error
        raise DelegateCompatibilityError(
            "ExecuTorch is installed but its Python bindings could not be loaded, which "
            "usually means it was built against a different C++ or CUDA runtime than this "
            f"delegate. The import failed with: {error}"
        ) from error

    # RTLD_NOW so an under-linked delegate reports the missing symbol here, rather than
    # crashing later inside execute(). RTLD_LOCAL because the delegate exports nothing anyone
    # needs; it resolves its own imports through its DT_NEEDED entries, so widening the
    # process-global namespace would only add collisions.
    path = _delegate_path()
    # Snapshot the registry before loading. ExecuTorch's register_backend keeps the FIRST
    # registration of a name and rejects later ones, so if TensorRTBackend is already present a
    # second copy of this delegate is live and our load does not win. Checking only that the name
    # is present afterward would report success while a different library serves the backend, so
    # require that this load is the one that added it.
    #
    # "Already present" is not the same as "someone else's": anything that dlopened this very file
    # first, a wheel check or a debugger or a wrapper, registers the name through OUR library.
    # Rejecting that case blamed a second delegate that does not exist, and kept rejecting it on
    # every retry, so adopt the existing registration instead when the file we would load is the one
    # already in the process.
    if BACKEND_NAME in _registered_backend_names():
        already_loaded = _delegate_already_loaded(path)
        if already_loaded is None:
            raise DelegateCompatibilityError(
                f"{BACKEND_NAME} is already registered before loading {path}, and that library is "
                "not the one in this process, so another copy of the delegate is present. "
                "ExecuTorch keeps the first registration, so the copy this package ships would "
                "not be the one used. Import this package once, and do not load a second "
                "Torch-TensorRT delegate alongside it."
            )
        _delegate = already_loaded
        return
    try:
        loaded = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as error:
        # The CPU-wheel diagnosis fits exactly one failure: the delegate has a DT_NEEDED on
        # libexecutorch_extension_cuda.so, which only ExecuTorch's CUDA wheels ship, and the pin
        # this package declares names no local version label, so a +cpu wheel satisfies it and
        # then cannot resolve that library. Every other OSError here means something else --
        # a missing TensorRT or CUDA runtime, an undefined symbol, a libstdc++ too old for the
        # delegate -- and answering all of them with "install a CUDA executorch" sends the
        # reader after the wrong thing, so keep the loader's own message for those.
        #
        # "Names the library" is not the same as "the library is missing": an ABI failure inside a
        # present libexecutorch_extension_cuda.so names it too, and telling that user to install
        # the CUDA wheel they already have is the same wrong-thing problem. So confirm it is
        # actually absent from the ExecuTorch that is installed before blaming a CPU wheel.
        if (
            "libexecutorch_extension_cuda" in str(error)
            and not _extension_cuda_present()
        ):
            raise DelegateCompatibilityError(
                f"Could not load the Torch-TensorRT ExecuTorch delegate from {path}. This "
                "requires a CUDA build of executorch, which ships "
                "libexecutorch_extension_cuda.so; a CPU build satisfies the version pin but "
                "not this dependency. Install torch, executorch, torch-tensorrt, and this "
                "package from the same release matrix."
            ) from error
        raise DelegateCompatibilityError(
            f"Could not load the Torch-TensorRT ExecuTorch delegate from {path}: {error}. "
            "The delegate links ExecuTorch's prebuilt runtime, TensorRT, and the CUDA runtime "
            "from their own wheels, so install torch, executorch, torch-tensorrt, and this "
            "package from the same release matrix."
        ) from error

    if not _is_registered():
        raise DelegateCompatibilityError(
            f"Loading {path} did not register {BACKEND_NAME} with the ExecuTorch runtime, so "
            "a delegated program would fail to load. The delegate and the installed "
            "ExecuTorch were probably built against different runtimes."
        )
    _delegate = loaded


def _delegate_already_loaded(path: str) -> ctypes.CDLL | None:
    """The handle for ``path`` if that exact library is already in this process, else ``None``.

    ``dlopen`` on an already-loaded library returns the existing handle and bumps its reference
    count rather than mapping a second copy, so this cannot introduce the duplicate it is checking
    for. ``RTLD_NOLOAD`` is what makes the question safe to ask: it refuses to load anything, so a
    library that is not present yields ``None`` instead of being pulled in as a side effect of the
    test. It is absent on some platforms, and this delegate is Linux-only, so treat a missing flag
    as "cannot tell" rather than guessing.
    """
    noload = getattr(os, "RTLD_NOLOAD", None)
    if noload is None:
        return None
    try:
        return ctypes.CDLL(path, mode=noload | os.RTLD_LOCAL)
    except OSError:
        return None


def _registered_backend_names() -> list[str]:
    from executorch.extension.pybindings.portable_lib import (
        _get_registered_backend_names,
    )

    return _get_registered_backend_names()


def _is_registered() -> bool:
    return BACKEND_NAME in _registered_backend_names()


# The import IS the registration, so a user never calls anything. The opt-out exists for callers that
# need the module without the side effect: the tests, which drive the failure branches against fakes
# and so must install those fakes before anything loads, and a packaging step that only wants the
# metadata. ExecuTorch does the same thing in its Qualcomm backend, whose __init__ reads
# EXECUTORCH_BUILDING_WHEEL to skip its own import-time SDK setup.
#
# Failure is deliberately loud rather than swallowed. This wheel exists only to register the backend,
# so a load it cannot complete leaves nothing useful behind, and the diagnosis here names the actual
# cause (a CPU-only ExecuTorch wheel, an ABI mismatch, an absent ExecuTorch) which ExecuTorch's own
# later "backend not available" cannot.
if os.getenv("TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION", "0").lower() not in (
    "1",
    "true",
    "yes",
):
    register()

__all__ = ["BACKEND_NAME", "DelegateCompatibilityError", "register"]
