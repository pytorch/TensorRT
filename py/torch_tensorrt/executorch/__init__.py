import importlib.util
from pathlib import Path
from typing import NoReturn, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.exir import EdgeCompileConfig


def _has_executorch_exir() -> bool:
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


if not _has_executorch_exir():

    def __getattr__(name: str) -> NoReturn:
        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch with executorch.exir is required. "
            'Install with: pip install "torch_tensorrt[executorch]"'
        )

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]
else:
    from torch_tensorrt.executorch.backend import TensorRTBackend
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    def get_edge_compile_config() -> "EdgeCompileConfig":
        """Return the EdgeCompileConfig used for Torch-TensorRT ExecuTorch export."""
        from executorch.exir import EdgeCompileConfig

        return EdgeCompileConfig(_check_ir_validity=False)

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]


# Handle for the loaded delegate, kept so the load is done once per process.
_runtime_delegate = None

# Why the load did not happen, kept so a caller that needs the delegate can say what went
# wrong. Without this, a delegate that exists but cannot load surfaces much later as an
# unregistered backend, which describes the symptom and hides the cause.
_runtime_delegate_error: Optional[str] = None


def _load_runtime_delegate() -> bool:
    """Load the prebuilt TensorRT delegate so it registers itself with ExecuTorch.

    The delegate registers through a static initializer, so it has to be loaded before a
    program runs a model that uses it. Returns True when a library was loaded, and False when
    this install carries none, which is the case for a build without the delegate.

    Safe to call more than once: the dynamic loader returns the already-loaded library.
    """
    global _runtime_delegate, _runtime_delegate_error
    if _runtime_delegate is not None:
        return True

    package_root = Path(__file__).resolve().parent.parent
    # Match any SONAME major so this does not need updating when the major changes.
    candidates = sorted(package_root.glob("lib/libexecutorch_backend_tensorrt.so*"))
    if not candidates:
        _runtime_delegate_error = (
            f"no delegate library found under {package_root / 'lib'}; this package was "
            "built without the ExecuTorch delegate"
        )
        return False

    import ctypes

    try:
        # RTLD_GLOBAL so the delegate can resolve runtime symbols already loaded by the
        # ExecuTorch extension, and so anything loaded later can resolve against it.
        _runtime_delegate = ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        # A delegate that cannot load is not a reason to make the export path unimportable,
        # since exporting needs no delegate. The usual cause is benign: the delegate links the
        # ExecuTorch runtime, and a program that only exports has not imported it.
        #
        # The error is kept rather than discarded. A genuinely broken delegate, a missing CUDA
        # or TensorRT dependency or a wrong search path, otherwise surfaces much later as an
        # unregistered backend.
        _runtime_delegate_error = f"{candidates[0]}: {error}"
        return False
    return True


def require_runtime_delegate() -> None:
    """Raise if the prebuilt TensorRT delegate is not loaded.

    Exporting does not need the delegate, so importing this module never fails. A caller about
    to run a delegated model does need it, and this reports why it is unavailable instead of
    letting the runtime report an unregistered backend later.

    A first attempt can fail for a reason that later stops being true. The delegate links the
    ExecuTorch runtime, so importing this module before ExecuTorch leaves that library unfindable,
    and the load fails through no fault of the installation. Retrying here means that ordering is
    not fatal for the rest of the process.
    """
    if _runtime_delegate is not None:
        return
    if _load_runtime_delegate():
        return
    detail = _runtime_delegate_error or "reason unknown"
    raise RuntimeError(
        "the ExecuTorch TensorRT delegate is not loaded, so a delegated model cannot run: "
        f"{detail}"
    )


_load_runtime_delegate()
