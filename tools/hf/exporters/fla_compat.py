"""Detect broken/missing ``fla`` installs and re-raise with the fix.

Kimi's remote code needs ``fla`` (flash-linear-attention), and
``transformers.exporters`` imports executorch, which imports ``fla`` when it is
installed. The PyPI ``fla-core`` releases (<=0.5.2) fail that import under
triton 3.9, and ``tilelang`` (probed by ``fla``) does not import on Python 3.14.
"""

from __future__ import annotations

import traceback

FLA_INSTALL_HINT = (
    "pip install --no-deps "
    "'git+https://github.com/fla-org/flash-linear-attention.git'"
)


def describe_fla_failure(exc: BaseException) -> str | None:
    """Return a fix description if ``exc`` came from a bad ``fla`` install."""
    trace = "".join(traceback.format_exception(exc))
    if "tilelang" in trace or "tvm_ffi" in trace:
        return (
            "importing fla failed because its optional 'tilelang' backend is "
            "installed but broken on this Python. Fix: pip uninstall -y tilelang"
        )
    if "/fla/" in trace and "not kernel arguments" in trace:
        return (
            "the installed fla-core is incompatible with this triton "
            "(autotune key names are not kernel arguments); PyPI fla-core "
            f"<=0.5.2 all hit this. Fix: pip uninstall -y fla-core; {FLA_INSTALL_HINT}"
        )
    if isinstance(exc, ModuleNotFoundError) and exc.name == "fla":
        return f"fla is not installed. Fix: {FLA_INSTALL_HINT}"
    if "/fla/" in trace:
        return f"importing fla failed. Try: {FLA_INSTALL_HINT}"
    return None


def reraise_with_fla_fix(exc: BaseException) -> None:
    """Raise a RuntimeError chained from ``exc`` if it is an fla problem."""
    fix = describe_fla_failure(exc)
    if fix is not None:
        raise RuntimeError(f"fla problem: {fix}") from exc
