"""Check installed metadata without importing or provisioning GPU packages."""

from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet

CORE_RANGE = ">=2.16.0.dev0"


class CoreCompatibilityError(ImportError):
    """The provisioned core cannot satisfy this adapter's public API contract."""


def check_core_compatibility() -> str:
    try:
        installed = version("torch-tensorrt")
    except PackageNotFoundError:
        try:
            installed = version("torch-tensorrt-rtx")
        except PackageNotFoundError as exc:
            raise CoreCompatibilityError(
                "Torch-TensorRT is missing. Provision a compatible GPU environment "
                "before starting execution jobs; the adapter does not install it."
            ) from exc
    if installed not in SpecifierSet(CORE_RANGE):
        raise CoreCompatibilityError(
            f"Unsupported Torch-TensorRT {installed}; expected {CORE_RANGE}. "
            "Use an adapter/core pair from the compatibility matrix."
        )
    return installed
