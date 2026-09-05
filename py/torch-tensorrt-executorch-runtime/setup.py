"""Build the precompiled Torch-TensorRT backend for ExecuTorch Python."""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import uuid

import torch
from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

try:
    # setuptools >= 70.1 vends the command; older toolchains still import it from wheel.
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - depends on the build toolchain version
    from wheel.bdist_wheel import bdist_wheel

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BAZEL_TARGET = "//py/torch-tensorrt-executorch-runtime/native:delegate_native"
BUILD_NONCE = os.getenv("TORCH_TENSORRT_EXECUTORCH_BUILD_NONCE", uuid.uuid4().hex)
CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"
# Named the way ExecuTorch names its own delegates, because that is what this now is. The wheel
# ships this exact filename: a consumer looking for a delegate beside ExecuTorch's own
# libexecutorch_backend_cuda.so finds the same shape here.
DELEGATE_LIBRARY = "libexecutorch_backend_tensorrt.so"
# Checked in rather than generated: it has no build-time inputs. Only the companion version file is
# written at build time, because the version is not known until then.
_CMAKE_CONFIG_SOURCE = HERE / "cmake" / "torchtrt_executorch-config.cmake"


def pinned_executorch_version() -> str:
    """Read the ExecuTorch version the repository pins, or "" if the pin file is unavailable."""
    pin_file = REPO_ROOT / "dev_dep_versions.yml"
    if not pin_file.is_file():
        return ""
    match = re.search(
        r'^__executorch_version__:\s*"?([^"\s]+)"?\s*$',
        pin_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else ""


def executorch_cmake_prefix_path() -> str:
    """Locate the CMake package of the ExecuTorch wheel this delegate builds against.

    The delegate links the runtime out of the installed wheel, so the wheel that is present
    while building is the one it becomes compatible with. Both the path and the version below
    come from one ``importlib.metadata`` distribution, not from ``executorch.__path__[0]``:
    ``executorch`` is a namespace package, so any directory on ``sys.path`` holding an
    ``executorch/`` subdirectory prepends a root, and index 0 could then name a source tree while
    the version check validated the installed wheel. The compiler and the check have to be looking
    at the same thing for either to mean anything.
    """
    distribution = importlib.metadata.distribution("executorch")
    package_root = pathlib.Path(str(distribution.locate_file("executorch")))
    if not package_root.is_dir():
        raise RuntimeError(
            f"The executorch distribution reports its package at {package_root}, which is not "
            "a directory. Reinstall ExecuTorch from the pinned nightly CUDA channel."
        )
    prefix = package_root / "share" / "cmake"
    if not (prefix / "executorch-config.cmake").is_file():
        raise RuntimeError(
            f"The installed ExecuTorch at {package_root} ships no CMake package, so the "
            "delegate cannot be configured against it. Install a wheel from the pinned "
            "nightly CUDA channel."
        )
    # The version too, not just the path. install_requires below names whatever is installed, so
    # building against the wrong wheel produced a coherent-looking artifact: the delegate links
    # that runtime, the ELF guard compares it against that same runtime, and the metadata requires
    # it -- all three agreeing on a runtime the repository does not pin. Local editable builds are
    # exempt via the escape hatch, because contributors legitimately test against other trees.
    pinned = pinned_executorch_version()
    installed = public_version(distribution.version)
    if (
        pinned
        and public_version(pinned) != installed
        and not os.getenv("TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH")
    ):
        raise RuntimeError(
            f"The installed ExecuTorch is {installed} but dev_dep_versions.yml pins "
            f"{pinned}. The delegate links this wheel's runtime and declares a dependency on "
            "it, so building against another version ships a wheel that requires the wrong "
            "ExecuTorch. Install the pinned wheel, or set "
            "TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH=1 to build anyway."
        )
    return str(prefix)


def torchtrt_version() -> str:
    if value := os.getenv("TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION"):
        return value
    version_py = REPO_ROOT / "py/torch_tensorrt/_version.py"
    if version_py.exists():
        if m := re.search(r'__version__\s*=\s*["\']([^"\']+)', version_py.read_text()):
            return m.group(1)
    # version.txt carries the in-development placeholder 2.15.0a0. The root build strips that
    # suffix for real artifacts and it is published on no index, so recording it as a
    # torch-tensorrt== requirement yields a wheel pip cannot install. Fail closed rather than
    # bake in that requirement: a from-source build sets the version it pairs with explicitly,
    # the way CI derives it from the installed torch-tensorrt.
    raise RuntimeError(
        "Cannot determine the torch-tensorrt version this runtime wheel requires. Set "
        "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION to the torch-tensorrt version it pairs with "
        "(CI reads it from the installed torch-tensorrt). Falling back to version.txt would "
        "record torch-tensorrt==2.15.0a0, which is published nowhere."
    )


def public_version(version: str) -> str:
    """Drop a PEP 440 local suffix that may not be present on package indexes."""
    return version.partition("+")[0]


def installed_version(distribution: str) -> str:
    """Return the version of a dependency in the native build environment."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"{distribution} must be installed to build the ExecuTorch runtime wheel"
        ) from error


def tensorrt_distribution() -> str:
    """Return the TensorRT distribution matching the PyTorch CUDA build."""
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("CUDA-enabled PyTorch is required to build this wheel")
    if cuda_version.startswith("12.6"):
        return "tensorrt-cu12"
    if cuda_version.startswith("13."):
        return "tensorrt-cu13"
    raise RuntimeError(f"Unsupported CUDA version: {cuda_version}")


# torch-tensorrt publishes both CUDA channels, so the delegate is built for both.
_SUPPORTED_CUDA_MAJORS = frozenset({"12", "13"})


def require_supported_cuda() -> None:
    """Reject builds whose PyTorch uses a CUDA the delegate is not published for.

    Both majors are accepted because torch-tensorrt publishes both channels. The check is on the
    major only: a minor bump inside a major does not change the ABI the delegate links against, and
    pinning a minor here made the wheel unbuildable against a newer nightly for no reason.
    """
    cuda_version = torch.version.cuda
    major = (cuda_version or "").split(".")[0]
    if major not in _SUPPORTED_CUDA_MAJORS:
        supported = " or ".join(sorted(_SUPPORTED_CUDA_MAJORS))
        raise RuntimeError(
            f"PyTorch built against CUDA {supported} is required to build this wheel "
            f"(found CUDA {cuda_version or 'None'})"
        )


class BazelBuild(build_py):
    """Build the delegate with Bazel and place it in the package under its real name.

    Not a ``build_ext``/``Extension``: the delegate exports no ``PyInit_``, references no
    Python C-API symbol, and links no libpython; it is a plain shared library that ctypes
    loads. Declaring it an extension made setuptools rename it to
    ``_executorch_backend_tensorrt.<abi>.so``, which both hides that it is an ExecuTorch
    delegate and implies a Python ABI it does not have. The platform tag the extension was
    buying is set directly instead: ``Distribution.has_ext_modules`` keeps the wheel
    non-pure, and ``WheelTag`` below sets the interpreter and ABI to py3/none.
    """

    def run(self) -> None:
        super().run()

        if sys.platform != "linux":
            raise RuntimeError("The ExecuTorch TensorRT delegate supports Linux only")

        bazel = shutil.which("bazelisk") or shutil.which("bazel")
        if bazel is None:
            raise RuntimeError("Could not find bazelisk or bazel in PATH")

        compilation_mode = (
            "dbg" if os.getenv("TORCH_TENSORRT_EXECUTORCH_DEBUG") else "opt"
        )
        command = [
            bazel,
            "build",
            BAZEL_TARGET,
            "--config=linux",
            "--config=python",
            f"--compilation_mode={compilation_mode}",
            f"--action_env=PYTHON_BIN_PATH={sys.executable}",
            f"--action_env=EXECUTORCH_CMAKE_PREFIX_PATH={executorch_cmake_prefix_path()}",
            f"--action_env=TORCH_TENSORRT_EXECUTORCH_BUILD_NONCE={BUILD_NONCE}",
        ]
        dist_dir_arch = (
            "aarch64-linux-gnu"
            if platform.machine() in {"aarch64", "arm64"}
            else "x86_64-linux-gnu"
        )
        dist_dir = REPO_ROOT / "third_party/dist_dir" / dist_dir_arch
        if dist_dir.is_dir():
            command.append(f"--distdir={dist_dir}")
        command.extend(shlex.split(os.getenv("BAZEL_ARGS", "")))

        env = os.environ.copy()
        env.setdefault("TORCH_PATH", str(pathlib.Path(torch.__file__).resolve().parent))
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

        bazel_bin = pathlib.Path(
            subprocess.check_output(
                [
                    bazel,
                    "info",
                    "bazel-bin",
                    f"--compilation_mode={compilation_mode}",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
            ).strip()
        )
        built = (
            bazel_bin
            / "py/torch-tensorrt-executorch-runtime/native/delegate_native/lib"
            / DELEGATE_LIBRARY
        )
        if not built.is_file():
            raise RuntimeError(f"Bazel did not produce {built}")

        output = (
            pathlib.Path(self.build_lib)
            / "torch_tensorrt_executorch_runtime"
            / "lib"
            / DELEGATE_LIBRARY
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Every stale artifact, not just a shared object in lib/. An incremental build over a tree
        # that once produced the bundled ExecuTorch runtime leaves those files under build_lib, and
        # build_py copies that directory into the wheel wholesale, so a local rebuild would
        # republish exactly the runtime and the Python API this package no longer ships.
        # package_data names one filename and would not pull them in; the staleness is in the build
        # tree, not the manifest. CI never sees it, building from a fresh checkout.
        #
        # The package ROOT as well as lib/: the delegate moved into lib/, but the files this
        # removes were written to the root by the previous layout, so scanning only the new
        # directory leaves every one of them in place.
        package_root = output.parent.parent
        for stale in (*package_root.glob("*.so*"), package_root / "runtime.py"):
            if stale.is_file():
                stale.unlink()
        for stale in output.parent.glob("*.so*"):
            if stale != output:
                stale.unlink()
        shutil.copy2(built, output)
        self._install_cmake_package(output.parent.parent)

    def _install_cmake_package(self, package_dir: pathlib.Path) -> None:
        """Ship a CMake package so a C++ app can link the delegate out of the wheel.

        ExecuTorch ships its backends as prebuilt shared libraries plus a CMake package, so a C++
        app links ``executorch::backend_cuda`` and the backend registers itself. Without an
        equivalent here the delegate is reachable only from Python, even though the shared library
        in this wheel is a drop-in sibling of ExecuTorch's own backends.

        The config is checked in; only the version file is generated, because the version is not
        known until the wheel is built.
        """
        cmake_dir = package_dir / "lib" / "cmake" / "torchtrt_executorch"
        cmake_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_CMAKE_CONFIG_SOURCE, cmake_dir / _CMAKE_CONFIG_SOURCE.name)
        version = torchtrt_version()
        # find_package compares dotted integers, so a dev suffix has to be dropped from the value
        # it reads while the full version stays visible for a human.
        numeric = re.match(r"[0-9]+(?:\.[0-9]+)*", version)
        public = numeric.group(0) if numeric else version
        # find_package never exposes the INSTALLED version's major, only the requested
        # one, so it is baked in here where it is known.
        major = public.split(".")[0]
        # SameMajorVersion semantics, written out rather than hand-rolled from a single
        # VERSION_LESS. A lone "is the installed version at least the requested one" check accepts
        # two requests it must refuse: the upper end of a range is never consulted, so a consumer
        # asking for 2.14...<2.15 is handed 2.15, and a request from a different major is accepted,
        # so a consumer written for 1.x links a 2.x delegate. Both were measured against
        # write_basic_package_version_file(COMPATIBILITY SameMajorVersion), which refuses them.
        (cmake_dir / "torchtrt_executorch-config-version.cmake").write_text(
            "# Generated by setup.py. The version is only known when the wheel is built.\n"
            f'set(PACKAGE_VERSION "{public}")\n'
            f'set(TORCHTRT_EXECUTORCH_FULL_VERSION "{version}")\n'
            "\n"
            f'if(PACKAGE_FIND_VERSION_MAJOR STREQUAL "{major}")\n'
            "  if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)\n"
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            "  else()\n"
            "    set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
            "  endif()\n"
            "else()\n"
            "  set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            "endif()\n"
            "\n"
            "if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)\n"
            "  set(PACKAGE_VERSION_EXACT TRUE)\n"
            "endif()\n"
            "\n"
            "# A version RANGE, find_package(pkg 2.14...<2.15). Without this the upper bound is\n"
            "# silently ignored and the range behaves like its lower bound alone.\n"
            "if(PACKAGE_FIND_VERSION_RANGE)\n"
            "  if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MIN)\n"
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            '  elseif(PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "INCLUDE"\n'
            "         AND PACKAGE_VERSION VERSION_GREATER PACKAGE_FIND_VERSION_MAX)\n"
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            '  elseif(PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "EXCLUDE"\n'
            "         AND NOT PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MAX)\n"
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            "  endif()\n"
            "endif()\n",
            encoding="utf-8",
        )


TENSORRT_DISTRIBUTION = tensorrt_distribution()


class WheelTag(bdist_wheel):
    """Tag the wheel py3-none-<platform>, not cp3XX-cp3XX-<platform>.

    The payload is one ctypes-loaded shared library with no Python ABI, so it is byte for byte
    identical across CPython versions and only the platform matters. has_ext_modules keeps
    Root-Is-Purelib false and the platform tag; this drops the per-interpreter half of the tag
    so one built wheel serves every CPython instead of one identical copy per version.
    """

    def get_tag(self) -> tuple[str, str, str]:
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class PlatformDistribution(Distribution):
    """Marks the wheel platform-specific even though it declares no extension module.

    The delegate is a compiled object, x86-64 or aarch64, so a pure-Python tag would be
    wrong. This is what ``ext_modules`` used to provide.
    """

    def has_ext_modules(self) -> bool:
        return True


require_supported_cuda()
executorch_version = installed_version("executorch")
tensorrt_version = installed_version(TENSORRT_DISTRIBUTION)
cuda_runtime_version = installed_version(CUDA_RUNTIME_DISTRIBUTION)
setup(
    name="torch-tensorrt-executorch-runtime",
    version=torchtrt_version(),
    description="Torch-TensorRT delegate for the ExecuTorch Python runtime",
    packages=find_packages(),
    distclass=PlatformDistribution,
    package_data={
        "torch_tensorrt_executorch_runtime": [
            f"lib/{DELEGATE_LIBRARY}",
            "lib/cmake/torchtrt_executorch/*.cmake",
        ]
    },
    cmdclass={"build_py": BazelBuild, "bdist_wheel": WheelTag},
    python_requires=">=3.10",
    install_requires=[
        f"torch=={public_version(torch.__version__)}",
        f"executorch=={public_version(executorch_version)}",
        f"torch-tensorrt=={public_version(torchtrt_version())}",
        f"{TENSORRT_DISTRIBUTION}=={public_version(tensorrt_version)}",
        f"{CUDA_RUNTIME_DISTRIBUTION}=={public_version(cuda_runtime_version)}",
    ],
    zip_safe=False,
)
