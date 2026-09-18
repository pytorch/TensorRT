# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the precompiled Torch-TensorRT backend for ExecuTorch Python."""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import uuid

import torch
import yaml
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
    """Read the ExecuTorch version the repository pins.

    Raises when the file is absent rather than returning an empty version, because the caller
    compares against whatever comes back and a build with no pin to compare against is exactly the
    case that comparison exists for.
    """
    pin_file = REPO_ROOT / "dev_dep_versions.yml"
    if not pin_file.is_file():
        raise RuntimeError(
            f"{pin_file} is missing, so the ExecuTorch pin cannot be checked. Build this wheel "
            "from a repository checkout."
        )
    return yaml.safe_load(pin_file.read_text(encoding="utf-8"))[
        "__executorch_version__"
    ]


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
    # The label matters as much as the version. Comparing only the public parts accepted a
    # processor-only build of the pinned date, which cannot supply the CUDA runtime the delegate
    # links, and the wheel this build then publishes requires the label it did not check.
    label = distribution.version.partition("+")[2]
    wrong_version = public_version(pinned) != installed
    wrong_build = not label.startswith("cu")
    if (wrong_version or wrong_build) and os.getenv(
        "TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH", ""
    ).lower() not in ("1", "true", "yes", "on"):
        raise RuntimeError(
            f"The installed ExecuTorch is {distribution.version} but dev_dep_versions.yml pins "
            f"{pinned} and the delegate needs a CUDA build. The delegate links this wheel's "
            "runtime and declares a dependency on it, so building against another version or "
            "another build ships a wheel that requires the wrong ExecuTorch. Install the pinned "
            "CUDA wheel, or set TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH=1 to build anyway."
        )
    return str(prefix)


def get_runtime_version() -> str:
    """Match the primary wheel's local-version convention with this wheel's base."""
    if version := os.getenv("TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION"):
        return version

    base_version = (HERE / "version.txt").read_text().strip()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        print("WARNING: Could not get git revision short hash, using default one")
        revision = "0000000"
    return f"{base_version}.dev0+{revision}"


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


def require_supported_cuda() -> None:
    """Require the CUDA major used by this delegate's dependencies and library paths."""
    cuda_version = torch.version.cuda
    if (cuda_version or "").split(".")[0] != "13":
        raise RuntimeError(
            "PyTorch built against CUDA 13 is required to build this wheel "
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

    def _generated_output_mapping(self) -> dict[str, str]:
        package = "torch_tensorrt_executorch_runtime"
        return {
            str(pathlib.Path(self.build_lib) / package / filename): str(
                pathlib.Path(self.get_package_dir(package)) / filename
            )
            for filename in (
                f"lib/{DELEGATE_LIBRARY}",
                "lib/cmake/torchtrt_executorch/torchtrt_executorch-config.cmake",
                "lib/cmake/torchtrt_executorch/torchtrt_executorch-config-version.cmake",
            )
        }

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *super().get_outputs(include_bytecode),
                    *self._generated_output_mapping(),
                ]
            )
        )

    def get_output_mapping(self) -> dict[str, str]:
        mapping = super().get_output_mapping()
        if self.editable_mode:
            mapping.update(self._generated_output_mapping())
        return mapping

    def run(self) -> None:
        # During an editable install setuptools routes a customized build_py through its own
        # _safely_run, which catches Exception and turns it into a warning pip hides, so a failed
        # native build reports "Successfully installed" with no delegate. SystemExit is not an
        # Exception, so it is the one thing that escapes. Everything this build raises has to become
        # one, including the RuntimeError from a missing bazel, which is the case that motivated
        # this: re-raising it unchanged left it inside the class setuptools swallows.
        try:
            self._build()
        except SystemExit:
            raise
        except BaseException as error:
            raise SystemExit(f"ExecuTorch delegate build failed: {error}") from error

    def _build(self) -> None:
        super().run()

        if sys.platform != "linux":
            raise RuntimeError("The ExecuTorch TensorRT delegate supports Linux only")

        bazel = shutil.which("bazelisk") or shutil.which("bazel")
        if bazel is None:
            raise RuntimeError("Could not find bazelisk or bazel in PATH")

        compilation_mode = (
            "dbg"
            if os.getenv("TORCH_TENSORRT_EXECUTORCH_DEBUG", "").lower()
            in ("1", "true", "yes", "on")
            else "opt"
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

        package = "torch_tensorrt_executorch_runtime"
        # Editable build_lib is temporary; generated package data must survive its removal.
        package_root = (
            pathlib.Path(self.get_package_dir(package))
            if self.editable_mode
            else pathlib.Path(self.build_lib) / package
        )
        output = package_root / "lib" / DELEGATE_LIBRARY
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
        if not self.editable_mode:
            # Shared objects only. runtime.py used to be removed here too, from when this package
            # stopped shipping a Python API, and it is shipped again now as the forwarder the
            # released main wheel imports by name. Removing it from the build output published a
            # wheel without it, so that import failed for anyone pairing an older main wheel with
            # this companion.
            for stale in package_root.glob("*.so*"):
                if stale.is_file():
                    stale.unlink()
        for stale in output.parent.glob("*.so*"):
            stale.unlink()
        # The build system leaves its output read only, and copy2 carries the mode across, so a
        # second build in the same tree used to fail with a permission error on its own previous
        # output. Removing the destination first covers that, and the copy is made writable so
        # anything downstream that rewrites it in place, such as the run path repair, still can.
        shutil.copy2(built, output)
        output.chmod(output.stat().st_mode | stat.S_IWUSR)
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
        version = self.distribution.get_version()
        # find_package compares dotted integers, so a dev suffix has to be dropped from the value
        # it reads while the full version stays visible for a human.
        numeric = re.match(r"[0-9]+(?:\.[0-9]+)*", version)
        public = numeric.group(0) if numeric else version
        # find_package never exposes the INSTALLED version's major, only the requested
        # one, so it is baked in here where it is known.
        major = public.split(".")[0]
        # SameMajorVersion bounds both endpoints, allowing only an exclusive next-major boundary.
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
            '  if(PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "INCLUDE"\n'
            f'     AND NOT PACKAGE_FIND_VERSION_MAX_MAJOR STREQUAL "{major}")\n'
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            '  elseif(PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "EXCLUDE"\n'
            f'         AND PACKAGE_FIND_VERSION_MAX VERSION_GREATER "{int(major) + 1}")\n'
            "    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            "  elseif(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MIN)\n"
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


TENSORRT_DISTRIBUTION = "tensorrt-cu13"


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


# These run while the file is read, and they have to. They produce install_requires, and dependency
# metadata is exactly what a metadata-only build asks for, so deriving pins from the build
# environment means that environment has to be present. The CUDA check leads, so an unsupported
# CUDA says so before anything else is attempted rather than failing later and less clearly.
require_supported_cuda()
executorch_version = installed_version("executorch")
tensorrt_version = installed_version(TENSORRT_DISTRIBUTION)
cuda_runtime_version = installed_version(CUDA_RUNTIME_DISTRIBUTION)
setup(
    name="torch-tensorrt-executorch-runtime",
    version=get_runtime_version(),
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
        # Full versions, local label included, for these three, but for two different reasons.
        # ExecuTorch is the one this delegate is compiled and linked against, so another build of it
        # is an ABI question. PyTorch and Torch-TensorRT are not linked here at all; they share a
        # process with a delegate that links one CUDA runtime and one TensorRT, so what matters is
        # that they come from the same CUDA row. A local label is the only part of a version that
        # names the row, and there is no wildcard for one, so it is an exact pin or nothing.
        f"torch=={torch.__version__}",
        f"executorch=={executorch_version}",
        f"torch-tensorrt=={installed_version('torch-tensorrt')}",
        f"{TENSORRT_DISTRIBUTION}=={public_version(tensorrt_version)}",
        f"{CUDA_RUNTIME_DISTRIBUTION}=={public_version(cuda_runtime_version)}",
    ],
    zip_safe=False,
)
