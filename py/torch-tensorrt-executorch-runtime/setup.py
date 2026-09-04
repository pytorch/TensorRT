"""Build the precompiled Torch-TensorRT backend for ExecuTorch Python."""

from __future__ import annotations

import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys
import uuid

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BAZEL_TARGET = "//py/torch-tensorrt-executorch-runtime/native:delegate_native"
BUILD_NONCE = os.getenv("TORCH_TENSORRT_EXECUTORCH_BUILD_NONCE", uuid.uuid4().hex)

# CI ships this as a companion of one specific torch-tensorrt wheel.  Give it
# that wheel's version so nightly/index uploads are immutable per build and
# pip selects the compatible companion naturally.  The checked-in value keeps
# standalone development builds deterministic.
RUNTIME_VERSION = os.getenv(
    "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION",
    (HERE / "version.txt").read_text().strip(),
)

TORCH_REQUIREMENT = "torch>=2.14.0,<2.15.0"
EXECUTORCH_REQUIREMENT = "executorch==1.4.1"
TORCH_TENSORRT_REQUIREMENT = "torch-tensorrt>=2.14.0,<2.15.0"


def get_tensorrt_requirement() -> str:
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError(
            "CUDA enabled PyTorch is required to build this wheel found None"
        )
    if cuda_version.startswith("12."):
        return "tensorrt-cu12>=11.1.0,<11.2"
    if cuda_version.startswith("13."):
        return "tensorrt-cu13>=11.1.0,<11.2"
    raise RuntimeError(f"Unsupported CUDA version: {cuda_version}")


def cuda_runtime_distribution() -> str:
    """Return the CUDA runtime distribution matching the PyTorch CUDA build.

    NVIDIA splits this one by major: the CUDA 12 wheels are published as
    ``nvidia-cuda-runtime-cu12``, while the unsuffixed ``nvidia-cuda-runtime``
    is the CUDA 13 line. Naming the unsuffixed one unconditionally made every
    CUDA 12 row fail with "No package metadata was found for
    nvidia-cuda-runtime", because what torch installed there was the suffixed
    distribution.
    """
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("CUDA-enabled PyTorch is required to build this wheel")
    if cuda_version.startswith("12."):
        return "nvidia-cuda-runtime-cu12"
    if cuda_version.startswith("13."):
        return "nvidia-cuda-runtime"
    raise RuntimeError(f"Unsupported CUDA version: {cuda_version}")


class BazelExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class BazelBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if sys.platform != "linux":
            raise RuntimeError("The ExecuTorch TensorRT delegate supports Linux only")

        output = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        bazel = shutil.which("bazelisk") or shutil.which("bazel")
        if bazel is None:
            raise RuntimeError("Could not find bazelisk or bazel in PATH")

        compilation_mode = "dbg" if self.debug else "opt"
        command = [
            bazel,
            "build",
            BAZEL_TARGET,
            "--config=linux",
            "--config=python",
            f"--compilation_mode={compilation_mode}",
            f"--action_env=PYTHON_BIN_PATH={sys.executable}",
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
        library_stem = (
            "_portable_lib" if ext.name.endswith("._portable_lib") else "data_loader"
        )
        built = (
            bazel_bin
            / "py/torch-tensorrt-executorch-runtime/native/delegate_native/lib"
            / f"{library_stem}.so"
        )
        if not built.is_file():
            raise RuntimeError(f"Bazel did not produce {built}")
        output.unlink(missing_ok=True)
        shutil.copy2(built, output)

        # Ship the two shared libraries the extensions load. ExecuTorch's published
        # wheel provides neither, so without them importing the runtime fails on a
        # missing shared object. Copied rather than declared as extensions because they
        # are plain dependencies, not Python modules.
        for dependency in ("libextension_cuda.so", "libaoti_cuda_shims.so"):
            source = built.parent / dependency
            if not source.is_file():
                raise RuntimeError(f"Bazel did not produce {source}")
            destination = output.parent / dependency
            destination.unlink(missing_ok=True)
            shutil.copy2(source, destination)


setup(
    name="torch-tensorrt-executorch-runtime",
    version=RUNTIME_VERSION,
    description="Torch-TensorRT delegate for the ExecuTorch Python runtime",
    packages=find_packages(),
    ext_modules=[
        BazelExtension("torch_tensorrt_executorch_runtime._portable_lib"),
        BazelExtension("torch_tensorrt_executorch_runtime.data_loader"),
    ],
    cmdclass={"build_ext": BazelBuild},
    python_requires=">=3.10",
    install_requires=[
        TORCH_REQUIREMENT,
        EXECUTORCH_REQUIREMENT,
        TORCH_TENSORRT_REQUIREMENT,
        get_tensorrt_requirement(),
    ],
    zip_safe=False,
)
