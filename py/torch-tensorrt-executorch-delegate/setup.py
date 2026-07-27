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
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BAZEL_TARGET = "//py/torch-tensorrt-executorch-delegate/native:delegate_native"
BUILD_NONCE = os.getenv("TORCH_TENSORRT_EXECUTORCH_BUILD_NONCE", uuid.uuid4().hex)


def torchtrt_version() -> str:
    if value := os.getenv("TORCH_TENSORRT_EXECUTORCH_DELEGATE_VERSION"):
        return value
    version_py = REPO_ROOT / "py/torch_tensorrt/_version.py"
    if version_py.exists():
        if m := re.search(r'__version__\s*=\s*["\']([^"\']+)', version_py.read_text()):
            return m.group(1)
    return (REPO_ROOT / "version.txt").read_text().strip()


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
            / "py/torch-tensorrt-executorch-delegate/native/delegate_native/lib"
            / f"{library_stem}.so"
        )
        if not built.is_file():
            raise RuntimeError(f"Bazel did not produce {built}")
        output.unlink(missing_ok=True)
        shutil.copy2(built, output)


executorch_version = importlib.metadata.version("executorch")
setup(
    name="torch-tensorrt-executorch-delegate",
    version=torchtrt_version(),
    description="Torch-TensorRT delegate for the ExecuTorch Python runtime",
    packages=find_packages(),
    ext_modules=[
        BazelExtension("torch_tensorrt_executorch_delegate._portable_lib"),
        BazelExtension("torch_tensorrt_executorch_delegate.data_loader"),
    ],
    cmdclass={"build_ext": BazelBuild},
    python_requires=">=3.10",
    install_requires=[
        f"torch=={torch.__version__}",
        f"executorch=={executorch_version}",
        f"torch-tensorrt=={torchtrt_version()}",
    ],
    zip_safe=False,
)
