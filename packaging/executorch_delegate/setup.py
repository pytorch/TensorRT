"""Build the precompiled Torch-TensorRT backend for ExecuTorch Python."""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import re
import subprocess
import sys

import torch
from torch.utils.cpp_extension import include_paths
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def torchtrt_version() -> str:
    if value := os.getenv("TORCH_TENSORRT_EXECUTORCH_DELEGATE_VERSION"):
        return value
    source = (REPO_ROOT / "py/torch_tensorrt/_version.py").read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', source, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not determine the Torch-TensorRT version")
    return match.group(1)


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if sys.platform != "linux":
            raise RuntimeError("The ExecuTorch TensorRT delegate supports Linux only")
        executorch_source = os.getenv("EXECUTORCH_SOURCE_DIR")
        if not executorch_source:
            raise RuntimeError(
                "Set EXECUTORCH_SOURCE_DIR to the pinned ExecuTorch source tree"
            )

        output = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        if output.exists():
            return
        build_dir = pathlib.Path(self.build_temp) / "executorch_delegate_native"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake",
            "-S",
            str(HERE / "native"),
            "-B",
            str(build_dir),
            f"-DEXECUTORCH_SOURCE_DIR={pathlib.Path(executorch_source).resolve()}",
            f"-DTORCH_TENSORRT_SOURCE_DIR={REPO_ROOT}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DTORCH_PRIMARY_INCLUDE_DIR={include_paths()[0]}",
            f"-DDELEGATE_LIBRARY_OUTPUT_DIRECTORY={output.parent}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        if value := os.getenv("TensorRT_ROOT"):
            configure.append(f"-DTensorRT_ROOT={value}")
        configure.extend(os.getenv("CMAKE_ARGS", "").split())
        subprocess.run(configure, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "torch_tensorrt_executorch_portable_lib",
                "--parallel",
                str(self.parallel or os.cpu_count() or 1),
            ],
            check=True,
        )
        library_stem = (
            "_portable_lib" if ext.name.endswith("._portable_lib") else "data_loader"
        )
        built = next(output.parent.glob(f"{library_stem}*.so"), None)
        if built is None:
            raise RuntimeError(
                f"CMake did not produce {library_stem} in {output.parent}"
            )
        if built != output:
            built.replace(output)


executorch_version = importlib.metadata.version("executorch")
setup(
    name="torch-tensorrt-executorch-delegate",
    version=torchtrt_version(),
    description="Torch-TensorRT delegate for the ExecuTorch Python runtime",
    packages=find_packages(),
    ext_modules=[
        CMakeExtension("torch_tensorrt_executorch_delegate._portable_lib"),
        CMakeExtension("torch_tensorrt_executorch_delegate.data_loader"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=[
        f"torch=={torch.__version__}",
        f"executorch=={executorch_version}",
    ],
    zip_safe=False,
)
