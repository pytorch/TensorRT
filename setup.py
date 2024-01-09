import glob
import os
import platform
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from distutils.cmd import Command
from pathlib import Path
from shutil import copyfile, rmtree
from typing import List

import setuptools
import yaml
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install
from torch.utils import cpp_extension
from wheel.bdist_wheel import bdist_wheel

__version__: str = "0.0.0"
__cuda_version__: str = "0.0"
__cudnn_version__: str = "0.0"
__tensorrt_version__: str = "0.0"

LEGACY_BASE_VERSION_SUFFIX_PATTERN = re.compile("a0$")


def get_root_dir() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def linux_path_to_windows(path: str) -> str:
    return os.path.join(*path.split("/"))


def get_base_version() -> str:
    root = get_root_dir()
    try:
        dirty_version = open(root / "version.txt", "r").read().strip()
    except FileNotFoundError:
        print("# WARNING: Base version not found defaulting BUILD_VERSION to 0.1.0")
        dirty_version = "0.1.0"
    # Strips trailing a0 from version.txt, not too sure why it's there in the
    # first place
    return re.sub(LEGACY_BASE_VERSION_SUFFIX_PATTERN, "", dirty_version)


def load_dep_info():
    global __cuda_version__
    global __cudnn_version__
    global __tensorrt_version__
    with open("dev_dep_versions.yml", "r") as stream:
        versions = yaml.safe_load(stream)
        if (gpu_arch_version := os.environ.get("CU_VERSION")) is not None:
            __cuda_version__ = (
                (gpu_arch_version[2:])[:-1] + "." + (gpu_arch_version[2:])[-1:]
            )
        else:
            __cuda_version__ = versions["__cuda_version__"]
        __cudnn_version__ = versions["__cudnn_version__"]
        __tensorrt_version__ = versions["__tensorrt_version__"]


load_dep_info()

dir_path = linux_path_to_windows(str(get_root_dir()) + "/py")

RELEASE = False
CI_BUILD = False


if "--release" in sys.argv:
    RELEASE = True
    sys.argv.remove("--release")

if (release_env_var := os.environ.get("RELEASE")) is not None:
    if release_env_var == "1":
        RELEASE = True

if (gpu_arch_version := os.environ.get("CU_VERSION")) is None:
    gpu_arch_version = f"cu{__cuda_version__.replace('.','')}"


if RELEASE:
    __version__ = os.environ.get("BUILD_VERSION")
else:
    __version__ = f"{get_base_version()}.dev0+{get_git_revision_short_hash()}"

if "--ci" in sys.argv:
    sys.argv.remove("--ci")
    if RELEASE:
        CI_BUILD = True

if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
    if ci_env_var == "1":
        CI_BUILD = True


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def gen_version_file():
    version_path = os.path.join(dir_path, "torch_tensorrt", "_version.py")

    with open(version_path, "w+") as f:
        print("creating version file")
        f.write('__version__ = "' + __version__ + '"\n')
        f.write('__cuda_version__ = "' + __cuda_version__ + '"\n')
        f.write('__cudnn_version__ = "' + __cudnn_version__ + '"\n')
        f.write('__tensorrt_version__ = "' + __tensorrt_version__ + '"\n')


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        gen_version_file()
        develop.run(self)


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        gen_version_file()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        gen_version_file()
        bdist_wheel.run(self)


class EditableWheelCommand(editable_wheel):
    description = "Builds the package in development mode"

    def initialize_options(self):
        editable_wheel.initialize_options(self)

    def finalize_options(self):
        editable_wheel.finalize_options(self)

    def run(self):
        gen_version_file()
        editable_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    PY_CLEAN_DIRS = [
        linux_path_to_windows("./build"),
        linux_path_to_windows("./dist"),
        linux_path_to_windows("./torch_tensorrt/__pycache__"),
        linux_path_to_windows("./torch_tensorrt/lib"),
        linux_path_to_windows("./torch_tensorrt/include"),
        linux_path_to_windows("./torch_tensorrt/bin"),
        linux_path_to_windows("./*.pyc"),
        linux_path_to_windows("./*.tgz"),
        linux_path_to_windows("./*.egg-info"),
    ]
    PY_CLEAN_FILES = [
        linux_path_to_windows("./torch_tensorrt/*.so"),
        linux_path_to_windows("./torch_tensorrt/_version.py"),
        linux_path_to_windows("./torch_tensorrt/BUILD"),
        linux_path_to_windows("./torch_tensorrt/WORKSPACE"),
        linux_path_to_windows("./torch_tensorrt/LICENSE"),
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass
        # for path_spec in self.PY_CLEAN_DIRS:
        #     # Make paths absolute and relative to this path
        #     abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
        #     for path in [str(p) for p in abs_paths]:
        #         if not path.startswith(dir_path):
        #             # Die if path in CLEAN_FILES is absolute + outside this directory
        #             raise ValueError("%s is not a path inside %s" % (path, dir_path))
        #         print("Removing %s" % os.path.relpath(path))
        #         rmtree(path)

        # for path_spec in self.PY_CLEAN_FILES:
        #     # Make paths absolute and relative to this path
        #     abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
        #     for path in [str(p) for p in abs_paths]:
        #         if not path.startswith(dir_path):
        #             # Die if path in CLEAN_FILES is absolute + outside this directory
        #             raise ValueError("%s is not a path inside %s" % (path, dir_path))
        #         print("Removing %s" % os.path.relpath(path))
        #         os.remove(path)


ext_modules = []

packages = [
    "torch_tensorrt",
    # "torch_tensorrt.dynamo",
    # "torch_tensorrt.dynamo.backend",
    # "torch_tensorrt.dynamo.conversion",
    # "torch_tensorrt.dynamo.conversion.impl",
    # "torch_tensorrt.dynamo.conversion.impl.activation",
    # "torch_tensorrt.dynamo.conversion.impl.condition",
    # "torch_tensorrt.dynamo.conversion.impl.elementwise",
    # "torch_tensorrt.dynamo.conversion.impl.normalization",
    # "torch_tensorrt.dynamo.conversion.impl.slice",
    # "torch_tensorrt.dynamo.conversion.impl.unary",
    # "torch_tensorrt.dynamo.lowering",
    # "torch_tensorrt.dynamo.lowering.passes",
    # "torch_tensorrt.dynamo.partitioning",
    # "torch_tensorrt.dynamo.runtime",
    # "torch_tensorrt.dynamo.tools",
    # "torch_tensorrt.fx",
    # "torch_tensorrt.fx.converters",
    # "torch_tensorrt.fx.converters.impl",
    # "torch_tensorrt.fx.passes",
    # "torch_tensorrt.fx.tools",
    # "torch_tensorrt.fx.tracer",
    # "torch_tensorrt.fx.tracer.acc_tracer",
    # "torch_tensorrt.fx.tracer.dispatch_tracer",
    # "torch_tensorrt.runtime",
]

package_dir = {
    "torch_tensorrt": linux_path_to_windows("py/torch_tensorrt"),
    # "torch_tensorrt.dynamo": linux_path_to_windows("py/torch_tensorrt/dynamo"),
    # "torch_tensorrt.dynamo.backend": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/backend"
    # ),
    # "torch_tensorrt.dynamo.conversion": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.activation": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/activation"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.condition": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/condition"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.elementwise": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/elementwise"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.normalization": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/normalization"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.slice": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/slice"
    # ),
    # "torch_tensorrt.dynamo.conversion.impl.unary": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/conversion/impl/unary"
    # ),
    # "torch_tensorrt.dynamo.lowering": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/lowering"
    # ),
    # "torch_tensorrt.dynamo.lowering.passes": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/lowering/passes"
    # ),
    # "torch_tensorrt.dynamo.partitioning": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/partitioning"
    # ),
    # "torch_tensorrt.dynamo.runtime": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/runtime"
    # ),
    # "torch_tensorrt.dynamo.tools": linux_path_to_windows(
    #     "py/torch_tensorrt/dynamo/tools"
    # ),
    # "torch_tensorrt.fx": linux_path_to_windows("py/torch_tensorrt/fx"),
    # "torch_tensorrt.fx.converters": linux_path_to_windows(
    #     "py/torch_tensorrt/fx/converters"
    # ),
    # "torch_tensorrt.fx.converters.impl": linux_path_to_windows(
    #     "py/torch_tensorrt/fx/converters/impl"
    # ),
    # "torch_tensorrt.fx.passes": linux_path_to_windows("py/torch_tensorrt/fx/passes"),
    # "torch_tensorrt.fx.tools": linux_path_to_windows("py/torch_tensorrt/fx/tools"),
    # "torch_tensorrt.fx.tracer": linux_path_to_windows("py/torch_tensorrt/fx/tracer"),
    # "torch_tensorrt.fx.tracer.acc_tracer": linux_path_to_windows(
    #     "py/torch_tensorrt/fx/tracer/acc_tracer"
    # ),
    # "torch_tensorrt.fx.tracer.dispatch_tracer": linux_path_to_windows(
    #     "py/torch_tensorrt/fx/tracer/dispatch_tracer"
    # ),
    # "torch_tensorrt.runtime": linux_path_to_windows("py/torch_tensorrt/runtime"),
}

package_data = {}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch_tensorrt",
    ext_modules=ext_modules,
    version=__version__,
    cmdclass={
        "install": InstallCommand,
        "clean": CleanCommand,
        "develop": DevelopCommand,
        "build_ext": cpp_extension.BuildExtension,
        "bdist_wheel": BdistCommand,
        "editable_wheel": EditableWheelCommand,
    },
    zip_safe=False,
    packages=packages,
    package_dir=package_dir,
    include_package_data=False,
    package_data=package_data,
    exclude_package_data={
        "": [
            linux_path_to_windows("py/torch_tensorrt/csrc/*.cpp"),
            linux_path_to_windows("py/torch_tensorrt/fx/test*"),
            linux_path_to_windows("torch_tensorrt/csrc/*.cpp"),
            linux_path_to_windows("torch_tensorrt/fx/test*"),
            linux_path_to_windows("test*"),
            linux_path_to_windows("*.cpp"),
        ],
        "torch_tensorrt": [
            linux_path_to_windows("py/torch_tensorrt/csrc/*.cpp"),
            linux_path_to_windows("py/torch_tensorrt/fx/test*"),
            linux_path_to_windows("torch_tensorrt/csrc/*.cpp"),
            linux_path_to_windows("torch_tensorrt/fx/test*"),
            linux_path_to_windows("test*"),
            linux_path_to_windows("*.cpp"),
        ],
        # "torch_tensorrt.dynamo": [linux_path_to_windows("test/*.py")],
        # "torch_tensorrt.fx": [linux_path_to_windows("test/*.py")],
    },
)
