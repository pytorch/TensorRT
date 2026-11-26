# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

import atexit
import glob
import os
import platform
import re
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path
from shutil import copyfile, rmtree, which

import torch
import yaml
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install
from torch.utils.cpp_extension import IS_WINDOWS

__version__: str = "0.0.0"
__cuda_version__: str = "0.0"
__tensorrt_version__: str = "0.0"
__tensorrt_rtx_version__: str = "0.0"
__tensorrt_llm_version__: str = "0.0"

LEGACY_BASE_VERSION_SUFFIX_PATTERN = re.compile("a0$")
# CI_PIPELINE_ID is the environment variable set by DLFW ci build
IS_DLFW_CI = os.environ.get("CI_PIPELINE_ID") is not None


def get_root_dir() -> Path:
    return Path(__file__).parent.absolute()


def get_git_revision_short_hash() -> str:
    # DLFW ci build does not have git
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        print("WARNING: Could not get git revision short hash, using default one")
        # in release/ngc/25.10 branch this is the commit hash of the pytorch commit that is used for dlfw package
        return "0000000"


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
    global __tensorrt_version__
    global __tensorrt_rtx_version__
    global __tensorrt_llm_version__
    global __executorch_version__
    with open("dev_dep_versions.yml", "r") as stream:
        versions = yaml.safe_load(stream)
        if (gpu_arch_version := os.environ.get("CU_VERSION")) is not None:
            __cuda_version__ = (
                (gpu_arch_version[2:])[:-1] + "." + (gpu_arch_version[2:])[-1:]
            )
        else:
            __cuda_version__ = versions["__cuda_version__"]
        __tensorrt_version__ = versions["__tensorrt_version__"]
        __tensorrt_rtx_version__ = versions["__tensorrt_rtx_version__"]
        __tensorrt_llm_version__ = versions["__tensorrt_llm_version__"]
        __executorch_version__ = versions["__executorch_version__"]


load_dep_info()

dir_path = os.path.join(str(get_root_dir()), "py")

MACHINE_ARCH = platform.machine().lower()
NATIVE_WINDOWS_ARM64 = IS_WINDOWS and MACHINE_ARCH in ("arm64", "aarch64")
IS_AARCH64 = not IS_WINDOWS and MACHINE_ARCH == "aarch64"
WINDOWS_ON_ARM = "--windows-on-arm" in sys.argv
if WINDOWS_ON_ARM:
    # setuptools does not know this project-specific option, so consume it
    # before command-line processing. The environment variable remains
    # supported for CI and scripted builds.
    sys.argv.remove("--windows-on-arm")
FORCED_WINDOWS_ARM64 = (
    WINDOWS_ON_ARM
    or os.environ.get("TORCHTRT_TARGET_PLATFORM", "").lower() == "windows-arm64"
)
if FORCED_WINDOWS_ARM64 and not IS_WINDOWS:
    raise RuntimeError("Windows ARM64 builds are supported only on Windows")
TARGET_WINDOWS_ARM64 = NATIVE_WINDOWS_ARM64 or FORCED_WINDOWS_ARM64
# An ARM64 Python process builds natively. An x64 Python process must opt in to
# the ARM64 target via --windows-on-arm or TORCHTRT_TARGET_PLATFORM.
WINDOWS_CROSS_COMPILE = TARGET_WINDOWS_ARM64 and not NATIVE_WINDOWS_ARM64
IS_JETPACK = False

PY_ONLY = False
LEGACY = False
RELEASE = False
CI_BUILD = False
USE_TRT_RTX = False


if "--use-rtx" in sys.argv:
    USE_TRT_RTX = True
    sys.argv.remove("--use-rtx")

if "--py-only" in sys.argv:
    PY_ONLY = True
    sys.argv.remove("--py-only")

if "--legacy" in sys.argv:
    LEGACY = True
    sys.argv.remove("--legacy")

if "--release" in sys.argv:
    RELEASE = True
    sys.argv.remove("--release")

if "--dlfw" in sys.argv:
    IS_DLFW_CI = True
    sys.argv.remove("--dlfw")

if (py_only_env_var := os.environ.get("PYTHON_ONLY")) is not None:
    if py_only_env_var == "1":
        PY_ONLY = True

if (use_rtx_env_var := os.environ.get("USE_TRT_RTX")) is not None:
    if use_rtx_env_var == "1" or use_rtx_env_var.lower() == "true":
        USE_TRT_RTX = True

# Distribution name: keep import package as `torch_tensorrt`, but vary project
# name so wheels for RTX vs standard TensorRT are distinct.
PROJECT_NAME = "torch_tensorrt_rtx" if USE_TRT_RTX else "torch_tensorrt"
#
# Ensure METADATA Name matches RTX vs standard build even with static pyproject.
# We temporarily rewrite [project].name in pyproject.toml for the current build.
#
_pyproject_path = get_root_dir() / "pyproject.toml"
_pyproject_backup = None
if USE_TRT_RTX:
    try:
        _pyproject_backup = _pyproject_path.read_text(encoding="utf-8")
        updated = []
        for line in _pyproject_backup.splitlines(keepends=True):
            if line.strip().startswith("name = ") and '"torch_tensorrt"' in line:
                updated.append('name = "torch-tensorrt-rtx"\n')
            else:
                updated.append(line)
        _new_content = "".join(updated)
        if _new_content != _pyproject_backup:
            _pyproject_path.write_text(_new_content, encoding="utf-8")
            atexit.register(
                lambda: _pyproject_path.write_text(_pyproject_backup, encoding="utf-8")
            )
    except Exception:
        # Non-fatal; filename will still distinguish variants.
        pass
if (release_env_var := os.environ.get("RELEASE")) is not None:
    if release_env_var == "1":
        RELEASE = True

if (gpu_arch_version := os.environ.get("CU_VERSION")) is None:
    gpu_arch_version = f"cu{__cuda_version__.replace('.','')}"

if IS_AARCH64 and (jetpack := os.environ.get("JETPACK_BUILD")) is not None:
    if jetpack == "1":
        IS_JETPACK = True

if RELEASE:
    __version__ = os.environ.get("BUILD_VERSION")
else:
    __version__ = f"{get_base_version()}.dev0+{get_git_revision_short_hash()}"

# The delegate is compiled from the ExecuTorch source revision pinned in MODULE.bazel, so the
# installed wheel should agree with it. The upper bound is the load-bearing half: ExecuTorch's C++
# runtime API is not stable across minor releases, and an unbounded floor would resolve a future
# minor against a backend built for this one. Everything below that ceiling resolves: later
# nightlies of the pinned minor, its release candidate, and its patch releases alike, since the
# exact pin belongs in the runtime package, which does derive it. Where a wheel's extras also
# require that package, they admit only the one ExecuTorch build it pins.
# The floor currently names a dev build, because the runtime split the delegate needs does not
# exist in any ExecuTorch release yet: 1.4.1's executorch/lib carries no standalone linkable
# runtime, and no CUDA wheel at all.
# That also makes this range prefer a release as soon as one exists, since a release sorts above
# every development build of the same version, so nothing here changes on the day it ships.
_executorch_major, _executorch_minor = __executorch_version__.split(".")[:2]
# ExecuTorch CUDA wheels support Linux and Windows x86-64. Windows ARM64 is
# excluded until ExecuTorch publishes a matching runtime wheel.
EXECUTORCH_REQUIREMENT = (
    f"executorch>={__executorch_version__},"
    f"<{_executorch_major}.{int(_executorch_minor) + 1}; "
    "(platform_system == 'Linux' or "
    "(platform_system == 'Windows' and platform_machine == 'AMD64'))"
)
EXTRAS_REQUIRE = {
    "executorch": [EXECUTORCH_REQUIREMENT],
    "all": [EXECUTORCH_REQUIREMENT],
}
# A published wheel's extras also require the exact companion its build job produced, but only
# where the wheel's own index carries the ExecuTorch build that companion pins: pip does not fail on
# an extra it cannot meet, it quietly installs another torch-tensorrt. A development build is only
# on the nightly index. A release is on the test index, and on Linux PyPI carries its CUDA build for
# the CUDA version of PyTorch's PyPI wheel, the one the PyPI companion is built for. A Windows PyPI
# wheel records no channel, because PyPI's Windows ExecuTorch is the CPU build. After promotion,
# the extra stays installable only if ExecuTorch and the companion reach the stable index before
# torch-tensorrt does. The workflow names the version and the channel before building either
# wheel, and the companion build reads the same version. Source builds leave both out, so the lock
# keeps following the source pins.
EXECUTORCH_CHANNEL = "nightly" if ".dev" in __executorch_version__ else "test"
EXECUTORCH_RUNTIME_VERSION = (
    os.environ.get("TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION", "") if RELEASE else ""
)
EXECUTORCH_RUNTIME_CHANNEL = (
    os.environ.get("TORCH_TENSORRT_EXECUTORCH_RUNTIME_CHANNEL", "")
    if EXECUTORCH_RUNTIME_VERSION
    else ""
)
if EXECUTORCH_RUNTIME_CHANNEL == EXECUTORCH_CHANNEL or (
    EXECUTORCH_RUNTIME_CHANNEL == "pypi" and EXECUTORCH_CHANNEL == "test"
):
    for _extra in ("executorch", "all"):
        EXTRAS_REQUIRE[_extra].append(
            f"torch-tensorrt-executorch-runtime=={EXECUTORCH_RUNTIME_VERSION}; "
            + EXECUTORCH_REQUIREMENT.partition(";")[2].strip()
        )

if "--ci" in sys.argv:
    sys.argv.remove("--ci")
    if RELEASE:
        CI_BUILD = True

if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
    if ci_env_var == "1":
        CI_BUILD = True

if IS_AARCH64:
    if "--jetpack" in sys.argv:
        sys.argv.remove("--jetpack")
        # this is used to simulate the jetpack build on aarch64 machines(non-tegra platforms)
        IS_JETPACK = True
    else:
        IS_SBSA = True

IS_SBSA = True if IS_AARCH64 and not IS_JETPACK else False

BAZEL_EXE = None
if not PY_ONLY:
    BAZEL_EXE = which("bazelisk")

    if BAZEL_EXE is None:
        BAZEL_EXE = which("bazel")
        if BAZEL_EXE is None:
            sys.exit("Could not find bazel in PATH")


def build_libtorchtrt_cxx11_abi(develop=True):
    cmd = [BAZEL_EXE, "build", "//:libtorchtrt"]

    if develop:
        cmd.append("--compilation_mode=dbg")
    else:
        cmd.append("--compilation_mode=opt")
    if not IS_WINDOWS:
        if IS_AARCH64:
            cmd.append("--distdir=third_party/dist_dir/aarch64-linux-gnu")
        else:
            cmd.append("--distdir=third_party/dist_dir/x86_64-linux-gnu")

    cmd.append("--config=python")

    if IS_WINDOWS:
        cmd.append("--config=windows")
        if TARGET_WINDOWS_ARM64:
            cmd.append("--platforms=//toolchains:windows_arm64")
        if WINDOWS_CROSS_COMPILE:
            cmd.append(
                "--extra_toolchains=@local_config_cc//:cc-toolchain-arm64_windows"
            )
    else:
        cmd.append("--config=linux")

    if USE_TRT_RTX:
        cmd.append("--config=rtx")
        print("TensorRT RTX build")

    if IS_DLFW_CI:
        cmd.append("--config=dlfw")
        print("DLFW build")

    if IS_JETPACK:
        cmd.append("--config=jetpack")
        print("Jetpack build")

    if IS_SBSA:
        print("SBSA build")

    if CI_BUILD:
        print("CI based build")
        if IS_AARCH64:
            cmd.append("--platforms=//toolchains:aarch64_linux")
        else:
            cmd.append("--platforms=//toolchains:ci_rhel_x86_64_linux")

    # Persist bazel's action cache across builds when a cache dir is provided.
    # The C++/CUDA compile is the serial long pole; with a warm --disk_cache an
    # unchanged (or partially-changed) tree reuses cached action outputs, turning
    # a from-scratch rebuild into seconds. CI sets BAZEL_DISK_CACHE and restores/
    # saves the dir via actions/cache (coarse key); no-op locally unless opted in.
    disk_cache = os.environ.get("BAZEL_DISK_CACHE")
    if disk_cache:
        cmd.append(f"--disk_cache={disk_cache}")
        print(f"Using bazel --disk_cache={disk_cache}")

    env = os.environ.copy()
    target_torch_path = env.get("TORCHTRT_TARGET_TORCH_ROOT")
    if TARGET_WINDOWS_ARM64 and target_torch_path:
        env["TORCH_PATH"] = target_torch_path
        print(f"Using target TORCH_PATH={target_torch_path}")
    elif "TORCH_PATH" not in env:
        stable_torch_path = resolve_torch_path()
        if stable_torch_path is not None:
            env["TORCH_PATH"] = stable_torch_path
            print(f"Using TORCH_PATH={stable_torch_path}")

    print(f"building libtorchtrt {cmd=}")
    status_code = subprocess.run(cmd, env=env).returncode

    if status_code != 0:
        sys.exit(status_code)


def resolve_torch_path():
    explicit_torch_path = os.environ.get("TORCH_PATH")
    if explicit_torch_path:
        return explicit_torch_path

    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for env_var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(env_var)
        if not prefix:
            continue

        candidate = os.path.join(prefix, "lib", version_dir, "site-packages", "torch")
        if os.path.isdir(os.path.join(candidate, "include", "c10")) and os.path.isdir(
            os.path.join(candidate, "lib")
        ):
            return candidate

    torch_path = os.path.dirname(torch.__file__)
    if os.path.isdir(os.path.join(torch_path, "include", "c10")) and os.path.isdir(
        os.path.join(torch_path, "lib")
    ):
        return torch_path

    return None


def gen_version_file():
    if not (IS_WINDOWS or os.path.exists(dir_path + "/torch_tensorrt/_version.py")):
        os.mknod(dir_path + "/torch_tensorrt/_version.py")

    with open(dir_path + "/torch_tensorrt/_version.py", "w") as f:
        print("creating version file")
        f.write(
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
            "AFFILIATES. All rights reserved.\n"
        )
        f.write("# SPDX-License-Identifier: BSD-3-Clause\n\n")
        f.write('__version__ = "' + __version__ + '"\n')
        f.write('__cuda_version__ = "' + __cuda_version__ + '"\n')
        f.write('__tensorrt_version__ = "' + __tensorrt_version__ + '"\n')
        f.write('__tensorrt_rtx_version__ = "' + __tensorrt_rtx_version__ + '"\n')
        f.write('__tensorrt_llm_version__ = "' + __tensorrt_llm_version__ + '"\n')
        f.write(
            '__executorch_runtime_version__ = "' + EXECUTORCH_RUNTIME_VERSION + '"\n'
        )
        f.write(
            '__executorch_runtime_channel__ = "' + EXECUTORCH_RUNTIME_CHANNEL + '"\n'
        )
        f.write('__executorch_channel__ = "' + EXECUTORCH_CHANNEL + '"\n')


def copy_libtorchtrt():
    package_path = Path(dir_path) / "torch_tensorrt"
    (package_path / "lib").mkdir(parents=True, exist_ok=True)
    bazel_bin = get_root_dir() / "bazel-bin"

    print("copying library into module")
    if IS_WINDOWS:
        copyfile(bazel_bin / "cpp/lib/torchtrt.dll", package_path / "lib/torchtrt.dll")
        copyfile(
            bazel_bin / "cpp/lib/torchtrt.dll.if.lib", package_path / "lib/torchtrt.lib"
        )
    else:
        subprocess.run(
            [
                "tar",
                "-xzf",
                str(bazel_bin / "libtorchtrt.tar.gz"),
                "--exclude=torch_tensorrt/src",
                "--exclude=torch_tensorrt/src/*",
                "--exclude=torch_tensorrt/examples",
                "--exclude=torch_tensorrt/examples/*",
                "--strip-components=1",
                "-C",
                str(package_path),
            ],
            check=True,
        )


def prepare_build(develop=False):
    if not PY_ONLY:
        build_libtorchtrt_cxx11_abi(develop=develop)
        copy_libtorchtrt()
    gen_version_file()


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        prepare_build(develop=True)
        super().run()


class InstallCommand(install):
    description = "Builds the package"

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        prepare_build(develop=False)
        super().run()


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def finalize_options(self):
        super().finalize_options()
        if TARGET_WINDOWS_ARM64:
            # Cross-builds run under x64 Python, so the target platform cannot be
            # inferred from the build interpreter.
            self.plat_name = "win_arm64"
        self.root_is_pure = False

    def get_tag(self):
        python_tag, abi_tag, platform_tag = super().get_tag()
        if TARGET_WINDOWS_ARM64:
            platform_tag = "win_arm64"
        return python_tag, abi_tag, platform_tag

    def run(self):
        # Ensure wheel metadata/project name reflects RTX vs standard build,
        # even when pyproject.toml provides static [project].name.
        try:
            self.distribution.metadata.name = PROJECT_NAME
        except Exception:
            pass
        prepare_build(develop=False)
        super().run()


class EditableWheelCommand(editable_wheel):
    description = "Builds the package in development mode"

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        prepare_build(develop=True)
        super().run()


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    PY_CLEAN_DIRS = [
        os.path.join(".", "build"),
        os.path.join(".", "dist"),
        os.path.join(".", "torch_tensorrt", "__pycache__"),
        os.path.join(".", "torch_tensorrt", "lib"),
        os.path.join(".", "torch_tensorrt", "include"),
        os.path.join(".", "torch_tensorrt", "bin"),
        os.path.join(".", "*.pyc"),
        os.path.join(".", "*.tgz"),
        os.path.join(".", "*.egg-info"),
    ]
    PY_CLEAN_FILES = [
        os.path.join(".", "torch_tensorrt", "*.so"),
        os.path.join(".", "torch_tensorrt", "_version.py"),
        os.path.join(".", "torch_tensorrt", "BUILD"),
        os.path.join(".", "torch_tensorrt", "WORKSPACE"),
        os.path.join(".", "torch_tensorrt", "LICENSE"),
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_DIRS:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print("Removing %s" % os.path.relpath(path))
                rmtree(path)

        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print("Removing %s" % os.path.relpath(path))
                os.remove(path)


dynamo_packages = [
    "torch_tensorrt",
    "torch_tensorrt.distributed",
    "torch_tensorrt.distributed.run",
    "torch_tensorrt.kernels",
    "torch_tensorrt.reporting",
    "torch_tensorrt.verification",
    "torch_tensorrt.dynamo",
    "torch_tensorrt.dynamo.backend",
    "torch_tensorrt.dynamo.conversion",
    "torch_tensorrt.dynamo.conversion.impl",
    "torch_tensorrt.dynamo.conversion.impl.activation",
    "torch_tensorrt.dynamo.conversion.impl.condition",
    "torch_tensorrt.dynamo.conversion.impl.elementwise",
    "torch_tensorrt.dynamo.conversion.impl.normalization",
    "torch_tensorrt.dynamo.conversion.impl.slice",
    "torch_tensorrt.dynamo.conversion.impl.unary",
    "torch_tensorrt.dynamo.conversion.plugins",
    "torch_tensorrt.dynamo.debug",
    "torch_tensorrt.dynamo.lowering",
    "torch_tensorrt.dynamo.lowering.passes",
    "torch_tensorrt.dynamo.lowering.constant_fold_exclusions",
    "torch_tensorrt.dynamo.partitioning",
    "torch_tensorrt.dynamo.runtime",
    "torch_tensorrt.dynamo.runtime.meta_ops",
    "torch_tensorrt.dynamo.tools",
    "torch_tensorrt.dynamo.tuning",
    "torch_tensorrt.executorch",
    "torch_tensorrt.runtime",
]

dynamo_package_dir = {
    "torch_tensorrt": "py/torch_tensorrt",
    "torch_tensorrt.distributed": "py/torch_tensorrt/distributed",
    "torch_tensorrt.distributed.run": "py/torch_tensorrt/distributed/run",
    "torch_tensorrt.kernels": "py/torch_tensorrt/kernels",
    "torch_tensorrt.reporting": "py/torch_tensorrt/reporting",
    "torch_tensorrt.verification": "py/torch_tensorrt/verification",
    "torch_tensorrt.dynamo": "py/torch_tensorrt/dynamo",
    "torch_tensorrt.dynamo.backend": "py/torch_tensorrt/dynamo/backend",
    "torch_tensorrt.dynamo.conversion": "py/torch_tensorrt/dynamo/conversion",
    "torch_tensorrt.dynamo.conversion.impl": "py/torch_tensorrt/dynamo/conversion/impl",
    "torch_tensorrt.dynamo.conversion.impl.activation": "py/torch_tensorrt/dynamo/conversion/impl/activation",
    "torch_tensorrt.dynamo.conversion.impl.condition": "py/torch_tensorrt/dynamo/conversion/impl/condition",
    "torch_tensorrt.dynamo.conversion.impl.elementwise": "py/torch_tensorrt/dynamo/conversion/impl/elementwise",
    "torch_tensorrt.dynamo.conversion.impl.normalization": "py/torch_tensorrt/dynamo/conversion/impl/normalization",
    "torch_tensorrt.dynamo.conversion.impl.slice": "py/torch_tensorrt/dynamo/conversion/impl/slice",
    "torch_tensorrt.dynamo.conversion.impl.unary": "py/torch_tensorrt/dynamo/conversion/impl/unary",
    "torch_tensorrt.dynamo.conversion.plugins": "py/torch_tensorrt/dynamo/conversion/plugins",
    "torch_tensorrt.dynamo.debug": "py/torch_tensorrt/dynamo/debug",
    "torch_tensorrt.dynamo.lowering": "py/torch_tensorrt/dynamo/lowering",
    "torch_tensorrt.dynamo.lowering.passes": "py/torch_tensorrt/dynamo/lowering/passes",
    "torch_tensorrt.dynamo.lowering.constant_fold_exclusions": "py/torch_tensorrt/dynamo/lowering/constant_fold_exclusions",
    "torch_tensorrt.dynamo.partitioning": "py/torch_tensorrt/dynamo/partitioning",
    "torch_tensorrt.dynamo.runtime": "py/torch_tensorrt/dynamo/runtime",
    "torch_tensorrt.dynamo.runtime.meta_ops": "py/torch_tensorrt/dynamo/runtime/meta_ops",
    "torch_tensorrt.dynamo.tools": "py/torch_tensorrt/dynamo/tools",
    "torch_tensorrt.dynamo.tuning": "py/torch_tensorrt/dynamo/tuning",
    "torch_tensorrt.executorch": "py/torch_tensorrt/executorch",
    "torch_tensorrt.runtime": "py/torch_tensorrt/runtime",
}

package_dir = dynamo_package_dir
packages = dynamo_packages
exclude_package_data = {
    "": [
        "py/torch_tensorrt/csrc/*.cpp",
        "torch_tensorrt/csrc/*.cpp",
        "test*",
        "*.cpp",
    ],
    "torch_tensorrt": [
        "py/torch_tensorrt/csrc/*.cpp",
        "torch_tensorrt/csrc/*.cpp",
        "test*",
        "*.cpp",
    ],
}

package_data = {}

if not PY_ONLY:
    package_data.update(
        {
            "torch_tensorrt": [
                "include/torch_tensorrt/*.h",
                "include/torch_tensorrt/core/*.h",
                "include/torch_tensorrt/core/runtime/*.h",
                "include/torch_tensorrt/core/util/*.h",
                "include/torch_tensorrt/core/util/logging/*.h",
                "lib/*",
            ]
        }
    )

with open(os.path.join(get_root_dir(), "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_jetpack_requirements(base_requirements):
    requirements = base_requirements + ["numpy<2.0.0"]
    if IS_DLFW_CI:
        return requirements
    else:
        return requirements + ["torch>=2.8.0,<2.9.0", "tensorrt>=10.3.0,<10.4.0"]


# The TensorRT requirement is the only part of this metadata that differs between an
# x86_64 host and an SBSA aarch64 one. uv records the answer of whichever machine ran
# `uv lock` and applies it to every platform, so a lock written on one architecture is
# stale on the other. Naming both behind environment markers makes the metadata the same
# everywhere, which is what lets a single uv.lock stay valid on both.
SBSA_MARKER = "platform_machine == 'aarch64' and sys_platform == 'linux'"
NON_SBSA_MARKER = "platform_machine != 'aarch64' or sys_platform != 'linux'"


def get_tensorrt_requirements():
    # TensorRT does not build wheels for Tegra, so SBSA takes the plain `tensorrt`
    # package from the tarball install; everywhere else takes the CUDA 13 trio.
    return [
        f"tensorrt>=11.4.0,<11.5.0; {SBSA_MARKER}",
        f"tensorrt-cu13>=11.4.0,<11.5.0; {NON_SBSA_MARKER}",
        f"tensorrt-cu13-bindings>=11.4.0,<11.5.0; {NON_SBSA_MARKER}",
        f"tensorrt-cu13-libs>=11.4.0,<11.5.0; {NON_SBSA_MARKER}",
    ]


def get_sbsa_requirements(base_requirements):
    requirements = base_requirements + ["numpy"]
    if IS_DLFW_CI:
        return requirements
    else:
        requirements = requirements + [
            "torch>=2.16.0.dev,<2.17.0",
        ]
        if USE_TRT_RTX:
            # TensorRT-RTX ships an aarch64 (SBSA) wheel; mirror get_x86_64_requirements.
            return requirements + [
                "tensorrt_rtx>=1.7.1.116,<1.8.0.0",
            ]
        # The sbsa torch_tensorrt wheel is the one thor uses, so the sbsa build takes the
        # plain tensorrt dependency; get_tensorrt_requirements carries that split.
        return (
            requirements
            + [
                "torch>=2.16.0.dev,<2.17.0",
            ]
            + get_tensorrt_requirements()
        )


def get_x86_64_requirements(base_requirements):
    requirements = base_requirements + ["numpy"]

    if IS_DLFW_CI:
        return requirements
    else:
        requirements = requirements + [
            "torch>=2.16.0.dev,<2.17.0",
        ]
        if USE_TRT_RTX:
            return requirements + [
                "tensorrt_rtx>=1.7.1.116,<1.8.0.0",
            ]
        else:
            cuda_version = torch.version.cuda
            if cuda_version.startswith("13"):
                requirements = requirements + get_tensorrt_requirements()
            else:
                raise ValueError(f"Unsupported CUDA version: {cuda_version}")

            return requirements


def get_requirements():
    base_requirements = [
        "packaging>=23",
        "typing-extensions>=4.7.0",
        "filelock>=4.2.1",
        "dllist",
        "psutil",
    ]

    if IS_JETPACK:
        requirements = get_jetpack_requirements(base_requirements)
    elif IS_SBSA:
        requirements = get_sbsa_requirements(base_requirements)
    else:
        # standard linux and windows requirements
        requirements = get_x86_64_requirements(base_requirements)
    return requirements


setup(
    name=PROJECT_NAME,
    version=__version__,
    cmdclass={
        "install": InstallCommand,
        "clean": CleanCommand,
        "develop": DevelopCommand,
        "bdist_wheel": BdistCommand,
        "editable_wheel": EditableWheelCommand,
    },
    zip_safe=False,
    install_requires=get_requirements(),
    extras_require=EXTRAS_REQUIRE,
    packages=packages,
    package_dir=package_dir,
    include_package_data=False,
    package_data=package_data,
    exclude_package_data=exclude_package_data,
)
