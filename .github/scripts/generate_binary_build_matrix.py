#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on four different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
    * Latest XPU
"""


import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

PYTHON_ARCHES_DICT = {
    "nightly": ["3.9", "3.10", "3.11", "3.12"],
    "test": ["3.9", "3.10", "3.11", "3.12"],
    "release": ["3.9", "3.10", "3.11", "3.12"],
}
CUDA_ARCHES_DICT = {
    "nightly": ["11.8", "12.4", "12.6"],
    "test": ["11.8", "12.1", "12.4"],
    "release": ["11.8", "12.1", "12.4"],
}
ROCM_ARCHES_DICT = {
    "nightly": ["6.1", "6.2"],
    "test": ["6.1", "6.2"],
    "release": ["6.1", "6.2"],
}

CUDA_CUDDN_VERSIONS = {
    "11.8": {"cuda": "11.8.0", "cudnn": "9"},
    "12.1": {"cuda": "12.1.1", "cudnn": "9"},
    "12.4": {"cuda": "12.4.1", "cudnn": "9"},
    "12.6": {"cuda": "12.6.2", "cudnn": "9"},
}

PACKAGE_TYPES = ["wheel", "conda", "libtorch"]
PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
RELEASE = "release"
DEBUG = "debug"
NIGHTLY = "nightly"
TEST = "test"

# OS constants
LINUX = "linux"
LINUX_AARCH64 = "linux-aarch64"
MACOS_ARM64 = "macos-arm64"
WINDOWS = "windows"

# Accelerator architectures
CPU = "cpu"
CPU_AARCH64 = "cpu-aarch64"
CUDA_AARCH64 = "cuda-aarch64"
CUDA = "cuda"
ROCM = "rocm"
XPU = "xpu"


CURRENT_NIGHTLY_VERSION = "2.6.0"
CURRENT_CANDIDATE_VERSION = "2.5.1"
CURRENT_STABLE_VERSION = "2.5.1"
CURRENT_VERSION = CURRENT_STABLE_VERSION

# By default use Nightly for CUDA arches
CUDA_ARCHES = CUDA_ARCHES_DICT[NIGHTLY]
ROCM_ARCHES = ROCM_ARCHES_DICT[NIGHTLY]
PYTHON_ARCHES = PYTHON_ARCHES_DICT[NIGHTLY]

# Container images
LIBTORCH_CONTAINER_IMAGES: Dict[Tuple[str, str], str]
WHEEL_CONTAINER_IMAGES: Dict[str, str]

LINUX_GPU_RUNNER = "linux.g5.4xlarge.nvidia.gpu"
LINUX_CPU_RUNNER = "linux.2xlarge"
LINUX_AARCH64_RUNNER = "linux.arm64.2xlarge"
LINUX_AARCH64_GPU_RUNNER = "linux.arm64.m7g.4xlarge"
WIN_GPU_RUNNER = "windows.g4dn.xlarge"
WIN_CPU_RUNNER = "windows.4xlarge"
MACOS_M1_RUNNER = "macos-m1-stable"

PACKAGES_TO_INSTALL_WHL = "torch torchvision torchaudio"
WHL_INSTALL_BASE = "pip3 install"
DOWNLOAD_URL_BASE = "https://download.pytorch.org"

ENABLE = "enable"
DISABLE = "disable"


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return CUDA
    elif arch_version in ROCM_ARCHES:
        return ROCM
    elif arch_version == CPU_AARCH64:
        return CPU_AARCH64
    elif arch_version == CUDA_AARCH64:
        return CUDA_AARCH64
    elif arch_version == XPU:
        return XPU
    else:  # arch_version should always be CPU in this case
        return CPU


def validation_runner(arch_type: str, os: str) -> str:
    if os == LINUX:
        if arch_type == CUDA:
            return LINUX_GPU_RUNNER
        else:
            return LINUX_CPU_RUNNER
    elif os == LINUX_AARCH64:
        if arch_type == CUDA_AARCH64:
            return LINUX_AARCH64_GPU_RUNNER
        else:
            return LINUX_AARCH64_RUNNER
    elif os == WINDOWS:
        if arch_type == CUDA:
            return WIN_GPU_RUNNER
        else:
            return WIN_CPU_RUNNER
    elif os == MACOS_ARM64:
        return MACOS_M1_RUNNER
    else:  # default to linux cpu runner
        return LINUX_CPU_RUNNER


def initialize_globals(channel: str, build_python_only: bool) -> None:
    global CURRENT_VERSION, CUDA_ARCHES, ROCM_ARCHES, PYTHON_ARCHES
    global WHEEL_CONTAINER_IMAGES, LIBTORCH_CONTAINER_IMAGES
    if channel == TEST:
        CURRENT_VERSION = CURRENT_CANDIDATE_VERSION
    else:
        CURRENT_VERSION = CURRENT_STABLE_VERSION

    CUDA_ARCHES = CUDA_ARCHES_DICT[channel]
    ROCM_ARCHES = ROCM_ARCHES_DICT[channel]
    if build_python_only:
        # Only select the oldest version of python if building a python only package
        PYTHON_ARCHES = [PYTHON_ARCHES_DICT[channel][0]]
    else:
        PYTHON_ARCHES = PYTHON_ARCHES_DICT[channel]
    WHEEL_CONTAINER_IMAGES = {
        "11.8": "pytorch/manylinux2_28-builder:cuda11.8",
        "12.1": "pytorch/manylinux2_28-builder:cuda12.1",
        "12.4": "pytorch/manylinux2_28-builder:cuda12.4",
        "12.6": "pytorch/manylinux2_28-builder:cuda12.6",
        **{
            gpu_arch: f"pytorch/manylinux2_28-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        CPU: "pytorch/manylinux2_28-builder:cpu",
        XPU: "pytorch/manylinux2_28-builder:xpu",
        # TODO: Migrate CUDA_AARCH64 image to manylinux2_28_aarch64-builder:cuda12.4
        CPU_AARCH64: "pytorch/manylinux2_28_aarch64-builder:cpu-aarch64",
        CUDA_AARCH64: "pytorch/manylinuxaarch64-builder:cuda12.4",
    }
    LIBTORCH_CONTAINER_IMAGES = {
        **{
            (gpu_arch, PRE_CXX11_ABI): f"pytorch/manylinux2_28-builder:cuda{gpu_arch}"
            for gpu_arch in CUDA_ARCHES
        },
        **{
            (gpu_arch, CXX11_ABI): f"pytorch/libtorch-cxx11-builder:cuda{gpu_arch}"
            for gpu_arch in CUDA_ARCHES
        },
        **{
            (gpu_arch, PRE_CXX11_ABI): f"pytorch/manylinux2_28-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        **{
            (gpu_arch, CXX11_ABI): f"pytorch/libtorch-cxx11-builder:rocm{gpu_arch}"
            for gpu_arch in ROCM_ARCHES
        },
        (CPU, PRE_CXX11_ABI): "pytorch/manylinux2_28-builder:cpu",
        (CPU, CXX11_ABI): "pytorch/libtorch-cxx11-builder:cpu",
    }


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        CPU: "cpu",
        CPU_AARCH64: CPU,
        CUDA_AARCH64: "cu124",
        CUDA: f"cu{gpu_arch_version.replace('.', '')}",
        ROCM: f"rocm{gpu_arch_version}",
        XPU: "xpu",
    }.get(gpu_arch_type, gpu_arch_version)


def list_without(in_list: List[str], without: List[str]) -> List[str]:
    return [item for item in in_list if item not in without]


def get_base_download_url_for_repo(
    repo: str, channel: str, gpu_arch_type: str, desired_cuda: str
) -> str:
    base_url_for_type = f"{DOWNLOAD_URL_BASE}/{repo}"
    base_url_for_type = (
        base_url_for_type if channel == RELEASE else f"{base_url_for_type}/{channel}"
    )

    if gpu_arch_type != CPU:
        base_url_for_type = f"{base_url_for_type}/{desired_cuda}"
    else:
        base_url_for_type = f"{base_url_for_type}/{gpu_arch_type}"

    return base_url_for_type


def get_libtorch_install_command(
    os: str,
    channel: str,
    gpu_arch_type: str,
    libtorch_variant: str,
    devtoolset: str,
    desired_cuda: str,
    libtorch_config: str,
) -> str:
    prefix = "libtorch" if os != WINDOWS else "libtorch-win"
    _libtorch_variant = (
        f"{libtorch_variant}-{libtorch_config}"
        if libtorch_config == "debug"
        else libtorch_variant
    )
    build_name = (
        f"{prefix}-{devtoolset}-{_libtorch_variant}-latest.zip"
        if devtoolset == "cxx11-abi"
        else f"{prefix}-{_libtorch_variant}-latest.zip"
    )

    if os == MACOS_ARM64:
        arch = "arm64"
        build_name = f"libtorch-macos-{arch}-latest.zip"
        if channel in [RELEASE, TEST]:
            build_name = f"libtorch-macos-{arch}-{CURRENT_VERSION}.zip"

    elif os == LINUX and (channel in (RELEASE, TEST)):
        build_name = (
            f"{prefix}-{devtoolset}-{_libtorch_variant}-{CURRENT_VERSION}%2B{desired_cuda}.zip"
            if devtoolset == "cxx11-abi"
            else f"{prefix}-{_libtorch_variant}-{CURRENT_VERSION}%2B{desired_cuda}.zip"
        )
    elif os == WINDOWS and (channel in (RELEASE, TEST)):
        build_name = (
            f"{prefix}-shared-with-deps-debug-{CURRENT_VERSION}%2B{desired_cuda}.zip"
            if libtorch_config == "debug"
            else f"{prefix}-shared-with-deps-{CURRENT_VERSION}%2B{desired_cuda}.zip"
        )
    elif os == WINDOWS and channel == NIGHTLY:
        build_name = (
            f"{prefix}-shared-with-deps-debug-latest.zip"
            if libtorch_config == "debug"
            else f"{prefix}-shared-with-deps-latest.zip"
        )

    return f"{get_base_download_url_for_repo('libtorch', channel, gpu_arch_type, desired_cuda)}/{build_name}"


def get_wheel_install_command(
    os: str,
    channel: str,
    gpu_arch_type: str,
    gpu_arch_version: str,
    desired_cuda: str,
    python_version: str,
    use_only_dl_pytorch_org: bool,
    use_split_build: bool = False,
) -> str:
    if use_split_build:
        if (gpu_arch_version in CUDA_ARCHES) and (os == LINUX) and (channel == NIGHTLY):
            return f"{WHL_INSTALL_BASE} {PACKAGES_TO_INSTALL_WHL} --index-url {get_base_download_url_for_repo('whl', channel, gpu_arch_type, desired_cuda)}_pypi_pkg"  # noqa: E501
        else:
            raise ValueError(
                "Split build is not supported for this configuration. It is only supported for CUDA 11.8, 12.4, 12.6 on Linux nightly builds."  # noqa: E501
            )
    if (
        channel == RELEASE
        and (not use_only_dl_pytorch_org)
        and (
            (gpu_arch_version == "12.4" and os == LINUX)
            or (gpu_arch_type == CPU and os in [WINDOWS, MACOS_ARM64])
            or (os == LINUX_AARCH64)
        )
    ):
        return f"{WHL_INSTALL_BASE} {PACKAGES_TO_INSTALL_WHL}"
    else:
        whl_install_command = (
            f"{WHL_INSTALL_BASE} --pre {PACKAGES_TO_INSTALL_WHL}"
            if channel == "nightly"
            else f"{WHL_INSTALL_BASE} {PACKAGES_TO_INSTALL_WHL}"
        )
        return f"{whl_install_command} --index-url {get_base_download_url_for_repo('whl', channel, gpu_arch_type, desired_cuda)}"  # noqa: E501


def generate_conda_matrix(
    os: str,
    channel: str,
    with_cuda: str,
    with_rocm: str,
    with_cpu: str,
    with_xpu: str,
    limit_pr_builds: bool,
    use_only_dl_pytorch_org: bool,
    use_split_build: bool = False,
    python_versions: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    # return empty list. Conda builds are deprecated, see https://github.com/pytorch/pytorch/issues/138506
    return ret


def generate_libtorch_matrix(
    os: str,
    channel: str,
    with_cuda: str,
    with_rocm: str,
    with_cpu: str,
    with_xpu: str,
    limit_pr_builds: bool,
    use_only_dl_pytorch_org: bool,
    use_split_build: bool = False,
    python_versions: Optional[List[str]] = None,
    abi_versions: Optional[List[str]] = None,
    arches: Optional[List[str]] = None,
    libtorch_variants: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []

    if arches is None:
        arches = []

        if with_cpu == ENABLE:
            arches += [CPU]

        if with_cuda == ENABLE and os in (LINUX, WINDOWS):
            arches += CUDA_ARCHES

        if with_rocm == ENABLE and os == LINUX:
            arches += ROCM_ARCHES

    if abi_versions is None:
        if os == WINDOWS:
            abi_versions = [RELEASE, DEBUG]
        elif os == LINUX:
            abi_versions = [PRE_CXX11_ABI, CXX11_ABI]
        elif os in [MACOS_ARM64]:
            abi_versions = [CXX11_ABI]
        else:
            abi_versions = []

    if libtorch_variants is None:
        libtorch_variants = [
            "shared-with-deps",
        ]

    global LIBTORCH_CONTAINER_IMAGES

    for abi_version in abi_versions:
        for arch_version in arches:
            for libtorch_variant in libtorch_variants:
                # one of the values in the following list must be exactly
                # CXX11_ABI, but the precise value of the other one doesn't
                # matter
                gpu_arch_type = arch_type(arch_version)
                gpu_arch_version = "" if arch_version == CPU else arch_version

                desired_cuda = translate_desired_cuda(gpu_arch_type, gpu_arch_version)
                devtoolset = abi_version if os != WINDOWS else ""
                libtorch_config = abi_version if os == WINDOWS else ""
                ret.append(
                    {
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": desired_cuda,
                        "libtorch_variant": libtorch_variant,
                        "libtorch_config": libtorch_config,
                        "devtoolset": devtoolset,
                        "container_image": (
                            LIBTORCH_CONTAINER_IMAGES[(arch_version, abi_version)]
                            if os != WINDOWS
                            else ""
                        ),
                        "package_type": "libtorch",
                        "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}".replace(  # noqa: E501
                            ".", "_"
                        ),
                        # Please noe since libtorch validations are minimal, we use CPU runners
                        "validation_runner": validation_runner(CPU, os),
                        "installation": get_libtorch_install_command(
                            os,
                            channel,
                            gpu_arch_type,
                            libtorch_variant,
                            devtoolset,
                            desired_cuda,
                            libtorch_config,
                        ),
                        "channel": channel,
                        "stable_version": CURRENT_VERSION,
                    }
                )
    return ret


def generate_wheels_matrix(
    os: str,
    channel: str,
    with_cuda: str,
    with_rocm: str,
    with_cpu: str,
    with_xpu: str,
    limit_pr_builds: bool,
    use_only_dl_pytorch_org: bool,
    use_split_build: bool = False,
    python_versions: Optional[List[str]] = None,
    arches: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    package_type = "wheel"

    if not python_versions:
        # Define default python version
        python_versions = list(PYTHON_ARCHES)

        # If the list of python versions is set explicitly by the caller, stick with it instead
        # of trying to add more versions behind the scene
        if channel == NIGHTLY and (os in (LINUX, MACOS_ARM64, LINUX_AARCH64)):
            python_versions += ["3.13"]

    if os == LINUX:
        # NOTE: We only build manywheel packages for linux
        package_type = "manywheel"

    upload_to_base_bucket = "yes"
    if arches is None:
        # Define default compute architectures
        arches = []

        if with_cpu == ENABLE:
            arches += [CPU]

        if os == LINUX_AARCH64:
            # Only want the one arch as the CPU type is different and
            # uses different build/test scripts
            arches = [CPU_AARCH64, CUDA_AARCH64]

        if with_cuda == ENABLE:
            upload_to_base_bucket = "no"
            if os in (LINUX, WINDOWS):
                arches += CUDA_ARCHES

        if with_rocm == ENABLE and os == LINUX:
            arches += ROCM_ARCHES

        if with_xpu == ENABLE and os in (LINUX, WINDOWS):
            arches += [XPU]

    if limit_pr_builds:
        python_versions = [python_versions[0]]

    global WHEEL_CONTAINER_IMAGES

    ret: List[Dict[str, Any]] = []
    for python_version in python_versions:
        for arch_version in arches:

            # TODO: Enable Python 3.13 support for ROCM
            if arch_version in ROCM_ARCHES and python_version == "3.13":
                continue

            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = (
                "" if arch_version in [CPU, CPU_AARCH64, XPU] else arch_version
            )

            desired_cuda = translate_desired_cuda(gpu_arch_type, gpu_arch_version)
            entry = {
                "python_version": python_version,
                "gpu_arch_type": gpu_arch_type,
                "gpu_arch_version": gpu_arch_version,
                "desired_cuda": desired_cuda,
                "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                "package_type": package_type,
                "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                    ".", "_"
                ),
                "validation_runner": validation_runner(gpu_arch_type, os),
                "installation": get_wheel_install_command(
                    os,
                    channel,
                    gpu_arch_type,
                    gpu_arch_version,
                    desired_cuda,
                    python_version,
                    use_only_dl_pytorch_org,
                ),
                "channel": channel,
                "upload_to_base_bucket": upload_to_base_bucket,
                "stable_version": CURRENT_VERSION,
                "use_split_build": False,
            }
            ret.append(entry)
            if (
                use_split_build
                and (gpu_arch_version in CUDA_ARCHES)
                and (os == LINUX)
                and (channel == NIGHTLY)
            ):
                entry = entry.copy()
                entry["build_name"] = (
                    f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}-split".replace(
                        ".", "_"
                    )
                )
                entry["use_split_build"] = True
                ret.append(entry)

    return ret


GENERATING_FUNCTIONS_BY_PACKAGE_TYPE: Dict[str, Callable[..., List[Dict[str, str]]]] = {
    "wheel": generate_wheels_matrix,
    "conda": generate_conda_matrix,
    "libtorch": generate_libtorch_matrix,
}


def generate_build_matrix(
    package_type: str,
    operating_system: str,
    channel: str,
    with_cuda: str,
    with_rocm: str,
    with_cpu: str,
    with_xpu: str,
    limit_pr_builds: str,
    use_only_dl_pytorch_org: str,
    build_python_only: str,
    use_split_build: str = "false",
    python_versions: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    includes = []

    package_types = package_type.split(",")
    if len(package_types) == 1:
        package_types = PACKAGE_TYPES if package_type == "all" else [package_type]

    channels = CUDA_ARCHES_DICT.keys() if channel == "all" else [channel]

    for channel in channels:
        for package in package_types:
            initialize_globals(channel, build_python_only == ENABLE)
            includes.extend(
                GENERATING_FUNCTIONS_BY_PACKAGE_TYPE[package](
                    operating_system,
                    channel,
                    with_cuda,
                    with_rocm,
                    with_cpu,
                    with_xpu,
                    limit_pr_builds == "true",
                    use_only_dl_pytorch_org == "true",
                    use_split_build == "true",
                    python_versions,
                )
            )

    return {"include": includes}


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-type",
        help="Package type to lookup for, also supports comma separated values",
        type=str,
        default=os.getenv("PACKAGE_TYPE", "wheel"),
    )
    parser.add_argument(
        "--operating-system",
        help="Operating system to generate for",
        type=str,
        default=os.getenv("OS", LINUX),
    )
    parser.add_argument(
        "--channel",
        help="Channel to use, default nightly",
        type=str,
        choices=["nightly", "test", "release", "all"],
        default=os.getenv("CHANNEL", "nightly"),
    )
    parser.add_argument(
        "--with-cuda",
        help="Build with Cuda?",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_CUDA", ENABLE),
    )
    parser.add_argument(
        "--with-rocm",
        help="Build with Rocm?",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_ROCM", ENABLE),
    )
    parser.add_argument(
        "--with-cpu",
        help="Build with CPU?",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_CPU", ENABLE),
    )
    parser.add_argument(
        "--with-xpu",
        help="Build with XPU?",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("WITH_XPU", ENABLE),
    )
    # By default this is false for this script but expectation is that the caller
    # workflow will default this to be true most of the time, where a pull
    # request is synchronized and does not contain the label "ciflow/binaries/all"
    parser.add_argument(
        "--limit-pr-builds",
        help="Limit PR builds to single python/cuda config",
        type=str,
        choices=["true", "false"],
        default=os.getenv("LIMIT_PR_BUILDS", "false"),
    )
    # This is used when testing release builds to test release binaries
    # only from download.pytorch.org. When pipy binaries are not released yet.
    parser.add_argument(
        "--use-only-dl-pytorch-org",
        help="Use only download.pytorch.org when gen wheel install command?",
        type=str,
        choices=["true", "false"],
        default=os.getenv("USE_ONLY_DL_PYTORCH_ORG", "false"),
    )
    # Generates a single version python for building python packages only
    # This basically makes it so that we only generate a matrix including the oldest
    # version of python that we support
    # For packages that look similar to torchtune-0.0.1-py3-none-any.whl
    parser.add_argument(
        "--build-python-only",
        help="Build python only",
        type=str,
        choices=[ENABLE, DISABLE],
        default=os.getenv("BUILD_PYTHON_ONLY", ENABLE),
    )

    parser.add_argument(
        "--use-split-build",
        help="Use split build for wheel",
        type=str,
        choices=["true", "false"],
        default=os.getenv("USE_SPLIT_BUILD", DISABLE),
    )

    parser.add_argument(
        "--python-versions",
        help="Only build the select JSON-encoded list of python versions",
        type=str,
        default=os.getenv("PYTHON_VERSIONS", "[]"),
    )

    options = parser.parse_args(args)
    try:
        python_versions = json.loads(options.python_versions)
    except json.JSONDecodeError:
        python_versions = None

    assert (
        options.with_cuda or options.with_rocm or options.with_xpu or options.with_cpu
    ), "Must build with either CUDA, ROCM, XPU, or CPU support."

    build_matrix = generate_build_matrix(
        options.package_type,
        options.operating_system,
        options.channel,
        options.with_cuda,
        options.with_rocm,
        options.with_cpu,
        options.with_xpu,
        options.limit_pr_builds,
        options.use_only_dl_pytorch_org,
        options.build_python_only,
        options.use_split_build,
        python_versions,
    )

    print(json.dumps(build_matrix))


if __name__ == "__main__":
    main(sys.argv[1:])
