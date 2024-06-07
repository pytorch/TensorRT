import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

__version__ = "0.0.0"
__cuda_version__ = "0.0"
__tensorrt_version__ = "0.0"

LEADING_V_PATTERN = re.compile("^v")
TRAILING_RC_PATTERN = re.compile("-rc[0-9]*$")
LEGACY_BASE_VERSION_SUFFIX_PATTERN = re.compile("a0$")


class NoGitTagException(Exception):
    pass


def get_root_dir() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


def get_tag() -> str:
    root = get_root_dir()
    # We're on a tag
    am_on_tag = (
        subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    tag = ""
    if am_on_tag:
        dirty_tag = (
            subprocess.check_output(["git", "describe"], cwd=root)
            .decode("ascii")
            .strip()
        )
        # Strip leading v that we typically do when we tag branches
        # ie: v1.7.1 -> 1.7.1
        tag = re.sub(LEADING_V_PATTERN, "", dirty_tag)
        # Strip trailing rc pattern
        # ie: 1.7.1-rc1 -> 1.7.1
        tag = re.sub(TRAILING_RC_PATTERN, "", tag)
    return tag


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


class PytorchVersion:
    def __init__(
        self,
        gpu_arch_version: str,
        no_build_suffix: bool,
        base_build_version: str,
    ) -> None:
        self.gpu_arch_version = gpu_arch_version
        self.no_build_suffix = no_build_suffix
        if base_build_version == "":
            base_build_version = get_base_version()
        self.base_build_version = base_build_version

    def get_post_build_suffix(self) -> str:
        if self.no_build_suffix or self.gpu_arch_version is None:
            return ""
        return f"+{self.gpu_arch_version}"

    def get_release_version(self) -> str:
        if self.base_build_version:
            return f"{self.base_build_version}{self.get_post_build_suffix()}"
        if not get_tag():
            raise NoGitTagException(
                "Not on a git tag, are you sure you want a release version?"
            )
        return f"{get_tag()}{self.get_post_build_suffix()}"

    def get_nightly_version(self) -> str:
        date_str = datetime.today().strftime("%Y%m%d")
        build_suffix = self.get_post_build_suffix()
        return f"{self.base_build_version}.dev{date_str}{build_suffix}"


def load_dep_info():
    global __cuda_version__
    global __tensorrt_version__
    with open("dev_dep_versions.yml", "r") as stream:
        versions = yaml.safe_load(stream)
        gpu_arch_version = os.environ.get("CU_VERSION")
        if gpu_arch_version is not None:
            __cuda_version__ = (
                (gpu_arch_version[2:])[:-1] + "." + (gpu_arch_version[2:])[-1:]
            )
        else:
            __cuda_version__ = versions["__cuda_version__"]
        __tensorrt_version__ = versions["__tensorrt_version__"]


load_dep_info()

gpu_arch_version = os.environ.get("CU_VERSION")
version = PytorchVersion(
    gpu_arch_version=gpu_arch_version,
    no_build_suffix=False,
    base_build_version=get_base_version(),
)


def torch_tensorrt_version_nightly():
    print(version.get_nightly_version())


def torch_tensorrt_version_release():
    print(version.get_release_version())


def cuda_version():
    print(__cuda_version__)


def tensorrt_version():
    print(__tensorrt_version__)
