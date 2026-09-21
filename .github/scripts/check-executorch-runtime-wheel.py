# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the repaired companion wheel before it enters the shared artifact."""

import argparse
import ast
import importlib.metadata
import re
import sys
from email.parser import BytesParser
from pathlib import Path
from typing import NoReturn

import yaml
from wheel.wheelfile import WheelFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def reject(message: str) -> NoReturn:
    sys.exit(f"FATAL: {message}")


_LINKED_DISTRIBUTIONS = ("executorch", "tensorrt-cu13", "nvidia-cuda-runtime")
# Required but not linked: ExecuTorch's Python imports it and declares it nowhere.
_UNPINNED_DISTRIBUTIONS = ("torch",)

# The pin each remaining requirement has to agree with. These name a release series rather than an
# exact build, so the comparison is on the leading release components the pin actually spells.
_REPOSITORY_PIN = {
    "tensorrt-cu13": "__tensorrt_version__",
    "nvidia-cuda-runtime": "__cuda_version__",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--architecture", choices=("x86_64", "aarch64"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "py/torch-tensorrt-executorch-runtime/setup.py"
    (library,) = [
        ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "DELEGATE_LIBRARY" for target in node.targets
        )
    ]
    with WheelFile(args.wheel) as archive:
        names = archive.namelist()
        objects = sorted(name for name in names if re.search(r"\.so(\.\d+)*$", name))
        expected = f"torch_tensorrt_executorch_runtime/lib/{library}"
        if objects != [expected]:
            reject(f"expected {expected} and no other shared libraries, got {objects}")
        # The platform tag is a claim about the payload, and until now nothing read the payload
        # to check it, so a wheel tagged for one architecture could carry a library built for
        # the other and pass. The ELF header names the machine in two bytes at offset 18.
        header = archive.read(expected)[:20]
        if header[:4] != b"\x7fELF":
            reject(f"{expected} is not an ELF object")
        machine = int.from_bytes(header[18:20], "little")
        wanted = {"x86_64": 0x3E, "aarch64": 0xB7}[args.architecture]
        if machine != wanted:
            names_by_machine = {0x3E: "x86_64", 0xB7: "aarch64"}
            reject(
                f"{expected} is built for "
                f"{names_by_machine.get(machine, hex(machine))}, but this wheel is tagged for "
                f"{args.architecture}"
            )
        for filename in (
            "executorch_backend_tensorrt-config.cmake",
            "executorch_backend_tensorrt-config-version.cmake",
        ):
            if (
                f"torch_tensorrt_executorch_runtime/lib/cmake/executorch_backend_tensorrt/{filename}"
                not in names
            ):
                reject(f"the wheel ships no CMake package: {filename} is missing")
        forbidden = [
            name
            for name in names
            if any(
                part in name
                for part in (
                    "_portable_lib",
                    "libexecutorch.so",
                    "libextension_cuda",
                    "libaoti_cuda_shims",
                )
            )
        ]
        if forbidden:
            reject(f"the wheel ships ExecuTorch runtime components: {forbidden}")

        name, version, _, tags = parse_wheel_filename(args.wheel.name)
        if name != "torch-tensorrt-executorch-runtime":
            reject(f"unexpected distribution: {name}")
        # The floors differ by architecture. The Arm build container carries no devtoolset, so the
        # C++ runtime symbols the delegate references are not absorbed statically the way they are
        # on x86, and the wheel genuinely needs the newer baseline.
        floor = {"x86_64": "2_28", "aarch64": "2_35"}[args.architecture]
        expected_tag = f"py3-none-manylinux_{floor}_{args.architecture}"
        if {str(tag) for tag in tags} != {expected_tag}:
            reject(f"expected repaired tag {expected_tag}, got {tags}")
        wheel_metadata = BytesParser().parsebytes(
            archive.read(f"{archive.dist_info_path}/WHEEL")
        )
        if wheel_metadata.get("Root-Is-Purelib") != "false":
            reject("wheel declares itself pure python")
        if set(wheel_metadata.get_all("Tag", [])) != {expected_tag}:
            reject("WHEEL tags disagree with the filename")
        metadata = BytesParser().parsebytes(
            archive.read(f"{archive.dist_info_path}/METADATA")
        )
        if (
            canonicalize_name(metadata["Name"]) != name
            or Version(metadata["Version"]) != version
        ):
            reject("METADATA name/version disagree with the filename")
        requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
        pins = yaml.safe_load((root / "dev_dep_versions.yml").read_text())
        pinned = pins["__executorch_version__"]
        # Present and unbounded. A bound here would have this wheel decide which PyTorch is
        # acceptable, which ExecuTorch deliberately leaves to whoever installs it.
        for name in _UNPINNED_DISTRIBUTIONS:
            matched = [r for r in requirements if canonicalize_name(r.name) == name]
            if len(matched) != 1:
                reject(f"the wheel must require {name} exactly once, found {matched}")
            elif str(matched[0].specifier):
                reject(
                    f"the wheel pins {matched[0]}, but {name} is required unbounded so the user "
                    "chooses it"
                )
        for forbidden in ("torch-tensorrt",):
            present = [
                r for r in requirements if canonicalize_name(r.name) == forbidden
            ]
            if present:
                reject(
                    f"the wheel requires {forbidden}, which the delegate does not link: {present}"
                )

        # Only the three the delegate links. Neither PyTorch nor Torch-TensorRT belongs here: the
        # library links neither, requiring Torch-TensorRT would make this wheel depend on the project
        # that builds it, and ExecuTorch leaves the choice of PyTorch build to the user.
        for distribution in _LINKED_DISTRIBUTIONS:
            if distribution == "executorch":
                # The delegate links one specific ExecuTorch build, so its requirement carries the
                # label naming that build. Without it the requirement is satisfied by a
                # processor-only build, or another CUDA build of the same date. Compare against the
                # installed wheel, whose label is the one the delegate actually linked, and check
                # the public part still matches the repository pin.
                installed = Version(importlib.metadata.version(distribution))
                if installed.public != pinned:
                    reject(
                        f"the repository pins executorch=={pinned}, but this wheel was built "
                        f"against {installed}, whose version differs from that pin"
                    )
                expected_version = str(installed)
            else:
                # Against the repository's own pin, not just against the environment that produced
                # the requirement. Taking the expected value from the same environment setup.py read
                # it from makes the comparison agree with itself, so a wheel built against the wrong
                # TensorRT or CUDA runtime validated clean.
                installed = Version(importlib.metadata.version(distribution))
                pin = pins[_REPOSITORY_PIN[distribution]]
                if distribution == "nvidia-cuda-runtime":
                    # The build row picks the minor, and the file pin names only one row, so
                    # compare what the builder itself requires: the CUDA major.
                    pin = pin.split(".")[0]
                if (
                    installed.release[: len(Version(pin).release)]
                    != Version(pin).release
                ):
                    reject(
                        f"the repository pins {_REPOSITORY_PIN[distribution]} {pin}, but this "
                        f"wheel was built against {distribution} {installed}"
                    )
                expected_version = installed.public
            matched = [
                r for r in requirements if canonicalize_name(r.name) == distribution
            ]
            if (
                len(matched) != 1
                or str(matched[0].specifier) != f"=={expected_version}"
                or matched[0].marker
                or matched[0].extras
                or matched[0].url
            ):
                reason = (
                    "the repository pins"
                    if distribution == "executorch"
                    else "the build used"
                )
                reject(
                    f"{reason} {distribution}=={expected_version}, but the wheel requires {matched}"
                )
        # The loop checks each of the three it names; a fourth requirement is invisible to it, and an
        # unlabelled one clears the local-label check below as well.
        unexpected = sorted(
            str(requirement)
            for requirement in requirements
            if canonicalize_name(requirement.name)
            not in _LINKED_DISTRIBUTIONS + _UNPINNED_DISTRIBUTIONS
        )
        if unexpected:
            reject(f"the wheel requires more than the delegate links: {unexpected}")
        # The three runtimes the delegate links carry the label naming the build, deliberately: it
        # links one specific build of each, and without the label the requirement is satisfied by a
        # processor-only build or another CUDA build of the same date. They resolve from the CUDA
        # channel this wheel already requires. Everything else stays label-free so it resolves
        # anywhere, and a label appearing there would narrow the wheel for no reason.
        label_allowed = {"executorch", "torch", "torch-tensorrt"}
        labelled = [
            requirement
            for requirement in requirements
            if "+" in str(requirement.specifier)
            and canonicalize_name(requirement.name) not in label_allowed
        ]
        if labelled:
            reject(
                f"a requirement this delegate does not link carries a local label: {labelled}"
            )
        # Reading every member verifies its RECORD hash, including the delegate payload.
        for filename in names:
            if not filename.endswith("/"):
                archive.read(filename)
    print(
        f"Validated {args.wheel.name}: one delegate, matching dependencies and {expected_tag}"
    )


if __name__ == "__main__":
    main()
