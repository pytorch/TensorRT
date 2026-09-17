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

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from wheel.wheelfile import WheelFile


def reject(message):
    sys.exit(f"FATAL: {message}")


_LINKED_AT_BUILD_TIME = frozenset({"torch", "torch-tensorrt"})


def main():
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
        else:
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
            "torchtrt_executorch-config.cmake",
            "torchtrt_executorch-config-version.cmake",
        ):
            if (
                f"torch_tensorrt_executorch_runtime/lib/cmake/torchtrt_executorch/{filename}"
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
        pinned = yaml.safe_load((root / "dev_dep_versions.yml").read_text())[
            "__executorch_version__"
        ]
        for distribution in (
            "executorch",
            "torch-tensorrt",
            "torch",
            "tensorrt-cu13",
            "nvidia-cuda-runtime",
        ):
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
            elif distribution in _LINKED_AT_BUILD_TIME:
                # PyTorch and Torch-TensorRT are linked the same way ExecuTorch is, so the
                # requirement names the whole installed version including any label. Comparing
                # against the public part alone reported a mismatch between a requirement and the
                # very version it was generated from.
                expected_version = str(
                    Version(importlib.metadata.version(distribution))
                )
            else:
                expected_version = Version(
                    importlib.metadata.version(distribution)
                ).public
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
        # The three runtimes the delegate links carry the label naming the build, deliberately: it
        # links one specific build of each, and without the label the requirement is satisfied by a
        # processor-only build or another CUDA build of the same date. They resolve from the CUDA
        # channel this wheel already requires. Everything else stays label-free so it resolves
        # anywhere, and a label appearing there would narrow the wheel for no reason.
        linked = {"executorch", "torch", "torch-tensorrt"}
        labelled = [
            requirement
            for requirement in requirements
            if "+" in str(requirement.specifier)
            and canonicalize_name(requirement.name) not in linked
        ]
        if labelled:
            reject(
                f"a requirement this delegate does not link carries a local label: {labelled}"
            )
        # ExecuTorch has to carry one, because its pin comes from a file in the repository that
        # always names a CUDA build. The other two take whatever the environment that built the
        # wheel had installed, and an environment can legitimately hold a version with no label,
        # so requiring one there would fail a wheel for something outside its control.
        unlabelled = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) == "executorch"
            and "+" not in str(requirement.specifier)
        ]
        if unlabelled:
            reject(
                "the ExecuTorch pin carries no label naming its build, so a processor-only "
                f"build would satisfy it: {unlabelled}"
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
