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
        floor = "2_39" if args.architecture == "aarch64" else "2_28"
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
            expected_version = (
                pinned
                if distribution == "executorch"
                else Version(importlib.metadata.version(distribution)).public
            )
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
        if any("+" in str(requirement.specifier) for requirement in requirements):
            reject("a requirement carries a local version label")
        # Reading every member verifies its RECORD hash, including the delegate payload.
        for filename in names:
            if not filename.endswith("/"):
                archive.read(filename)
    print(
        f"Validated {args.wheel.name}: one delegate, matching dependencies and {expected_tag}"
    )


if __name__ == "__main__":
    main()
