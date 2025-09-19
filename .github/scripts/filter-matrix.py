#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# currently we don't support python 3.13t due to tensorrt does not support 3.13t
disabled_python_versions: List[str] = ["3.13t", "3.14", "3.14t"]
disabled_cuda_versions: List[str] = ["cu130"]

# jetpack 6.2 only officially supports python 3.10 and cu126
jetpack_python_versions: List[str] = ["3.10"]
jetpack_cuda_versions: List[str] = ["cu126"]

jetpack_container_image: str = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
sbsa_container_image: str = "quay.io/pypa/manylinux_2_34_aarch64"


def validate_matrix(matrix_dict: Dict[str, Any]) -> None:
    """Validate the structure of the input matrix."""
    if not isinstance(matrix_dict, dict):
        raise ValueError("Matrix must be a dictionary")
    if "include" not in matrix_dict:
        raise ValueError("Matrix must contain 'include' key")
    if not isinstance(matrix_dict["include"], list):
        raise ValueError("Matrix 'include' must be a list")


def filter_matrix_item(
    item: Dict[str, Any],
    is_jetpack: bool,
    limit_pr_builds: bool,
) -> bool:
    """Filter a single matrix item based on the build type and requirements."""
    if item["python_version"] in disabled_python_versions:
        # Skipping disabled Python version
        return False
    if item["desired_cuda"] in disabled_cuda_versions:
        # Skipping disabled CUDA version
        return False
    if is_jetpack:
        if limit_pr_builds:
            # pr build,matrix passed from test-infra is cu128, python 3.9, change to cu126, python 3.10
            item["desired_cuda"] = "cu126"
            item["python_version"] = "3.10"
            item["container_image"] = jetpack_container_image
            return True
        else:
            # nightly/main build, matrix passed from test-infra is cu128, all python versions, change to cu126, python 3.10
            if item["python_version"] in jetpack_python_versions:
                item["desired_cuda"] = "cu126"
                item["container_image"] = jetpack_container_image
                return True
            return False
    else:
        if item["gpu_arch_type"] == "cuda-aarch64":
            # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
            # however, TensorRT requires glibc2.31 on aarch64 platform
            # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
            item["container_image"] = sbsa_container_image
            return True
        return True


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix blob",
        type=str,
        default="",
    )

    parser.add_argument(
        "--jetpack",
        help="is jetpack",
        type=str,
        choices=["true", "false"],
        default="false",
    )

    parser.add_argument(
        "--limit-pr-builds",
        help="If it is a PR build",
        type=str,
        choices=["true", "false"],
        default=os.getenv("LIMIT_PR_BUILDS", "false"),
    )

    options = parser.parse_args(args)
    if options.matrix == "":
        raise ValueError("--matrix needs to be provided")

    try:
        matrix_dict = json.loads(options.matrix)
        validate_matrix(matrix_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in matrix: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid matrix structure: {e}")

    includes = matrix_dict["include"]
    filtered_includes = []

    for item in includes:
        if filter_matrix_item(
            item,
            options.jetpack == "true",
            options.limit_pr_builds == "true",
        ):
            filtered_includes.append(item)

    filtered_matrix_dict = {"include": filtered_includes}
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
