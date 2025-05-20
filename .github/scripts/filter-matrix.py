#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List

# currently we don't support python 3.13t due to tensorrt does not support 3.13t
disabled_python_versions: List[str] = ["3.13t"]

# jetpack 6.2 supports python 3.10 and cuda 12.6
jetpack_python_versions: List[str] = ["3.9", "3.10", "3.11", "3.12"]

jetpack_cuda_versions: List[str] = ["cu126"]

jetpack_container_image: str = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
sbsa_container_image: str = "quay.io/pypa/manylinux_2_34_aarch64"


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
        raise Exception("--matrix needs to be provided")

    matrix_dict = json.loads(options.matrix)
    includes = matrix_dict["include"]
    filtered_includes = []
    for item in includes:
        if item["python_version"] in disabled_python_versions:
            continue
        if options.jetpack == "true":
            if item["python_version"] in jetpack_python_versions:
                # in the PR Branch, we only have cu128 passed in as matrix from test-infra, change to cu126
                if options.limit_pr_builds == "true":
                    item["desired_cuda"] = "cu126"
                    item["container_image"] = jetpack_container_image
                    filtered_includes.append(item)
                else:
                    if item["desired_cuda"] in jetpack_cuda_versions:
                        item["container_image"] = jetpack_container_image
                        filtered_includes.append(item)
        else:
            if item["gpu_arch_type"] == "cuda-aarch64":
                # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
                # however, TensorRT requires glibc2.31 on aarch64 platform
                # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
                item["container_image"] = sbsa_container_image
                filtered_includes.append(item)
            else:
                filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
