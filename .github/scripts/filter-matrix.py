#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# this was introduced to avoid build for py3.13, currently we support py3.13, but keep this for future use with other python versions
disabled_python_versions: List[str] = []


def replace_aarch64_container_image(item: Dict[str, Any]) -> Dict[str, Any] | None:
    if item["gpu_arch_type"] == "cuda-aarch64":
        if item["desired_cuda"] == "cu128":
            # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
            # however, TensorRT requires glibc2.31 on aarch64 platform
            # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
            # item["container_image"] = "sameli/manylinux_2_34_aarch64_cuda_12.8"
            item["container_image"] = "quay.io/pypa/manylinux_2_34_aarch64"
            return item
        else:
            # for aarch64 do not test on other cuda versions
            return None
    # if it is not aarch64, do nothing and return the item
    else:
        return item


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix blob",
        type=str,
        default="",
    )

    parser.add_argument(
        "--limit-pr-builds",
        help="Limit PR builds to single python/cuda config(py3.11/cu12.8): true or false",
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
        if options.limit_pr_builds == "true":
            # currently if it is the pr build, it build using py3.9 with all cuda versions, we want to change to py3.11 with singlecu12.8
            if item["desired_cuda"] == "cu128":
                item["python_version"] = "3.11"
                build_names = item["build_name"].split("-")
                build_names[1] = "py3_11"
                item["build_name"] = "-".join(build_names)
                item = replace_aarch64_container_image(item)
                filtered_includes.append(item)
        else:
            item = replace_aarch64_container_image(item)
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
