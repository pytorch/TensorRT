#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List

# currently we don't support python 3.13t due to tensorrt does not support 3.13t
disabled_python_versions: List[str] = ["3.13t"]


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix blob",
        type=str,
        default="",
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
        if item["gpu_arch_type"] == "cuda-aarch64":
            # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
            # however, TensorRT requires glibc2.31 on aarch64 platform
            # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
            item["container_image"] = "quay.io/pypa/manylinux_2_34_aarch64"
            filtered_includes.append(item)
        else:
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
