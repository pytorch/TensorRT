#!/usr/bin/env python3

import argparse
import json
import sys
from typing import List

disabled_python_versions: List[str] = ["3.13"]


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
            # for aarch64 only test cu128 for now
            if item["desired_cuda"] == "cu128":
                # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
                # however, TensorRT requires glibc2.31 on aarch64 platform
                # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
                item["container_image"] = "sameli/manylinux_2_34_aarch64_cuda_12.8"
                filtered_includes.append(item)
        else:
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
