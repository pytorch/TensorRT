#!/usr/bin/env python3

import argparse
import json
import sys

RELEASE_CUDA_VERSION = {
    "wheel": ["cu128"],
    "tarball": ["cu128"],
}
RELEASE_PYTHON_VERSION = {
    "wheel": ["3.9", "3.10", "3.11", "3.12"],
    "tarball": ["3.11"],
}

CXX11_TARBALL_CONTAINER_IMAGE = {
    "cu128": "pytorch/libtorch-cxx11-builder:cuda12.8-main",
}


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel_matrix",
        help="wheel matrix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tarball_matrix",
        help="tarball matrix",
        type=str,
        default="",
    )
    options = parser.parse_args(args)
    cuda_versions = []
    python_versions = []

    if options.tarball_matrix != "":
        cuda_versions = RELEASE_CUDA_VERSION["tarball"]
        python_versions = RELEASE_PYTHON_VERSION["tarball"]
        matrix_dict = json.loads(options.tarball_matrix)
    elif options.wheel_matrix != "":
        cuda_versions = RELEASE_CUDA_VERSION["wheel"]
        python_versions = RELEASE_PYTHON_VERSION["wheel"]
        matrix_dict = json.loads(options.wheel_matrix)
    else:
        raise Exception(
            "Either --wheel_matrix or --tarball_matrix needs to be provided"
        )

    includes = matrix_dict["include"]
    filtered_includes = []
    for item in includes:
        if (
            item["desired_cuda"] in cuda_versions
            and item["python_version"] in python_versions
        ):
            if options.tarball_matrix != "":
                item["container_image"] = CXX11_TARBALL_CONTAINER_IMAGE[
                    item["desired_cuda"]
                ]
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
