#!/usr/bin/env python3

import argparse
import json
import sys

RELEASE_CUDA_VERSION = ["cu124"]
RELEASE_PYTHON_VERSION = ["3.8", "3.9", "3.10", "3.11"]


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="test matrix",
        type=str,
        default="",
    )
    options = parser.parse_args(args)
    matrix_dict = json.loads(options.matrix)
    includes = matrix_dict["include"]
    filtered_includes = []
    for item in includes:
        if (
            item["desired_cuda"] in RELEASE_CUDA_VERSION
            and item["python_version"] in RELEASE_PYTHON_VERSION
        ):
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
