#!/usr/bin/env python3

import argparse
import json
import sys
from typing import List

disabled_python_versions: List[str] = []


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
        if item["python_version"] not in disabled_python_versions:
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
