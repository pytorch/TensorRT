#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List

# this was introduced to avoid build for py3.13, currently we support py3.13, but keep this for future use with other python versions
disabled_python_versions: List[str] = []


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
                item["python_version"] = "py3.11"
                build_names = item["build_name"].split("-")
                build_names[1] = "py3_11"
                item["build_name"] = "-".join(build_names)
                filtered_includes.append(item)
        else:
            filtered_includes.append(item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
