#!/usr/bin/env python3

"""Generate test modules list to be utilized through gihub actions

Will output a list of modules needs run test cases:
*
"""

import argparse
import os
import sys
from typing import List

import regex

MAX_FOLDER_DEPTH = 3

# TODO: discuss with Naren for a better mapping
Folder_To_TestModules_Dict = {
    "py/torch_tensorrt/dynamo": [
        "dynamo_frontend",
        "dynamo_core",
        "dynamo_converter",
        "dynamo_serde",
        "torch_compile_backend",
    ],
    "py/torch_tensorrt/ts": ["ts_frontend"],
    "py/torch_tensorrt": ["py_core"],
}

# The following folder files change will trigger the build but will skip all the test modules
Folder_To_Skip_TestModules = {
    "docker/*",
    "docs/*",
    "docsrc/*",
    "examples/*",
    "notenooks/*",
    "packaging/*",
    "third_party/*",
    "toolchains/*",
    "tools/*",
}

# TODO: discuss with Naren for a basic set of tests here
# this is just an example only
Base_Test_Modules = ["torch_compile_backend"]
Full_Test_Modules = [
    "ts_frontend",
    "py_core",
    "dynamo_frontend",
    "dynamo_core",
    "dynamo_converter",
    "dynamo_serde",
    "torch_compile_backend",
]


def filter_files_to_folders(
    files: str,
) -> List[str]:
    fileList = files.split(" ")
    folders = []

    for file in fileList:
        if file == "":
            continue
        filePath = os.path.dirname(file)
        splits = filePath.split("/", MAX_FOLDER_DEPTH)
        if len(splits) > MAX_FOLDER_DEPTH:
            splits = splits[: len(splits) - 1]
        folder = "/".join(splits)
        folders.append(folder)

    ret = list(dict.fromkeys(folders))
    return ret


def generate_test_modules(
    folders: List[str],
) -> List[str]:
    testModules = []
    for folder in folders:
        skip = False
        for to_skip in Folder_To_Skip_TestModules:
            if regex.match(to_skip, folder):
                skip = True
                break
        if skip == True:
            continue
        if folder in Folder_To_TestModules_Dict.keys():
            modules = Folder_To_TestModules_Dict[folder]
            testModules.extend(modules)
        else:
            # if there is files changed in other folders, always run the base tests
            testModules.extend(Base_Test_Modules)
    return list(dict.fromkeys(testModules))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-name",
        help="",
        type=str,
        default="",
    )
    parser.add_argument(
        "--files",
        help="",
        type=str,
        default="",
    )
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    options = parse_args()
    assert options.event_name != "", "Must provide the --event-name str"

    # if it is the Pull Request:
    # for the pull request, it will run the full tests of the test module if the related files changed
    # else only run the l0 testcases
    if options.event_name == "pull_request":
        assert (
            options.files != ""
        ), "Must provide the --files str, if the --event-name is pull_request"
        folders = filter_files_to_folders(options.files)
        test_modules = generate_test_modules(folders)
    else:
        # for the non pull_request, always run the full tests of all the test modules
        test_modules = Full_Test_Modules

    print(test_modules)


if __name__ == "__main__":
    main()
