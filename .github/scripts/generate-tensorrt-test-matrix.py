#!/usr/bin/env python3

import argparse
import copy
import json
import sys

# please update the cuda version you want to test with the future tensorRT version here
# channel: nightly if the future tensorRT version test workflow is triggered from the main branch or your personal branch
# channel: test if the future tensorRT version test workflow is triggered from the release branch(release/2.5 etc....)
CUDA_VERSIONS_DICT = {
    "nightly": ["cu124"],
    "test": ["cu121", "cu124"],
    "release": ["cu121", "cu124"],
}

# please update the python version you want to test with the future tensorRT version here
# channel: nightly if the future tensorRT version test workflow is triggered from the main branch or your personal branch
# channel: test if the future tensorRT version test workflow is triggered from the release branch(release/2.5 etc....)
PYTHON_VERSIONS_DICT = {
    "nightly": ["3.9"],
    "test": ["3.9", "3.10", "3.11", "3.12"],
    "release": ["3.9", "3.10", "3.11", "3.12"],
}

# please update the future tensorRT version you want to test here
TENSORRT_VERSIONS_DICT = {
    "windows": {
        "10.4.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/zip/TensorRT-10.4.0.26.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.4.0.26",
            "sha256": "3a7de83778b9e9f812fd8901e07e0d7d6fc54ce633fcff2e340f994df2c6356c",
        },
        "10.5.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/zip/TensorRT-10.5.0.18.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.5.0.18",
            "sha256": "e6436f4164db4e44d727354dccf7d93755efb70d6fbfd6fa95bdfeb2e7331b24",
        },
        "10.6.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.6.0/zip/TensorRT-10.6.0.26.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.6.0.26",
            "sha256": "6c6d92c108a1b3368423e8f69f08d31269830f1e4c9da43b37ba34a176797254",
        },
        "10.7.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/zip/TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.7.0.23",
            "sha256": "fbdef004578e7ccd5ee51fe7f846b57422364a743372fd8f9f1d7dbd33f62879",
        },
    },
    "linux": {
        "10.4.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.4.0.26",
            "sha256": "cb0273ecb3ba4db8993a408eedd354712301a6c7f20704c52cdf9f78aa97bbdb",
        },
        "10.5.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.5.0.18",
            "sha256": "f404d379d639552a3e026cd5267213bd6df18a4eb899d6e47815bbdb34854958",
        },
        "10.6.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.6.0/tars/TensorRT-10.6.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.6.0.26",
            "sha256": "33d3c2f3f4c84dc7991a4337a6fde9ed33f5c8e5c4f03ac2eb6b994a382b03a0",
        },
        "10.7.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.7.0.23",
            "sha256": "d7f16520457caaf97ad8a7e94d802f89d77aedf9f361a255f2c216e2a3a40a11",
        },
    },
}


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix",
        type=str,
        default="",
    )

    options = parser.parse_args(args)
    if options.matrix == "":
        raise Exception("--matrix is empty, please provide the matrix json str")

    matrix_dict = json.loads(options.matrix)
    includes = matrix_dict["include"]
    assert len(includes) > 0
    if "channel" not in includes[0]:
        raise Exception(f"channel field is missing from the matrix: {options.matrix}")
    channel = includes[0]["channel"]
    if channel not in ("nightly", "test", "release"):
        raise Exception(
            f"channel field: {channel} is not supported, currently supported value: nightly, test, release"
        )

    if "validation_runner" not in includes[0]:
        raise Exception(
            f"validation_runner field is missing from the matrix: {options.matrix}"
        )
    if "windows" in includes[0]["validation_runner"]:
        arch = "windows"
    elif "linux" in includes[0]["validation_runner"]:
        arch = "linux"
    else:
        raise Exception(
            f"{includes[0].validation_runner} is not the supported arch, currently only support windows and linux"
        )

    cuda_versions = CUDA_VERSIONS_DICT[channel]
    python_versions = PYTHON_VERSIONS_DICT[channel]
    tensorrt_versions = TENSORRT_VERSIONS_DICT[arch]

    filtered_includes = []
    for item in includes:
        if (
            item["desired_cuda"] in cuda_versions
            and item["python_version"] in python_versions
        ):
            for tensorrt_version, tensorrt_json in tensorrt_versions.items():
                new_item = copy.deepcopy(item)
                tensorrt_json["version"] = tensorrt_version
                new_item["tensorrt"] = tensorrt_json
                filtered_includes.append(new_item)
    filtered_matrix_dict = {}
    filtered_matrix_dict["include"] = filtered_includes
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
