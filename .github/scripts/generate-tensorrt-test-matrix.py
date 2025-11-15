#!/usr/bin/env python3

import argparse
import copy
import json
import sys

import requests  # type: ignore[import-untyped]

# please update the cuda version you want to test with the future tensorRT version here
# channel: nightly if the future tensorRT version test workflow is triggered from the main branch or your personal branch
# channel: test if the future tensorRT version test workflow is triggered from the release branch(release/2.5 etc....)
CUDA_VERSIONS_DICT = {
    "nightly": ["cu130"],
    "test": ["cu126", "cu128", "cu130"],
    "release": ["cu126", "cu128", "cu130"],
}

# please update the python version you want to test with the future tensorRT version here
# channel: nightly if the future tensorRT version test workflow is triggered from the main branch or your personal branch
# channel: test if the future tensorRT version test workflow is triggered from the release branch(release/2.5 etc....)
PYTHON_VERSIONS_DICT = {
    "nightly": ["3.11"],
    "test": ["3.10", "3.11", "3.12", "3.13"],
    "release": ["3.10", "3.11", "3.12", "3.13"],
}

# please update the future tensorRT version you want to test here
TENSORRT_VERSIONS_DICT = {
    "windows": {
        "10.3.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/zip/TensorRT-10.3.0.26.Windows.win10.cuda-12.5.zip",
            "strip_prefix": "TensorRT-10.3.0.26",
        },
        "10.7.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/zip/TensorRT-10.7.0.23.Windows.win10.cuda-12.6.zip",
            "strip_prefix": "TensorRT-10.7.0.23",
        },
        "10.8.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/zip/TensorRT-10.8.0.43.Windows.win10.cuda-12.8.zip",
            "strip_prefix": "TensorRT-10.8.0.43",
        },
        "10.9.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/zip/TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip",
            "strip_prefix": "TensorRT-10.9.0.34",
        },
        "10.10.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/zip/TensorRT-10.10.0.31.Windows.win10.cuda-12.9.zip",
            "strip_prefix": "TensorRT-10.10.0.31",
        },
        "10.11.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/zip/TensorRT-10.11.0.33.Windows.win10.cuda-12.9.zip",
            "strip_prefix": "TensorRT-10.11.0.33",
        },
        "10.12.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.12.0/zip/TensorRT-10.12.0.36.Windows.win10.cuda-12.9.zip",
            "strip_prefix": "TensorRT-10.12.0.36",
        },
        "10.13.2": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.2/zip/TensorRT-10.13.2.6.Windows.win10.cuda-12.9.zip",
            "strip_prefix": "TensorRT-10.13.2.6",
        },
        "10.14.1": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/zip/TensorRT-10.14.1.48.Windows.win10.cuda-12.9.zip",
            "strip_prefix": "TensorRT-10.14.1.48",
        },
    },
    "linux": {
        "10.3.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz",
            "strip_prefix": "TensorRT-10.3.0.26",
        },
        "10.7.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz",
            "strip_prefix": "TensorRT-10.7.0.23",
        },
        "10.8.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz",
            "strip_prefix": "TensorRT-10.8.0.43",
        },
        "10.9.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz",
            "strip_prefix": "TensorRT-10.9.0.34",
        },
        "10.10.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/tars/TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz",
            "strip_prefix": "TensorRT-10.10.0.31",
        },
        "10.11.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/tars/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz",
            "strip_prefix": "TensorRT-10.11.0.33",
        },
        "10.12.0": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.12.0/tars/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9.tar.gz",
            "strip_prefix": "TensorRT-10.12.0.36",
        },
        "10.13.2": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.2/tars/TensorRT-10.13.2.6.Linux.x86_64-gnu.cuda-12.9.tar.gz",
            "strip_prefix": "TensorRT-10.13.2.6",
        },
        "10.14.1": {
            "urls": "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/tars/TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz",
            "strip_prefix": "TensorRT-10.14.1.48",
        },
    },
}


def check_new_tensorrt_version(
    major: int = 10, patch_from: int = 0
) -> tuple[bool, str, str, str, str]:

    def check_file_availability(url: str) -> bool:
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                content_disposition = response.headers.get("Content-Disposition", "")
                if "application" in content_type or "attachment" in content_disposition:
                    return True
            return False
        except requests.RequestException:
            return False

    trt_linux_release_url = ""
    trt_win_release_url = ""

    # calculate the next minor version
    minor = int(list(TENSORRT_VERSIONS_DICT["linux"].keys())[-1].split(".")[1]) + 1
    trt_version = f"{major}.{minor}.0"
    for patch in range(patch_from, 80):
        for cuda_minor in range(4, 11):
            trt_linux_release_url_candidate = f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{trt_version}/tars/TensorRT-{trt_version}.{patch}.Linux.x86_64-gnu.cuda-12.{cuda_minor}.tar.gz"
            if check_file_availability(trt_linux_release_url_candidate):
                trt_linux_release_url = trt_linux_release_url_candidate
                trt_win_release_url = f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{trt_version}/zip/TensorRT-{trt_version}.{patch}.Windows.win10.cuda-12.{cuda_minor}.zip"
                return (
                    True,
                    trt_version,
                    str(patch),
                    trt_linux_release_url,
                    trt_win_release_url,
                )
    return False, "", "", "", ""


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

    (
        new_trt_available,
        trt_version,
        trt_patch,
        trt_linux_release_url,
        trt_win_release_url,
    ) = check_new_tensorrt_version(major=10, patch_from=0)
    if new_trt_available:
        TENSORRT_VERSIONS_DICT["linux"][trt_version] = {}
        TENSORRT_VERSIONS_DICT["linux"][trt_version]["urls"] = trt_linux_release_url
        TENSORRT_VERSIONS_DICT["linux"][trt_version][
            "strip_prefix"
        ] = f"TensorRT-{trt_version}.{trt_patch}"
        TENSORRT_VERSIONS_DICT["windows"][trt_version] = {}
        TENSORRT_VERSIONS_DICT["windows"][trt_version]["urls"] = trt_win_release_url
        TENSORRT_VERSIONS_DICT["windows"][trt_version][
            "strip_prefix"
        ] = f"TensorRT-{trt_version}.{trt_patch}"

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
