#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# currently we don't support python 3.13t due to tensorrt does not support 3.13t
disabled_python_versions: List[str] = ["3.13t", "3.14", "3.14t"]
disabled_cuda_versions: List[str] = []

# jetpack 6.2 only officially supports python 3.10 and cu126
jetpack_python_versions: List[str] = ["3.10"]
jetpack_cuda_versions: List[str] = ["cu126"]

jetpack_container_image: str = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
sbsa_container_image: str = "quay.io/pypa/manylinux_2_39_aarch64"


def validate_matrix(matrix_dict: Dict[str, Any]) -> None:
    """Validate the structure of the input matrix."""
    if not isinstance(matrix_dict, dict):
        raise ValueError("Matrix must be a dictionary")
    if "include" not in matrix_dict:
        raise ValueError("Matrix must contain 'include' key")
    if not isinstance(matrix_dict["include"], list):
        raise ValueError("Matrix 'include' must be a list")


def filter_matrix_item(
    item: Dict[str, Any],
    is_jetpack: bool,
    limit_pr_builds: bool,
) -> bool:
    """Filter a single matrix item based on the build type and requirements."""
    if item["python_version"] in disabled_python_versions:
        # Skipping disabled Python version
        return False
    if item["desired_cuda"] in disabled_cuda_versions:
        # Skipping disabled CUDA version
        return False
    if is_jetpack:
        # pr build,matrix passed from test-infra is cu126,cu128 and cu130, python 3.10, filter to cu126, python 3.10
        # nightly/main build, matrix passed from test-infra is cu126, cu128 and cu130, all python versions, filter to cu126, python 3.10
        if (
            item["python_version"] in jetpack_python_versions
            and item["desired_cuda"] in jetpack_cuda_versions
        ):
            item["container_image"] = jetpack_container_image
            return True
        return False
    else:
        if item["gpu_arch_type"] == "cuda-aarch64":
            # pytorch image:pytorch/manylinuxaarch64-builder:cuda12.8 comes with glibc2.28
            # however, TensorRT requires glibc2.31 on aarch64 platform
            # TODO: in future, if pytorch supports aarch64 with glibc2.31, we should switch to use the pytorch image
            item["container_image"] = sbsa_container_image
            return True
        return True


def create_distributed_config(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create distributed test configuration from a regular config.

    Takes a standard test config and modifies it for distributed testing:
    - Changes runner to multi-GPU instance
    - Adds num_gpus field
    - Adds config marker
    """
    import sys

    # Create a copy to avoid modifying the original
    dist_item = item.copy()

    # Debug: Show original config
    print(f"[DEBUG] Creating distributed config from:", file=sys.stderr)
    print(f"[DEBUG]   Python: {item.get('python_version')}", file=sys.stderr)
    print(f"[DEBUG]   CUDA: {item.get('desired_cuda')}", file=sys.stderr)
    print(
        f"[DEBUG]   Original runner: {item.get('validation_runner')}", file=sys.stderr
    )

    # Override runner to use multi-GPU instance
    dist_item["validation_runner"] = "linux.g4dn.12xlarge.nvidia.gpu"

    # Add distributed-specific fields
    dist_item["num_gpus"] = 2
    dist_item["config"] = "distributed"

    # Debug: Show modified config
    print(f"[DEBUG]   New runner: {dist_item['validation_runner']}", file=sys.stderr)
    print(f"[DEBUG]   GPUs: {dist_item['num_gpus']}", file=sys.stderr)

    return dist_item


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        help="matrix blob",
        type=str,
        default="",
    )

    parser.add_argument(
        "--jetpack",
        help="is jetpack",
        type=str,
        choices=["true", "false"],
        default="false",
    )

    parser.add_argument(
        "--limit-pr-builds",
        help="If it is a PR build",
        type=str,
        choices=["true", "false"],
        default=os.getenv("LIMIT_PR_BUILDS", "false"),
    )

    options = parser.parse_args(args)
    if options.matrix == "":
        raise ValueError("--matrix needs to be provided")

    try:
        matrix_dict = json.loads(options.matrix)
        validate_matrix(matrix_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in matrix: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid matrix structure: {e}")

    includes = matrix_dict["include"]
    filtered_includes = []
    distributed_includes = []  # NEW: separate list for distributed configs

    print(f"[DEBUG] Processing {len(includes)} input configs", file=sys.stderr)

    for item in includes:
        py_ver = item.get("python_version", "unknown")
        cuda_ver = item.get("desired_cuda", "unknown")

        print(f"[DEBUG] Checking config: py={py_ver}, cuda={cuda_ver}", file=sys.stderr)

        if filter_matrix_item(
            item,
            options.jetpack == "true",
            options.limit_pr_builds == "true",
        ):
            print(f"[DEBUG] passed filter - adding to build matrix", file=sys.stderr)
            filtered_includes.append(item) 
            distributed_includes.append(create_distributed_config(item))
        else:
            print(f"[DEBUG] FILTERED OUT", file=sys.stderr)

    # Debug: Show summary
    print(f"[DEBUG] Final counts:", file=sys.stderr)
    print(f"[DEBUG]   Regular configs: {len(filtered_includes)}", file=sys.stderr)
    print(
        f"[DEBUG]   Distributed configs: {len(distributed_includes)}", file=sys.stderr
    )

    # Debug: Show which configs will be built
    print(
        f"[DEBUG] Configs that will be BUILT (in filtered_includes):", file=sys.stderr
    )
    for item in filtered_includes:
        print(
            f"[DEBUG]   - py={item.get('python_version')}, cuda={item.get('desired_cuda')}",
            file=sys.stderr,
        )

    print(
        f"[DEBUG] Configs for DISTRIBUTED TESTS (in distributed_includes):",
        file=sys.stderr,
    )
    for item in distributed_includes:
        print(
            f"[DEBUG]   - py={item.get('python_version')}, cuda={item.get('desired_cuda')}, gpus={item.get('num_gpus')}",
            file=sys.stderr,
        )

    # NEW: Output both regular and distributed configs
    filtered_matrix_dict = {
        "include": filtered_includes,
        "distributed_include": distributed_includes,  # NEW field
    }

    # Output to stdout (consumed by GitHub Actions)
    print(json.dumps(filtered_matrix_dict))


if __name__ == "__main__":
    main(sys.argv[1:])
