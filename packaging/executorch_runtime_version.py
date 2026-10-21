# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the version of the ExecuTorch runtime companion built beside a torch-tensorrt wheel.

The companion keeps its own base version and takes the wheel's channel suffix and CUDA label,
so 2.16.0.dev20261019+cu132 pairs with 0.3.0.dev20261019+cu132. A release wheel for PyPI has no
label, and neither does its companion. The build workflows export the result before building
either wheel, so the companion and the main wheel's requirement on it read the same value.

Usage: BUILD_VERSION=... CU_VERSION=... python packaging/executorch_runtime_version.py [true|false]
The argument says whether this is a release wheel for PyPI.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def companion_version(build_version: str, cuda: str, release_wheel: bool) -> str:
    main = (ROOT / "version.txt").read_text().strip().removesuffix("a0")
    base = (ROOT / "py/torch-tensorrt-executorch-runtime/version.txt").read_text()
    public = build_version.partition("+")[0]
    if not public.startswith(main):
        raise ValueError(f"Unexpected torch-tensorrt build version: {build_version}")
    label = "" if release_wheel else f"+{cuda}"
    return f"{base.strip()}{public[len(main):]}{label}"


if __name__ == "__main__":
    print(
        companion_version(
            os.environ["BUILD_VERSION"],
            os.environ["CU_VERSION"],
            sys.argv[1:] == ["true"],
        )
    )
