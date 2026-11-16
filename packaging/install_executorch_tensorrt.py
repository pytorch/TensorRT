# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Install the TensorRT dependencies selected by the standard core wheel."""

import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> None:
    distribution = importlib.metadata.distribution("torch-tensorrt")
    requirements = [Requirement(value) for value in distribution.requires or []]
    selected = [
        str(requirement)
        for requirement in requirements
        if canonicalize_name(requirement.name)
        in {"tensorrt", "tensorrt-cu13", "tensorrt-cu13-bindings", "tensorrt-cu13-libs"}
        and (requirement.marker is None or requirement.marker.evaluate())
    ]
    if not selected:
        sys.exit("The main wheel declares no CUDA 13 TensorRT dependencies")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *selected])


if __name__ == "__main__":
    main()
