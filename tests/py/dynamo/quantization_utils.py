# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os


def quantization_available(package: str) -> bool:
    """Check optional test dependencies without importing their Torch extensions."""
    setting = (
        "INSTALL_MODELOPT_QUANTIZATION"
        if package == "modelopt"
        else "INSTALL_TORCHAO_QUANTIZATION"
    )
    return (
        os.environ.get(setting) != "0" and importlib.util.find_spec(package) is not None
    )
