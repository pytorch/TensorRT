# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os


def quantization_available(package: str) -> bool:
    """Check optional test dependencies without importing their Torch extensions."""
    return (
        os.environ.get("INSTALL_QUANTIZATION") != "0"
        and importlib.util.find_spec(package) is not None
    )
