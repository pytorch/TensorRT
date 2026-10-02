# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch_tensorrt.dynamo.conversion.plugins import (  # noqa: F401
    _auto_functionalized_converter,
)
from torch_tensorrt.dynamo.conversion.plugins._custom_op import custom_op
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import generate_plugin
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)

__all__ = ["custom_op", "generate_plugin", "generate_plugin_converter"]
