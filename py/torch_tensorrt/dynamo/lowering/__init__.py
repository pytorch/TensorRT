# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch_tensorrt.dynamo.lowering._SubgraphBuilder import SubgraphBuilder

from ._decomposition_groups import (
    TORCH_TRT_DECOMPOSITIONS,
    torch_disabled_decompositions,
    torch_enabled_decompositions,
)
from ._decompositions import get_decompositions  # noqa: F401
from .passes import *
