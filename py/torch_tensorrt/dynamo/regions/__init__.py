# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental explicit-region compiler support."""

from torch_tensorrt.region._errors import RegionError

from ._capture import (
    assert_no_unresolved_regions,
    is_region_hop,
    materialize_torch_regions,
    normalize_region_scopes,
    region_hop_records,
)
from ._types import (
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    get_region_records,
    is_torch_region_node,
    validate_region_settings,
)

__all__ = [
    "RegionError",
    "RegionRecord",
    "assert_no_unresolved_regions",
    "attach_region_records",
    "audit_region_placement",
    "get_region_records",
    "is_region_hop",
    "is_torch_region_node",
    "materialize_torch_regions",
    "normalize_region_scopes",
    "region_hop_records",
    "validate_region_settings",
]
