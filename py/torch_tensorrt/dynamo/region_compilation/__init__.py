# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental explicit-region compiler support."""

from torch_tensorrt.region._errors import RegionError

from ._capture import (
    assert_no_unresolved_regions,
    check_decomposed_regions,
    has_region_markers,
    is_region_hop,
    materialize_regions,
    normalize_region_scopes,
    normalize_traced_scopes,
    region_hop_records,
    validate_decomposed_bodies,
)
from ._runtime import configure_region_backends
from ._selection import (
    bind_rules,
    find_compiled_model,
    resolve_rules,
    select_exported_regions,
    select_traced_regions,
)
from ._types import (
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    get_region_records,
    is_torch_region_node,
    validate_region_settings,
)
from ._validation import (
    validate_eval_mode,
    validate_model_mode,
    validate_region_mutations,
)

__all__ = [
    "RegionError",
    "RegionRecord",
    "assert_no_unresolved_regions",
    "attach_region_records",
    "bind_rules",
    "find_compiled_model",
    "audit_region_placement",
    "check_decomposed_regions",
    "configure_region_backends",
    "get_region_records",
    "has_region_markers",
    "is_region_hop",
    "is_torch_region_node",
    "materialize_regions",
    "normalize_region_scopes",
    "normalize_traced_scopes",
    "region_hop_records",
    "resolve_rules",
    "select_exported_regions",
    "select_traced_regions",
    "validate_decomposed_bodies",
    "validate_eval_mode",
    "validate_model_mode",
    "validate_region_mutations",
    "validate_region_settings",
]
