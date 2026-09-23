# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from ._core import (
    CONSTANT_FOLD_EXCLUSION_META_KEY,
    ConstantFoldExclusionRule,
    register_constant_fold_exclusion_rule,
    unregister_constant_fold_exclusion_rule,
    validate_disabled_constant_fold_exclusions,
)
from .attention_mask import ATTENTION_MASK_ARANGE_RULE_ID

__all__ = [
    "ATTENTION_MASK_ARANGE_RULE_ID",
    "CONSTANT_FOLD_EXCLUSION_META_KEY",
    "ConstantFoldExclusionRule",
    "register_constant_fold_exclusion_rule",
    "unregister_constant_fold_exclusion_rule",
    "validate_disabled_constant_fold_exclusions",
]
