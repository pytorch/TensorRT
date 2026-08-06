from typing import Any, Optional

import torch
from torch_tensorrt.dynamo.lowering.constant_fold_exclusions._core import (
    CONSTANT_FOLD_EXCLUSION_META_KEY,
    _CONSTANT_FOLD_EXCLUSION_RULES,
    _mark_constant_fold_exclusion,
    validate_disabled_constant_fold_exclusions,
)


def mark_constant_fold_exclusions(
    gm: torch.fx.GraphModule, settings: Optional[Any] = None
) -> torch.fx.GraphModule:
    """Apply the registered rules that exclude FX nodes from constant folding.

    This pass is the single authority on which rules are in effect. It runs
    immediately before ``constant_fold`` and is the only marking path that sees
    ``settings``: rules that mark nodes while a decomposition is traced run
    during ``run_decompositions``, long before a settings object is reachable.
    Those marks are therefore revoked here rather than suppressed where they are
    made, so a caller only has to communicate the disabled rules once.
    """
    disabled_rule_ids = validate_disabled_constant_fold_exclusions(
        settings.disabled_constant_fold_exclusions if settings is not None else ()
    )

    for node in gm.graph.nodes:
        for rule_id, rule in _CONSTANT_FOLD_EXCLUSION_RULES.items():
            if rule_id in disabled_rule_ids:
                continue
            _mark_constant_fold_exclusion(rule(node), rule_id)

    if disabled_rule_ids:
        for node in gm.graph.nodes:
            marking_rule_ids = node.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY)
            if marking_rule_ids:
                marking_rule_ids -= disabled_rule_ids

    return gm
