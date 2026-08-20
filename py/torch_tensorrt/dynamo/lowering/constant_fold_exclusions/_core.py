from typing import Callable, Collection, Iterable

import torch

CONSTANT_FOLD_EXCLUSION_META_KEY = "_torch_tensorrt_constant_fold_exclusions"

ConstantFoldExclusionRule = Callable[[torch.fx.Node], Iterable[torch.fx.Node]]
_CONSTANT_FOLD_EXCLUSION_RULES: dict[str, ConstantFoldExclusionRule] = {}


def _mark_constant_fold_exclusion(nodes: Iterable[torch.fx.Node], rule_id: str) -> None:
    """Record which rule wants ``nodes`` kept out of constant folding.

    The marks carry their rule ID rather than a bare flag so the exclusion pass
    can revoke the ones belonging to disabled rules, whichever marking path
    produced them.
    """
    for node in nodes:
        node.meta.setdefault(CONSTANT_FOLD_EXCLUSION_META_KEY, set()).add(rule_id)


def register_constant_fold_exclusion_rule(
    rule_id: str,
) -> Callable[[ConstantFoldExclusionRule], ConstantFoldExclusionRule]:
    """Register a named rule that selects FX nodes to exclude from folding."""
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("A constant-fold exclusion rule ID must be a non-empty string")

    def register(rule: ConstantFoldExclusionRule) -> ConstantFoldExclusionRule:
        if rule_id in _CONSTANT_FOLD_EXCLUSION_RULES:
            raise ValueError(
                f"Constant-fold exclusion rule {rule_id!r} is already registered"
            )

        _CONSTANT_FOLD_EXCLUSION_RULES[rule_id] = rule
        return rule

    return register


def validate_disabled_constant_fold_exclusions(
    rule_ids: Collection[str],
) -> set[str]:
    """Validate disabled rule IDs and return them as a set."""
    if isinstance(rule_ids, str):
        raise TypeError(
            "disabled_constant_fold_exclusions must be a collection of rule IDs, "
            "not a single string"
        )

    disabled_rule_ids = set(rule_ids)
    unknown_rule_ids = disabled_rule_ids - _CONSTANT_FOLD_EXCLUSION_RULES.keys()
    if unknown_rule_ids:
        available_rule_ids = ", ".join(sorted(_CONSTANT_FOLD_EXCLUSION_RULES))
        raise ValueError(
            "Unknown constant-fold exclusion rule IDs: "
            f"{sorted(unknown_rule_ids)}. Available rule IDs: "
            f"[{available_rule_ids}]"
        )

    return disabled_rule_ids
