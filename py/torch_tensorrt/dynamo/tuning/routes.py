"""Build-route expression parsing and expansion (trtexec sampleTuning parity)."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_BRACKET_VALUES_RE = re.compile(r"\[([^\]]*)\]")


@dataclass
class BuildRouteKnobDef:
    option: str
    allowed_values: str
    default_value: str
    help: str
    values: List[str] = field(default_factory=list)
    is_bounded: bool = False


@dataclass
class BuildRouteParsedExpr:
    knob_name: str
    values: List[str]
    is_fixed: bool = False


class BuildRouteKnobDatabase:
    """Knob definitions loaded from ``IBuilderConfig.all_build_routes`` JSON."""

    def __init__(self) -> None:
        self._knobs: Dict[str, BuildRouteKnobDef] = {}
        self._knob_order: List[str] = []
        self.tuner_version: str = "unknown"

    def load_from_json(self, json_str: str) -> bool:
        self._knobs.clear()
        self._knob_order.clear()
        self.tuner_version = "unknown"
        if not json_str or not json_str.strip():
            return False
        try:
            root = json.loads(json_str)
        except json.JSONDecodeError:
            return False

        if isinstance(root.get("tuner_version"), str):
            self.tuner_version = root["tuner_version"]

        options = root.get("tuner_options")
        if not isinstance(options, list):
            return False

        for item in options:
            if not isinstance(item, dict):
                continue
            option = item.get("option", "")
            if not option:
                continue
            allowed = item.get("allowed_values", "") or ""
            values = self.parse_allowed_values(allowed)
            knob = BuildRouteKnobDef(
                option=option,
                allowed_values=allowed,
                default_value=str(item.get("default_value", "")),
                help=str(item.get("help", "")),
                values=values,
                is_bounded=bool(values),
            )
            self._knob_order.append(option)
            self._knobs[option] = knob
        return bool(self._knobs)

    @staticmethod
    def parse_allowed_values(allowed_str: str) -> List[str]:
        match = _BRACKET_VALUES_RE.search(allowed_str)
        if not match:
            return []
        values_str = match.group(1)
        if "..." in values_str:  # "..." is open-ended range marker
            return []
        return [v.strip() for v in values_str.split("|") if v.strip()]

    def has_knob(self, knob_name: str) -> bool:
        return knob_name in self._knobs

    def get_knob(self, knob_name: str) -> Optional[BuildRouteKnobDef]:
        return self._knobs.get(knob_name)

    def get_default_value(self, knob_name: str) -> str:
        knob = self.get_knob(knob_name)
        return knob.default_value if knob is not None else ""

    def validate_values(self, knob_name: str, values: Sequence[str]) -> bool:
        knob = self.get_knob(knob_name)
        if knob is None:
            return False
        if not knob.is_bounded:
            return True
        allowed = set(knob.values)
        return all(v in allowed for v in values)

    def build_default_path(self) -> str:
        parts = []
        for name in self._knob_order:
            knob = self._knobs[name]
            parts.append(f"{knob.option}={knob.default_value}")
        return " ".join(parts)


class BuildRouteExprParser:
    """Parse ``-knob=[a|b] -fixed=on`` expressions against a knob database."""

    def __init__(self, db: BuildRouteKnobDatabase) -> None:
        self._db = db
        self.error: str = ""

    def parse(self, input_str: str) -> Optional[List[BuildRouteParsedExpr]]:
        self.error = ""
        if not input_str or not input_str.strip():
            self.error = "Empty input"
            return None
        tokens = self.tokenize(input_str)
        if not tokens:
            self.error = "No expressions found"
            return None
        result: List[BuildRouteParsedExpr] = []
        for token in tokens:
            expr = self._parse_expr(token)
            if expr is None:
                return None
            result.append(expr)
        return result

    @staticmethod
    def tokenize(input_str: str) -> List[str]:
        tokens: List[str] = []
        current: List[str] = []
        bracket_depth = 0
        for c in input_str:
            if c == "[":
                bracket_depth += 1
                current.append(c)
            elif c == "]":
                bracket_depth -= 1
                current.append(c)
            elif c == " " and bracket_depth == 0:
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(c)
        if current:
            tokens.append("".join(current))
        return tokens

    def _parse_expr(self, expr: str) -> Optional[BuildRouteParsedExpr]:
        eq_pos = expr.find("=")
        if eq_pos < 0:
            self.error = f"Invalid expression (no '='): {expr}"
            return None
        knob_name = expr[:eq_pos].strip()
        if not self._db.has_knob(knob_name):
            self.error = f"Unknown knob: {knob_name}"
            return None
        value_str = expr[eq_pos + 1 :].strip()
        if value_str.startswith("[") and value_str.endswith("]"):
            inner = value_str[1:-1]
            values = [v.strip() for v in inner.split("|") if v.strip()]
            if not values:
                self.error = f"Empty value list for knob: {knob_name}"
                return None
            if not self._db.validate_values(knob_name, values):
                self.error = f"Invalid values for knob {knob_name}: {values}"
                return None
            return BuildRouteParsedExpr(
                knob_name=knob_name, values=values, is_fixed=False
            )
        if not self._db.validate_values(knob_name, [value_str]):
            # Fixed values may still be legal defaults; allow if knob unbounded
            knob = self._db.get_knob(knob_name)
            if knob is None or (knob.is_bounded and value_str not in knob.values):
                self.error = f"Invalid fixed value for knob {knob_name}: {value_str}"
                return None
        return BuildRouteParsedExpr(
            knob_name=knob_name, values=[value_str], is_fixed=True
        )


def _route_from_assignment(names: Sequence[str], values: Sequence[str]) -> str:
    return " ".join(f"{n}={v}" for n, v in zip(names, values))


def expand_routes_full(exprs: Sequence[BuildRouteParsedExpr]) -> List[str]:
    """Cartesian product over variable knobs (trtexec ``full``)."""
    names = [e.knob_name for e in exprs]
    value_lists = [e.values for e in exprs]
    return [
        _route_from_assignment(names, combo)
        for combo in itertools.product(*value_lists)
    ]


def expand_routes_fast(
    exprs: Sequence[BuildRouteParsedExpr], db: BuildRouteKnobDatabase
) -> List[str]:
    """Baseline (defaults) + one-knob-at-a-time variants (trtexec ``fast``)."""
    names = [e.knob_name for e in exprs]
    defaults: List[str] = []
    for e in exprs:
        if e.is_fixed:
            defaults.append(e.values[0])
        else:
            default = db.get_default_value(e.knob_name)
            if default in e.values:
                defaults.append(default)
            else:
                defaults.append(e.values[0])

    routes = [_route_from_assignment(names, defaults)]
    seen = set(routes)

    for i, e in enumerate(exprs):
        if e.is_fixed:
            continue
        for val in e.values:
            if val == defaults[i]:
                continue
            combo = list(defaults)
            combo[i] = val
            route = _route_from_assignment(names, combo)
            if route not in seen:
                seen.add(route)
                routes.append(route)
    return routes


def expand_routes_mixed(
    exprs: Sequence[BuildRouteParsedExpr],
    db: BuildRouteKnobDatabase,
    positive_knob_indices: Sequence[int],
) -> List[str]:
    """Exhaustive sweep over knobs that improved latency in the fast phase."""
    names = [e.knob_name for e in exprs]
    defaults: List[str] = []
    for e in exprs:
        if e.is_fixed:
            defaults.append(e.values[0])
        else:
            default = db.get_default_value(e.knob_name)
            if default in e.values:
                defaults.append(default)
            else:
                defaults.append(e.values[0])

    positive = set(positive_knob_indices)
    value_lists: List[List[str]] = []
    for i, e in enumerate(exprs):
        if e.is_fixed or i not in positive:
            value_lists.append([defaults[i]])
        else:
            value_lists.append(list(e.values))

    routes = [
        _route_from_assignment(names, combo)
        for combo in itertools.product(*value_lists)
    ]
    # Prefer stable unique order
    seen = set()
    unique: List[str] = []
    for r in routes:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def load_tuning_expr_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return " ".join(line for line in lines if line)


def resolve_tuning_expression(
    tune_build_routes: str = "",
    tune_build_route_file: Optional[str] = None,
) -> str:
    if tune_build_routes and tune_build_route_file:
        raise ValueError(
            "Cannot specify both tune_build_routes and tune_build_route_file."
        )
    if tune_build_route_file:
        return load_tuning_expr_from_file(tune_build_route_file)
    return tune_build_routes or ""


def expand_build_routes(
    expression: str,
    search: str,
    db: BuildRouteKnobDatabase,
    *,
    dry_run: bool = False,
) -> Tuple[List[BuildRouteParsedExpr], List[str]]:
    """Parse and expand a tuning expression.

    Returns:
        (parsed_exprs, route_strings). For ``mixed``, only the fast phase is
        expanded here; the sweeper runs the second phase after measuring.
    """
    search = search.lower()
    if search not in {"fast", "full", "mixed"}:
        raise ValueError(
            f"Unknown tuning_search={search}; expected 'fast', 'full', or 'mixed'."
        )
    if dry_run and search == "mixed":
        raise ValueError("tuning_dry_run is incompatible with tuning_search='mixed'.")

    parser = BuildRouteExprParser(db)
    exprs = parser.parse(expression)
    if exprs is None:
        raise ValueError(f"Failed to parse tune_build_routes: {parser.error}")

    if search == "full":
        return exprs, expand_routes_full(exprs)
    # fast and mixed phase-1 use the same expansion
    return exprs, expand_routes_fast(exprs, db)


def identify_positive_knobs(
    exprs: Sequence[BuildRouteParsedExpr],
    gpu_times: Sequence[Optional[float]],
    db: BuildRouteKnobDatabase,
) -> List[int]:
    """Return indices of knobs whose one-off variants beat the baseline."""
    if not gpu_times or gpu_times[0] is None:
        return []
    baseline = gpu_times[0]
    defaults: List[str] = []
    for e in exprs:
        if e.is_fixed:
            defaults.append(e.values[0])
        else:
            default = db.get_default_value(e.knob_name)
            if default in e.values:
                defaults.append(default)
            else:
                defaults.append(e.values[0])

    positive: List[int] = []
    idx = 1
    for i, e in enumerate(exprs):
        if e.is_fixed:
            continue
        for val in e.values:
            if val == defaults[i]:
                continue
            if idx >= len(gpu_times):
                return positive
            t = gpu_times[idx]
            if t is not None and t < baseline and i not in positive:
                positive.append(i)
            idx += 1
    return positive
