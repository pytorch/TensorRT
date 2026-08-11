"""Global Performance Tuner helpers for Torch-TensorRT Dynamo."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import tensorrt as trt

_LOGGER = logging.getLogger(__name__)


def is_global_perf_tuner_available() -> bool:
    """Return True if TensorRT exposes Global Performance Tuner build-route APIs."""
    if not hasattr(trt.IBuilderConfig, "build_route") or not hasattr(
        trt.IBuilderConfig, "all_build_routes"
    ):
        return False
    try:
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        config = builder.create_builder_config()
        routes = getattr(config, "all_build_routes", "") or ""
        return bool(routes.strip())
    except Exception as exc:  # pragma: no cover - depends on local TRT/CUDA
        _LOGGER.debug(f"Global Performance Tuner probe failed: {exc}")
        return False


def require_global_perf_tuner(reason: str) -> None:
    """Raise if GPT is unavailable when the user requested a GPT feature."""
    if not is_global_perf_tuner_available():
        raise RuntimeError(
            f"{reason} requires TensorRT Global Performance Tuner "
            "(IBuilderConfig.build_route / all_build_routes). "
            "This feature is available since TensorRT 11.1 and is currently not available in TensorRT-RTX or Windows."
        )


def get_all_build_routes_raw() -> str:
    """Return the raw JSON string from ``IBuilderConfig.all_build_routes``."""
    require_global_perf_tuner("Querying build routes")
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    config = builder.create_builder_config()
    return config.all_build_routes or ""


def get_all_build_routes(knob: Optional[str] = None) -> Dict[str, Any]:
    """Parse ``all_build_routes`` JSON, optionally filtering to one knob.

    Args:
        knob: Optional knob name (with or without leading "-"), matching "trtexec --helpBuildRoute[=knob]".

    Returns:
        Parsed knob database dict with "tuner_version" and "tuner_options".
    """
    raw = get_all_build_routes_raw()
    try:
        root: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse all_build_routes JSON from TensorRT: {exc}"
        ) from exc

    if knob is None or knob == "":
        return root

    wanted = knob[1:] if knob.startswith("-") else knob
    options = root.get("tuner_options", [])
    filtered = []
    for opt in options:
        name = opt.get("option", "")
        bare = name[1:] if isinstance(name, str) and name.startswith("-") else name
        if bare == wanted:
            filtered.append(opt)

    if not filtered:
        raise ValueError(
            f"No such knob in the Global Performance Tuner database: {knob}. "
            "Call get_all_build_routes() without a filter to list knobs."
        )

    out: Dict[str, Any] = {}
    if "tuner_version" in root:
        out["tuner_version"] = root["tuner_version"]
    out["tuner_options"] = filtered
    return out


def gpt_settings_requested(settings: Any) -> bool:
    """True if CompilationSettings requests any GPT feature."""
    if getattr(settings, "build_route", ""):
        return True
    if getattr(settings, "tune_build_routes", ""):
        return True
    if getattr(settings, "tune_build_route_file", None):
        return True
    if getattr(settings, "tuning_continue", False):
        return True
    if getattr(settings, "tuning_dry_run", False):
        return True
    return False
