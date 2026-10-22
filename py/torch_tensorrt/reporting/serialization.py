# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Strict JSON roundtrips and schemas, implemented with the Python standard library."""

from __future__ import annotations

import dataclasses
import json
import math
import types
from enum import Enum
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from .models import REPORT_TYPES, SCHEMA_VERSION, Model


class ContractError(ValueError):
    """A document does not conform to its declared reporting contract."""


def _encode(value: Any) -> Any:
    """Recursively convert supported contract values into strict JSON values."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and isinstance(value, Model):
        result = {
            field.name: _encode(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        if hasattr(type(value), "report_type"):
            result["report_type"] = type(value).report_type
        return result
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("JSON object keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    raise ContractError(f"Unsupported JSON value type: {type(value).__name__}")


def to_dict(value: Any) -> dict[str, Any]:
    """Encode and validate a top-level contract object."""
    result = _encode(value)
    if not isinstance(result, dict):
        raise ContractError("A contract document must be an object")
    if isinstance(value, Model):
        # Validate even documents constructed without runtime type checking.
        result = _encode(_decode(type(value), result, "$"))
    return cast(dict[str, Any], result)


def to_json(value: Any) -> str:
    """Serialize a supported value as deterministic strict JSON."""
    result = to_dict(value) if isinstance(value, Model) else _encode(value)
    return json.dumps(
        result,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    """Reject the nonstandard NaN and infinity constants accepted by json."""
    raise ContractError(f"JSON numbers must be finite: {value}")


def from_json(text: str, expected_type: type[Model] | None = None) -> Model:
    """Parse strict JSON and decode it as a reporting contract."""
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("Invalid JSON document") from exc
    return from_dict(value, expected_type)


def from_dict(value: dict[str, Any], expected_type: type[Model] | None = None) -> Model:
    """Decode a JSON-compatible object as an explicit or declared model type."""
    if not isinstance(value, dict):
        raise ContractError("A contract document must be an object")
    if expected_type is None:
        report_type = value.get("report_type")
        if not isinstance(report_type, str):
            raise ContractError("Unknown or missing report_type")
        expected_type = REPORT_TYPES.get(report_type)
        if expected_type is None:
            raise ContractError("Unknown or missing report_type")
    return cast(Model, _decode(expected_type, value, "$"))


def _decode(annotation: Any, value: Any, path: str) -> Any:
    """Validate and reconstruct a value according to a runtime type annotation."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return _encode(value)
    if origin in (Union, types.UnionType):
        for candidate in args:
            try:
                return _decode(candidate, value, path)
            except (ContractError, ValueError, TypeError):
                pass
        raise ContractError(f"{path}: value does not match allowed types")
    if annotation is type(None):
        if value is not None:
            raise ContractError(f"{path}: expected null")
        return None
    if origin is list:
        if not isinstance(value, list):
            raise ContractError(f"{path}: expected array")
        return [
            _decode(args[0], item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise ContractError(f"{path}: expected object with string keys")
        return {
            key: _decode(args[1], item, f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not isinstance(value, str):
            raise ContractError(f"{path}: expected enum string")
        try:
            return annotation(value)
        except ValueError as exc:
            raise ContractError(f"{path}: unknown enum value") from exc
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ContractError(f"{path}: expected object")
        fields = {field.name: field for field in dataclasses.fields(annotation)}
        allowed = set(fields)
        if hasattr(annotation, "report_type"):
            allowed.add("report_type")
            if value.get("report_type") != annotation.report_type:
                raise ContractError(f"{path}: unexpected or missing report_type")
            if value.get("schema_version") != SCHEMA_VERSION:
                raise ContractError(f"{path}: unknown or missing schema_version")
        unknown = set(value) - allowed
        if unknown:
            raise ContractError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
        hints = get_type_hints(annotation)
        kwargs = {}
        for name, field in fields.items():
            if name in value:
                kwargs[name] = _decode(hints[name], value[name], f"{path}.{name}")
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                raise ContractError(f"{path}.{name}: missing required field")
        try:
            return annotation(**kwargs)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"{path}: {exc}") from exc
    if annotation is bool:
        if type(value) is not bool:
            raise ContractError(f"{path}: expected boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ContractError(f"{path}: expected integer")
        return value
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ContractError(f"{path}: expected finite number")
        return float(value)
    if annotation is str:
        if type(value) is not str:
            raise ContractError(f"{path}: expected string")
        return value
    raise ContractError(f"{path}: unsupported annotation {annotation}")


def json_schema(model_type: type[Model]) -> dict[str, Any]:
    """Return a Draft 2020-12 schema for one contract without a runtime dependency."""
    definitions: dict[str, Any] = {}

    def schema(annotation: Any) -> dict[str, Any]:
        """Translate one supported annotation into a schema fragment."""
        origin, args = get_origin(annotation), get_args(annotation)
        if annotation is Any:
            return {}
        if origin in (Union, types.UnionType):
            return {"anyOf": [schema(arg) for arg in args]}
        if annotation is type(None):
            return {"type": "null"}
        if origin is list:
            return {"type": "array", "items": schema(args[0])}
        if origin is dict:
            return {"type": "object", "additionalProperties": schema(args[1])}
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return {"type": "string", "enum": [item.value for item in annotation]}
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            name = annotation.__name__
            if name not in definitions:
                definitions[name] = {}
                hints = get_type_hints(annotation)
                properties = {
                    f.name: schema(hints[f.name])
                    for f in dataclasses.fields(annotation)
                }
                required = [
                    f.name
                    for f in dataclasses.fields(annotation)
                    if f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING
                ]
                if hasattr(annotation, "report_type"):
                    properties["report_type"] = {"const": annotation.report_type}
                    properties["schema_version"] = {"const": SCHEMA_VERSION}
                    required.extend(["report_type", "schema_version"])
                for field_name in ("created_at", "timestamp"):
                    if field_name in properties:
                        properties[field_name] = {
                            "type": "string",
                            "format": "date-time",
                            "pattern": "Z$",
                        }
                if name == "ArtifactEntry":
                    properties["sha256"] = {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    }
                    properties["size_bytes"] = {"type": "integer", "minimum": 0}
                    properties["path"] = {
                        "type": "string",
                        "pattern": r"^(?!/)(?!.*\\)(?!.*(?:^|/)\.{1,2}(?:/|$))[^/]+(?:/[^/]+)*$",
                    }
                if name == "Failure":
                    properties["message"] = {"type": "string", "maxLength": 2048}
                if name == "Event":
                    properties["sequence"] = {"type": "integer", "minimum": 0}
                if name == "Coverage":
                    properties = {
                        key: {"type": "integer", "minimum": 0} for key in properties
                    }
                definitions[name] = {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                    "required": sorted(set(required)),
                }
            return {"$ref": f"#/$defs/{name}"}
        return {
            "type": {str: "string", int: "integer", float: "number", bool: "boolean"}[
                annotation
            ]
        }

    root = schema(model_type)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **root,
        "$defs": definitions,
    }
