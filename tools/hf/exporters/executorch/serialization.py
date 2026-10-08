"""Versioned Edge-LLM component payloads for ExecuTorch delegates."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

EDGE_LLM_MAGIC = b"EL01"
EDGE_LLM_ABI_VERSION = 1
HEADER_FORMAT = "<4sIIIQ8s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _align_to_16(offset: int) -> int:
    return (offset + 15) & ~15


@dataclass(frozen=True)
class EdgeOutputSpec:
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EdgeOutputSpec":
        try:
            shape = tuple(int(dim) for dim in value["shape"])
            dtype = str(value["dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Edge output specification: {value!r}") from exc
        if any(dim < 0 for dim in shape):
            raise ValueError(
                f"Edge output shape must be static and non-negative: {shape}"
            )
        if not dtype:
            raise ValueError("Edge output dtype must not be empty")
        return cls(shape=shape, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class EdgeComponentMetadata:
    component: str
    runner: str
    outputs: tuple[EdgeOutputSpec, ...]
    runner_config: dict[str, Any] = field(default_factory=dict)
    abi_version: int = EDGE_LLM_ABI_VERSION

    def validate(self) -> None:
        if self.abi_version != EDGE_LLM_ABI_VERSION:
            raise ValueError(
                f"Unsupported Edge-LLM ABI version {self.abi_version}; "
                f"expected {EDGE_LLM_ABI_VERSION}"
            )
        if self.component not in {"vision", "language", "action"}:
            raise ValueError(f"Unsupported Edge-LLM component {self.component!r}")
        if not self.runner:
            raise ValueError("Edge-LLM runner must not be empty")
        if not self.outputs:
            raise ValueError("Edge-LLM component must declare at least one output")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(
            {
                "abi_version": self.abi_version,
                "component": self.component,
                "runner": self.runner,
                "outputs": [output.to_dict() for output in self.outputs],
                "runner_config": self.runner_config,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "EdgeComponentMetadata":
        try:
            parsed = json.loads(value)
            metadata = cls(
                abi_version=int(parsed["abi_version"]),
                component=str(parsed["component"]),
                runner=str(parsed["runner"]),
                outputs=tuple(
                    EdgeOutputSpec.from_dict(output) for output in parsed["outputs"]
                ),
                runner_config=dict(parsed.get("runner_config", {})),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid Edge-LLM component metadata") from exc
        metadata.validate()
        return metadata


def serialize_edge_component(trt_blob: bytes, metadata: EdgeComponentMetadata) -> bytes:
    """Wrap a TR01/TR02 TensorRT blob in an EL01 component envelope."""
    metadata_json = metadata.to_json().encode("utf-8")
    metadata_offset = HEADER_SIZE
    payload_offset = _align_to_16(metadata_offset + len(metadata_json))
    header = struct.pack(
        HEADER_FORMAT,
        EDGE_LLM_MAGIC,
        metadata_offset,
        len(metadata_json),
        payload_offset,
        len(trt_blob),
        b"\x00" * 8,
    )
    padding = b"\x00" * (payload_offset - metadata_offset - len(metadata_json))
    return header + metadata_json + padding + trt_blob


def deserialize_edge_component(
    payload: bytes,
) -> tuple[bytes, EdgeComponentMetadata]:
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"Edge-LLM payload is too small: {len(payload)} bytes")
    magic, metadata_offset, metadata_size, blob_offset, blob_size, _ = struct.unpack(
        HEADER_FORMAT, payload[:HEADER_SIZE]
    )
    if magic != EDGE_LLM_MAGIC:
        raise ValueError(f"Invalid Edge-LLM payload magic: {magic!r}")
    if metadata_offset < HEADER_SIZE:
        raise ValueError("Edge-LLM metadata starts inside the payload header")
    if blob_offset % 16 != 0:
        raise ValueError("Nested TensorRT blob is not 16-byte aligned")
    if metadata_offset + metadata_size > blob_offset:
        raise ValueError("Edge-LLM metadata overlaps the nested TensorRT blob")
    if blob_offset + blob_size > len(payload):
        raise ValueError("Nested TensorRT blob extends past the Edge-LLM payload")

    metadata = EdgeComponentMetadata.from_json(
        payload[metadata_offset : metadata_offset + metadata_size]
    )
    return payload[blob_offset : blob_offset + blob_size], metadata
