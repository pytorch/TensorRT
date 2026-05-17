"""Self-contained TensorRT engine blob format for ExecuTorch.

The ExecuTorch runtime path must not depend on libtorch, TorchScript custom
classes, or the Torch-TensorRT C++ runtime ABI.  This module therefore emits a
small standalone blob that contains JSON metadata plus raw TensorRT engine
bytes.  The C++ ExecuTorch backend parses the same layout directly.
"""

import dataclasses
import json
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

TENSORRT_MAGIC = b"TR01"
HEADER_FORMAT = "<4sIIIQ8s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _align_to_16(offset: int) -> int:
    return (offset + 15) & ~15


@dataclass
class TensorRTIOBinding:
    name: str
    dtype: str = ""
    shape: List[int] = field(default_factory=list)
    is_input: bool = True


@dataclass
class TensorRTBlobMetadata:
    io_bindings: List[TensorRTIOBinding] = field(default_factory=list)
    hardware_compatible: bool = False
    device_id: int = 0
    serialized_metadata: str = ""
    target_platform: str = ""

    def to_json(self) -> bytes:
        # Keep field order stable because the C++ parser is intentionally small
        # and searches forward after io_bindings for the scalar fields.
        data = {
            "io_bindings": [
                {
                    "name": binding.name,
                    "dtype": binding.dtype,
                    "shape": binding.shape,
                    "is_input": binding.is_input,
                }
                for binding in self.io_bindings
            ],
            "hardware_compatible": self.hardware_compatible,
            "device_id": self.device_id,
            "serialized_metadata": self.serialized_metadata,
            "target_platform": self.target_platform,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "TensorRTBlobMetadata":
        parsed = json.loads(data.decode("utf-8"))
        binding_fields = {f.name for f in dataclasses.fields(TensorRTIOBinding)}
        io_bindings = [
            TensorRTIOBinding(
                **{k: v for k, v in binding.items() if k in binding_fields}
            )
            for binding in parsed.get("io_bindings", [])
        ]
        return cls(
            io_bindings=io_bindings,
            hardware_compatible=parsed.get("hardware_compatible", False),
            device_id=parsed.get("device_id", 0),
            serialized_metadata=parsed.get("serialized_metadata", ""),
            target_platform=parsed.get("target_platform", ""),
        )


def serialize_engine(engine_bytes: bytes, metadata: TensorRTBlobMetadata) -> bytes:
    metadata_json = metadata.to_json()
    metadata_offset = HEADER_SIZE
    engine_offset = _align_to_16(metadata_offset + len(metadata_json))
    reserved = b"\x01" + b"\x00" * 7
    header = struct.pack(
        HEADER_FORMAT,
        TENSORRT_MAGIC,
        metadata_offset,
        len(metadata_json),
        engine_offset,
        len(engine_bytes),
        reserved,
    )
    padding = b"\x00" * (engine_offset - metadata_offset - len(metadata_json))
    return header + metadata_json + padding + engine_bytes


def deserialize_engine(blob: bytes) -> Tuple[bytes, TensorRTBlobMetadata]:
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"Blob too small: {len(blob)} bytes")
    magic, metadata_offset, metadata_size, engine_offset, engine_size, _ = (
        struct.unpack(HEADER_FORMAT, blob[:HEADER_SIZE])
    )
    if magic != TENSORRT_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    if engine_offset % 16 != 0:
        raise ValueError(f"Engine offset is not 16-byte aligned: {engine_offset}")
    if metadata_offset < HEADER_SIZE:
        raise ValueError(
            f"Metadata offset {metadata_offset} is inside the header "
            f"(size={HEADER_SIZE})"
        )
    if metadata_offset + metadata_size > len(blob):
        raise ValueError("Metadata extends past blob")
    if engine_offset + engine_size > len(blob):
        raise ValueError("Engine extends past blob")
    if metadata_offset + metadata_size > engine_offset:
        raise ValueError("Metadata overlaps engine")

    metadata = TensorRTBlobMetadata.from_json(
        blob[metadata_offset : metadata_offset + metadata_size]
    )
    engine = blob[engine_offset : engine_offset + engine_size]
    return engine, metadata
