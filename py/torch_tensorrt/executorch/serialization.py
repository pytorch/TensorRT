# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
from typing import Dict, List, Tuple, Union

# The magic is the only field an older parser validates, so it marks what an older
# parser would misread:
# - TR01: the engine follows the metadata inline.
# - TR02: inline engine, and the metadata carries aliased_io. A pre-aliasing parser
#   would bind each aliased output to its own allocation and return wrong results.
# - TR03: the engine is stored as named data (engine_key) and the blob carries none.
#   A pre-TR03 parser would read an engine of size zero and fail only inside
#   TensorRT, with an error that doesn't say the runtime is too old. A parser that
#   knows TR03 also knows aliased_io, so TR03 covers named engines with or without it.
# TR01 blobs stay readable by every runtime; TR02 and TR03 need one that knows them.
TENSORRT_MAGIC = b"TR01"
TENSORRT_MAGIC_ALIASED_IO = b"TR02"
TENSORRT_MAGIC_NAMED_ENGINE = b"TR03"
SUPPORTED_MAGICS = (
    TENSORRT_MAGIC,
    TENSORRT_MAGIC_ALIASED_IO,
    TENSORRT_MAGIC_NAMED_ENGINE,
)
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
    # Aliased output->input bindings: out_name -> (in_name, kind). "kind" is an
    # AliasKind value ("kv_cache_update" or "user"); the C++ backend binds each
    # aliased engine output to its aliased input's tensor (in-place) so the
    # update lands in the caller-owned buffer.
    aliased_io: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    hardware_compatible: bool = False
    device_id: int = 0
    serialized_metadata: str = ""
    target_platform: str = ""
    # Named-data key of the engine when it is stored outside the blob, which then
    # carries an empty engine. Empty means the engine follows the metadata inline.
    engine_key: str = ""

    def to_json(self) -> bytes:
        # Keep field order stable because the C++ parser is intentionally small.
        # It walks io_bindings, then aliased_io, then searches forward from the
        # end of whichever of those two it last walked for the scalar fields --
        # so a scalar written before either array is not found and keeps its
        # C++-side default while the parse still succeeds. Any field added here
        # that the C++ side reads by key must go after both arrays.
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
            # List form (not a dict) so the small C++ parser can walk it like
            # io_bindings. Emitted right after io_bindings, before the scalars.
            "aliased_io": [
                {"output": out, "input": inp, "kind": kind}
                for out, (inp, kind) in self.aliased_io.items()
            ],
            "hardware_compatible": self.hardware_compatible,
            "device_id": self.device_id,
            "serialized_metadata": self.serialized_metadata,
            "target_platform": self.target_platform,
        }
        if self.engine_key:
            data["engine_key"] = self.engine_key
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
        aliased_io = {
            b["output"]: (b["input"], b.get("kind", "kv_cache_update"))
            for b in parsed.get("aliased_io", [])
        }
        return cls(
            io_bindings=io_bindings,
            aliased_io=aliased_io,
            hardware_compatible=parsed.get("hardware_compatible", False),
            device_id=parsed.get("device_id", 0),
            serialized_metadata=parsed.get("serialized_metadata", ""),
            target_platform=parsed.get("target_platform", ""),
            engine_key=parsed.get("engine_key", ""),
        )


def serialize_engine(
    engine_bytes: Union[bytes, bytearray, memoryview],
    metadata: TensorRTBlobMetadata,
) -> bytes:
    if bool(metadata.engine_key) == bool(memoryview(engine_bytes).nbytes):
        raise ValueError(
            "A TensorRT blob carries its engine either inline or as named data "
            "(engine_key), never both or neither"
        )
    if metadata.engine_key:
        magic = TENSORRT_MAGIC_NAMED_ENGINE
    elif metadata.aliased_io:
        magic = TENSORRT_MAGIC_ALIASED_IO
    else:
        magic = TENSORRT_MAGIC
    metadata_json = metadata.to_json()
    metadata_offset = HEADER_SIZE
    engine_offset = _align_to_16(metadata_offset + len(metadata_json))
    reserved = b"\x01" + b"\x00" * 7
    header = struct.pack(
        HEADER_FORMAT,
        magic,
        metadata_offset,
        len(metadata_json),
        engine_offset,
        memoryview(engine_bytes).nbytes,
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
    if magic not in SUPPORTED_MAGICS:
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
    if (magic == TENSORRT_MAGIC_NAMED_ENGINE) != bool(metadata.engine_key):
        raise ValueError(
            f"Magic {magic!r} does not match the engine_key in the metadata"
        )
    if bool(metadata.engine_key) == bool(engine_size):
        raise ValueError(
            "Blob carries its engine both inline and as named data, or neither"
        )
    engine = blob[engine_offset : engine_offset + engine_size]
    return engine, metadata
