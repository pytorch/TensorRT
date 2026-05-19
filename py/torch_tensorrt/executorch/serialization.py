# Serialization for ExecuTorch TensorRT blob.
#
# Wire format ("TR01"):
#
#   Offset  Size  Field
#   ------  ----  -----
#   0       4     Magic bytes b"TR01"
#   4       4     Metadata offset (LE uint32, bytes from blob start)
#   8       4     Metadata size (LE uint32, bytes)
#   12      4     Engine offset (LE uint32, bytes from blob start, 16-byte aligned)
#   16      8     Engine size (LE uint64, bytes; uint64 supports >4 GB engines)
#   24      8     Reserved (1-byte schema version + 7-byte padding)
#
# Metadata region: the legacy vector<string> format produced by
# `_pack_engine_info()` with ENGINE_IDX blanked. Engine region: raw engine
# bytes at a 16-byte aligned offset so the runtime can mmap or DMA the
# payload without an extra copy.
#
# The C++ deserializer in core/runtime/executorch/TensorRTBackend.cpp
# recognises blobs by magic bytes; legacy blobs without "TR01" continue to
# parse via the existing length-prefixed vector<string> path.

import struct
from typing import List, Union

from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ENGINE_IDX,
    SERIALIZATION_LEN,
)

TENSORRT_MAGIC = b"TR01"
HEADER_FORMAT = "<4sIIIQ8s"  # magic, meta_off, meta_size, eng_off, eng_size, reserved
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
SCHEMA_VERSION = 1
ENGINE_ALIGNMENT = 16


def _pack_vector_of_strings(parts: List[Union[str, bytes, None]]) -> bytes:
    """Inner format: 4-byte count + (4-byte len + raw bytes) per entry."""
    if len(parts) < SERIALIZATION_LEN:
        parts = list(parts) + [""] * (SERIALIZATION_LEN - len(parts))
    out: List[bytes] = []
    for i in range(SERIALIZATION_LEN):
        raw = parts[i]
        if raw is None:
            raw = b""
        elif isinstance(raw, str):
            raw = raw.encode("utf-8")
        elif not isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw)
        out.append(struct.pack("<I", len(raw)))
        out.append(bytes(raw))
    return struct.pack("<I", SERIALIZATION_LEN) + b"".join(out)


def serialize_engine_info(engine_info: List[Union[str, bytes]]) -> bytes:
    """Encode engine info list to the TR01 wire format.

    Splits the engine bytes out of the input list and writes them at a
    16-byte aligned offset, leaving the rest of the metadata in an
    embedded length-prefixed vector<string> block. The corresponding
    C++ deserializer rebuilds the original list by re-injecting the
    engine bytes at ENGINE_IDX before handing it to
    `core::runtime::TRTEngine`.
    """
    info = list(engine_info)
    if len(info) < SERIALIZATION_LEN:
        info = info + [""] * (SERIALIZATION_LEN - len(info))

    engine = info[ENGINE_IDX]
    if engine is None:
        engine_bytes = b""
    elif isinstance(engine, (bytes, bytearray)):
        engine_bytes = bytes(engine)
    elif isinstance(engine, str):
        # Legacy producers may hand us a (base64-encoded) string. Preserve
        # bytes by encoding as latin-1 so we never silently corrupt the
        # blob; consumers that expect the legacy text form will still see
        # the same characters back at the same slot.
        engine_bytes = engine.encode("latin-1")
    else:
        engine_bytes = bytes(engine)

    info[ENGINE_IDX] = b""  # blank in the metadata block; carried separately
    metadata = _pack_vector_of_strings(info)

    metadata_offset = HEADER_SIZE
    engine_offset = (metadata_offset + len(metadata) + (ENGINE_ALIGNMENT - 1)) & ~(
        ENGINE_ALIGNMENT - 1
    )
    padding = b"\x00" * (engine_offset - metadata_offset - len(metadata))
    reserved = bytes([SCHEMA_VERSION]) + b"\x00" * 7

    header = struct.pack(
        HEADER_FORMAT,
        TENSORRT_MAGIC,
        metadata_offset,
        len(metadata),
        engine_offset,
        len(engine_bytes),
        reserved,
    )
    return header + metadata + padding + engine_bytes
