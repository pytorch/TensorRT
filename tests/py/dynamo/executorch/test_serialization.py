"""Round-trip + invariant tests for the TR01 ExecuTorch wire format."""

import os
import struct
import unittest

import pytest
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ENGINE_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch.serialization import (
    ENGINE_ALIGNMENT,
    HEADER_FORMAT,
    HEADER_SIZE,
    SCHEMA_VERSION,
    TENSORRT_MAGIC,
    serialize_engine_info,
)

assertions = unittest.TestCase()


def _make_info(engine_bytes: bytes):
    info = [""] * SERIALIZATION_LEN
    info[ENGINE_IDX] = engine_bytes
    return info


def _parse_header(blob: bytes):
    magic, meta_off, meta_size, eng_off, eng_size, reserved = struct.unpack(
        HEADER_FORMAT, blob[:HEADER_SIZE]
    )
    return {
        "magic": magic,
        "metadata_offset": meta_off,
        "metadata_size": meta_size,
        "engine_offset": eng_off,
        "engine_size": eng_size,
        "version": reserved[0],
    }


@pytest.mark.unit
def test_header_magic_and_version():
    """Every blob starts with `TR01` and reports the current schema version."""
    blob = serialize_engine_info(_make_info(b"engine"))
    header = _parse_header(blob)
    assertions.assertEqual(header["magic"], TENSORRT_MAGIC)
    assertions.assertEqual(header["version"], SCHEMA_VERSION)


@pytest.mark.unit
def test_engine_offset_is_16_byte_aligned():
    """Engine payload is placed at a 16-byte aligned offset so the runtime can
    mmap / DMA / SIMD-load it without an intermediate copy."""
    # Try a few metadata sizes to exercise different padding amounts.
    for engine_len in (0, 1, 15, 16, 17, 4096):
        blob = serialize_engine_info(_make_info(os.urandom(engine_len)))
        header = _parse_header(blob)
        assertions.assertEqual(
            header["engine_offset"] % ENGINE_ALIGNMENT,
            0,
            msg=f"engine_offset {header['engine_offset']} not aligned (engine_len={engine_len})",
        )


@pytest.mark.unit
def test_engine_bytes_round_trip_small():
    """Engine payload survives the serialized layout byte-for-byte."""
    engine = os.urandom(100)
    blob = serialize_engine_info(_make_info(engine))
    header = _parse_header(blob)
    assertions.assertEqual(header["engine_size"], len(engine))
    extracted = blob[header["engine_offset"] : header["engine_offset"] + len(engine)]
    assertions.assertEqual(extracted, engine)


@pytest.mark.unit
def test_engine_bytes_round_trip_large():
    """10 MB engine round-trips. (Smoke test for >metadata-only payloads.)"""
    engine = os.urandom(10 * 1024 * 1024)
    blob = serialize_engine_info(_make_info(engine))
    header = _parse_header(blob)
    extracted = blob[header["engine_offset"] : header["engine_offset"] + len(engine)]
    assertions.assertEqual(extracted, engine)


@pytest.mark.unit
def test_metadata_blanks_engine_slot():
    """The embedded metadata region must not duplicate the engine payload —
    the engine lives once, in the aligned engine section."""
    engine = os.urandom(1024)
    blob = serialize_engine_info(_make_info(engine))
    header = _parse_header(blob)
    metadata = blob[
        header["metadata_offset"] : header["metadata_offset"] + header["metadata_size"]
    ]
    assertions.assertNotIn(
        engine,
        metadata,
        msg="engine bytes appeared in the metadata region (double-serialized)",
    )


@pytest.mark.unit
def test_handles_oversized_info_list():
    """An over-long info list is truncated to SERIALIZATION_LEN; an under-long
    one is padded — never raises."""
    short = [""] * (SERIALIZATION_LEN - 2)
    short[ENGINE_IDX] = b"short"
    blob_short = serialize_engine_info(short)
    assertions.assertEqual(blob_short[:4], TENSORRT_MAGIC)

    long = [""] * (SERIALIZATION_LEN + 5)
    long[ENGINE_IDX] = b"long"
    blob_long = serialize_engine_info(long)
    assertions.assertEqual(blob_long[:4], TENSORRT_MAGIC)


@pytest.mark.unit
def test_empty_engine():
    """An empty engine slot must still produce a valid TR01 blob (e.g. for
    minimal sanity tests on the consumer side)."""
    blob = serialize_engine_info(_make_info(b""))
    header = _parse_header(blob)
    assertions.assertEqual(header["engine_size"], 0)
    assertions.assertEqual(header["engine_offset"] % ENGINE_ALIGNMENT, 0)
