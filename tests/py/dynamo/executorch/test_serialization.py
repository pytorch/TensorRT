import json

import pytest
from torch_tensorrt.executorch.serialization import (
    HEADER_SIZE,
    TENSORRT_MAGIC,
    TENSORRT_MAGIC_ALIASED_IO,
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    deserialize_engine,
    serialize_engine,
)


@pytest.mark.unit
def test_serialize_engine_writes_tr01_blob():
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name="x", is_input=True),
            TensorRTIOBinding(name="y", is_input=False),
        ],
        device_id=1,
        hardware_compatible=True,
    )

    blob = serialize_engine(b"engine-bytes", metadata)

    assert blob[:4] == TENSORRT_MAGIC
    assert len(blob) > HEADER_SIZE

    engine, parsed = deserialize_engine(blob)
    assert engine == b"engine-bytes"
    assert parsed.device_id == 1
    assert parsed.hardware_compatible is True
    assert [b.name for b in parsed.io_bindings] == ["x", "y"]
    assert [b.is_input for b in parsed.io_bindings] == [True, False]


@pytest.mark.unit
def test_deserialize_engine_rejects_bad_magic():
    with pytest.raises(ValueError, match="Invalid magic"):
        deserialize_engine(b"NOPE" + b"\x00" * (HEADER_SIZE - 4))


@pytest.mark.unit
def test_serialize_engine_round_trips_aliased_io():
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name="in_k", is_input=True),
            TensorRTIOBinding(name="out_k", is_input=False),
            TensorRTIOBinding(name="in_u", is_input=True),
            TensorRTIOBinding(name="out_u", is_input=False),
        ],
        aliased_io={
            "out_k": ("in_k", "kv_cache_update"),
            "out_u": ("in_u", "user"),
        },
    )

    engine, parsed = deserialize_engine(serialize_engine(b"eng", metadata))
    assert engine == b"eng"
    assert parsed.aliased_io == {
        "out_k": ("in_k", "kv_cache_update"),
        "out_u": ("in_u", "user"),
    }


@pytest.mark.unit
def test_metadata_from_json_without_aliased_io_defaults_empty():
    # Blobs written before aliased_io existed omit the key entirely; parsing must
    # default to an empty mapping rather than raising (backward compatibility).
    metadata = TensorRTBlobMetadata(
        io_bindings=[TensorRTIOBinding(name="x", is_input=True)]
    )
    data = json.loads(metadata.to_json().decode("utf-8"))
    del data["aliased_io"]

    restored = TensorRTBlobMetadata.from_json(json.dumps(data).encode("utf-8"))
    assert restored.aliased_io == {}


@pytest.mark.unit
def test_aliased_io_blob_uses_the_bumped_magic():
    """aliased_io changes what a blob means: a parser that ignores it binds each
    aliased output to its own allocation instead of the input it aliases, and
    returns wrong results. The magic is the only field such a parser validates, so
    aliased blobs must not present as TR01."""
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name="x", is_input=True),
            TensorRTIOBinding(name="out_k", is_input=False),
        ],
        aliased_io={"out_k": ("x", "kv_cache_update")},
    )
    blob = serialize_engine(b"engine-bytes", metadata)
    assert blob[:4] == TENSORRT_MAGIC_ALIASED_IO
    assert blob[:4] != TENSORRT_MAGIC


@pytest.mark.unit
def test_blob_without_aliased_io_keeps_the_original_magic():
    """Nothing about a non-aliased blob is new, so it stays loadable by a parser
    that predates aliased_io."""
    metadata = TensorRTBlobMetadata(
        io_bindings=[TensorRTIOBinding(name="x", is_input=True)]
    )
    assert serialize_engine(b"engine-bytes", metadata)[:4] == TENSORRT_MAGIC


@pytest.mark.unit
@pytest.mark.parametrize("aliased_io", [{}, {"out_k": ("x", "kv_cache_update")}])
def test_deserialize_accepts_both_magics(aliased_io):
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name="x", is_input=True),
            TensorRTIOBinding(name="out_k", is_input=False),
        ],
        aliased_io=aliased_io,
    )
    engine, parsed = deserialize_engine(serialize_engine(b"engine-bytes", metadata))
    assert engine == b"engine-bytes"
    assert parsed.aliased_io == aliased_io


@pytest.mark.unit
def test_to_json_writes_every_scalar_after_both_arrays():
    """The ordering rule the C++ parser's scalar scans depend on.

    ``TensorRTBlobHeader.cpp`` walks ``io_bindings``, then ``aliased_io``, then
    searches forward from the end of whichever of those it last walked for the
    scalar fields, so a scalar written ahead of either array is not found and
    keeps its C++-side default while the parse still succeeds -- no error, and a
    ``device_id`` of 0 means the engine deserializes on a GPU nobody named. The
    C++ half of this rule is
    ``ParsesEveryScalarFromTheWriterKeyOrder`` in
    ``tests/cpp/executorch/test_executorch_blob_header.cpp``; this is the half
    that fails when a field is added to ``to_json`` in the wrong place.

    Every key that side reads by key is listed, not only the two scalars: a new
    scalar it learns to read is only safe in the same position.
    """
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name="in_k", dtype="float32", shape=[1, 2]),
            TensorRTIOBinding(name="out_k", dtype="float32", is_input=False),
        ],
        aliased_io={"out_k": ("in_k", "kv_cache_update")},
        hardware_compatible=True,
        device_id=6,
        target_platform="linux_x86_64",
    )

    text = metadata.to_json().decode("utf-8")
    arrays_end = max(
        text.index("]", text.index(key)) for key in ('"io_bindings"', '"aliased_io"')
    )
    for key in ('"hardware_compatible"', '"device_id"'):
        assert text.index(key) > arrays_end, f"{key} is written before an array"

    # Read back through the writer's own reader as well, so the ordering
    # assertion is made about a payload that is otherwise correct.
    restored = TensorRTBlobMetadata.from_json(metadata.to_json())
    assert restored.hardware_compatible is True
    assert restored.device_id == 6
    assert restored.aliased_io == {"out_k": ("in_k", "kv_cache_update")}
    assert [b.name for b in restored.io_bindings] == ["in_k", "out_k"]
