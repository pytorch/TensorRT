import pytest
from torch_tensorrt.executorch.serialization import (
    HEADER_SIZE,
    TENSORRT_MAGIC,
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
