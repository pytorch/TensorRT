import struct

import pytest
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import SERIALIZATION_LEN
from torch_tensorrt.executorch.serialization import serialize_engine_info


@pytest.mark.unit
def test_serialize_engine_info_pads_and_encodes_entries():
    assert SERIALIZATION_LEN > 0

    blob = serialize_engine_info(["alpha", b"\x00\x01"])

    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == SERIALIZATION_LEN

    first_len = struct.unpack_from("<I", blob, 4)[0]
    assert blob[8 : 8 + first_len] == b"alpha"


@pytest.mark.unit
def test_serialize_engine_info_handles_none_entries():
    blob = serialize_engine_info(["alpha", None])
    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == SERIALIZATION_LEN
