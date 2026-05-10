# Serialization for ExecuTorch TensorRT blob: same format as TRT runtime (vector of strings).
# Uses the same list format as TorchTensorRTModule._pack_engine_info, then encodes to bytes.
# Only valid when ENABLED_FEATURES.torch_tensorrt_runtime is True.

import struct
from typing import List, Union

from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import SERIALIZATION_LEN


def serialize_engine_info(engine_info: List[Union[str, bytes]]) -> bytes:
    """Encode engine info list (same format as TorchTensorRTModule._pack_engine_info) to bytes.

    Takes the list produced by _pack_engine_info (or equivalent) and writes it in the
    TRT runtime vector<string> format: 4-byte count (SERIALIZATION_LEN), then for each
    entry 4-byte length (LE) + raw bytes. C++ can deserialize to std::vector<std::string>
    and pass to TRTEngine(std::vector<std::string> serialized_info).
    """
    if len(engine_info) < SERIALIZATION_LEN:
        engine_info = list(engine_info) + [""] * (SERIALIZATION_LEN - len(engine_info))
    parts: List[bytes] = []
    for i in range(SERIALIZATION_LEN):
        raw = engine_info[i]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        elif raw is None:
            raw = b""
        else:
            raw = bytes(raw)
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    return struct.pack("<I", SERIALIZATION_LEN) + b"".join(parts)
