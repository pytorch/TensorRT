"""Engine-cache blob round-trip for ``aliased_io``.

When TensorRT writes a buffer in place, the converter appends that buffer as an extra
network output, because ``IKVCacheUpdateLayer`` requires its output to be a network
output. ``aliased_io`` records which outputs are that internal plumbing, and
``TorchTensorRTModule.forward`` uses it to hide them from the caller again.

If the cache drops the map, the hiding step is skipped and the plumbing tensor is
returned to the caller, so the same model yields a different number of outputs on a
cache hit than on a fresh build.

These are pure blob tests: no GPU, no TensorRT engine, so they run in the fast lane.
"""

import pickle
import unittest

from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings

ALIASED_IO = {"output_1": ("buf_cache", "kv_cache_update")}


class TestAliasedIOCacheRoundTrip(unittest.TestCase):
    def test_pack_unpack_round_trips_aliased_io(self):
        blob = BaseEngineCache.pack(
            serialized_engine=b"engine",
            input_names=["input_0"],
            output_names=["output_0", "output_1"],
            input_specs=(),
            compilation_settings=CompilationSettings(),
            requires_output_allocator=False,
            requires_native_multidevice=False,
            aliased_io=ALIASED_IO,
        )
        self.assertEqual(BaseEngineCache.unpack(blob)[7], ALIASED_IO)

    def test_unpack_tolerates_a_blob_without_aliased_io(self):
        """A blob written before aliased_io was cached must still unpack."""
        legacy = pickle.dumps(
            {
                "serialized_engine": b"engine",
                "input_names": ["input_0"],
                "output_names": ["output_0"],
                "input_specs": (),
                "compilation_settings": CompilationSettings(),
                "requires_output_allocator": False,
                "requires_native_multidevice": False,
            }
        )
        self.assertEqual(BaseEngineCache.unpack(legacy)[7], {})


if __name__ == "__main__":
    unittest.main()
