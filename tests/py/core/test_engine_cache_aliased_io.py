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
import tempfile
import unittest

from torch_tensorrt.dynamo._engine_cache import BaseEngineCache, DiskEngineCache
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

    def test_check_rejects_a_blob_written_before_aliased_io_was_cached(self):
        """A pre-fix entry must be a miss, not a silently empty map.

        Such a blob is indistinguishable from one for a model with no aliased IO, and
        guessing "none" returns the engine's internal plumbing tensor to the caller,
        changing output arity on a cache hit. A miss rebuilds the engine once instead.
        """
        cache_dir = tempfile.mkdtemp()
        cache = DiskEngineCache(cache_dir, 1 << 30)

        fresh = BaseEngineCache.pack(
            serialized_engine=b"engine",
            input_names=["input_0"],
            output_names=["output_0", "output_1"],
            input_specs=(),
            compilation_settings=CompilationSettings(),
            requires_output_allocator=False,
            requires_native_multidevice=False,
            aliased_io=ALIASED_IO,
        )
        cache.save("fresh", fresh)
        hit = cache.check("fresh")
        self.assertIsNotNone(hit, "a blob carrying aliased_io must still be reusable")
        self.assertEqual(hit[7], ALIASED_IO)

        cache.save("legacy", self._legacy_blob())
        self.assertIsNone(cache.check("legacy"))

    @staticmethod
    def _legacy_blob():
        return pickle.dumps(
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

    def test_unpack_tolerates_a_blob_without_aliased_io(self):
        """unpack stays lenient: it is a deserializer, not the reuse decision.

        The reuse decision lives in check(), which rejects such a blob.
        """
        self.assertEqual(BaseEngineCache.unpack(self._legacy_blob())[7], {})


if __name__ == "__main__":
    unittest.main()
