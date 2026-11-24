import copy
import io
import pickle
import unittest
from unittest import mock

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._engine_handle import EngineHandle
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


class _FakeEngine:
    def __init__(self, payload: bytes = b"plan") -> None:
        self.payload = payload
        self.serialize_calls = 0

    def serialize(self) -> bytes:
        self.serialize_calls += 1
        return self.payload


class TestEngineHandle(TestCase):
    def test_requires_a_form(self):
        with self.assertRaises(ValueError):
            EngineHandle()

    def test_serializes_live_engine_once(self):
        engine = _FakeEngine()
        handle = EngineHandle(cuda_engine=engine)
        self.assertFalse(handle.has_serialized)
        self.assertEqual(handle.serialized(), b"plan")
        self.assertEqual(handle.serialized(), b"plan")
        self.assertEqual(engine.serialize_calls, 1)

    def test_deserializes_bytes_once(self):
        runtime = mock.Mock()
        runtime.deserialize_cuda_engine.return_value = sentinel = object()
        handle = EngineHandle(serialized=b"plan")
        self.assertIs(handle.cuda_engine(runtime), sentinel)
        self.assertIs(handle.cuda_engine(runtime), sentinel)
        runtime.deserialize_cuda_engine.assert_called_once_with(b"plan")

    def test_release_keeps_bytes(self):
        handle = EngineHandle(cuda_engine=_FakeEngine())
        handle.release_cuda_engine()
        self.assertFalse(handle.has_cuda_engine)
        self.assertEqual(handle.serialized(), b"plan")

    def test_refitted_invalidates_or_replaces_bytes(self):
        engine = _FakeEngine()
        handle = EngineHandle(serialized=b"old", cuda_engine=engine)
        handle.refitted()
        self.assertEqual(handle.serialized(), b"plan")
        handle.refitted(b"refit")
        self.assertEqual(handle.serialized(), b"refit")
        with self.assertRaises(RuntimeError):
            EngineHandle(serialized=b"x").refitted()

    def test_deepcopy_shares_and_pickle_uses_bytes(self):
        handle = EngineHandle(cuda_engine=_FakeEngine())
        self.assertIs(copy.deepcopy(handle), handle)
        restored = pickle.loads(pickle.dumps(handle))
        self.assertFalse(restored.has_cuda_engine)
        self.assertEqual(restored.serialized(), b"plan")


class TestLiveEngineWrap(TestCase):
    def _compile_add(self):
        class Add(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.add(a, b)

        model = Add().eval().cuda()
        inputs = (
            torch.randn((2, 4), device="cuda"),
            torch.randn((2, 4), device="cuda"),
        )
        compiled = torchtrt.compile(
            model,
            inputs=list(inputs),
            ir="dynamo",
            min_block_size=1,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        trt_mod = next(
            m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)
        )
        return model, compiled, trt_mod, inputs

    def tearDown(self):
        torch._dynamo.reset()

    def test_compile_engine_state_per_runtime(self):
        model, compiled, trt_mod, inputs = self._compile_add()
        handle = trt_mod._engine_handle
        self.assertIsNotNone(handle)

        if ENABLED_FEATURES.torch_tensorrt_runtime:
            # C++ runtime deserialized its own copy; the builder engine is released.
            self.assertFalse(handle.has_cuda_engine)
            self.assertTrue(handle.has_serialized)
        else:
            # Python runtime adopted the builder engine; nothing serialized yet.
            self.assertTrue(handle.has_cuda_engine)
            self.assertFalse(handle.has_serialized)
            self.assertIs(trt_mod.engine._engine_handle, handle)

        cos_sim = cosine_similarity(model(*inputs), compiled(*inputs))
        self.assertGreater(cos_sim, COSINE_THRESHOLD)

    def test_state_dict_round_trip(self):
        model, compiled, trt_mod, inputs = self._compile_add()
        packed = trt_mod.get_extra_state()
        self.assertIsNotNone(packed[1])
        self.assertTrue(trt_mod._engine_handle.has_serialized)

        buffer = io.BytesIO()
        torch.save(compiled.state_dict(), buffer)
        buffer.seek(0)
        compiled.load_state_dict(torch.load(buffer, weights_only=False))

        cos_sim = cosine_similarity(model(*inputs), compiled(*inputs))
        self.assertGreater(cos_sim, COSINE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
