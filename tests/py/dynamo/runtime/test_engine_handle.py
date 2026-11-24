import copy
import io
import pickle
import unittest
from unittest import mock

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import _conversion
from torch_tensorrt.dynamo.conversion._conversion import SerializedInterpreterResult
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

    def test_foreign_live_engine_is_reloaded_from_bytes(self):
        # e.g. a builder outside this process returns a bytes-only proxy engine.
        runtime = mock.Mock()
        runtime.deserialize_cuda_engine.return_value = sentinel = object()
        handle = EngineHandle(cuda_engine=_FakeEngine())
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

    def test_rejects_non_bytes(self):
        with self.assertRaises(TypeError):
            EngineHandle(serialized="plan")  # type: ignore[arg-type]

    def test_failed_deserialize_raises(self):
        runtime = mock.Mock()
        runtime.deserialize_cuda_engine.return_value = None
        with self.assertRaises(RuntimeError):
            EngineHandle(serialized=b"corrupt").cuda_engine(runtime)

    def test_repr_names_held_forms(self):
        handle = EngineHandle(cuda_engine=_FakeEngine())
        self.assertEqual(repr(handle), "EngineHandle(live)")
        handle.serialized()
        self.assertEqual(repr(handle), "EngineHandle(bytes+live)")


def _fake_interpreter_result(engine: EngineHandle) -> SerializedInterpreterResult:
    return SerializedInterpreterResult(
        engine=engine,
        input_names=["x"],
        output_names=["y"],
        requires_output_allocator=False,
        symbolic_shape_expressions={},
        requires_native_multidevice=False,
    )


class TestConvertModuleEngineForm(TestCase):
    """Which builds may keep the live builder engine (no TensorRT build needed)."""

    def _convert(self, **settings_kwargs):
        handle = EngineHandle(cuda_engine=_FakeEngine())
        settings = CompilationSettings(lazy_engine_init=True, **settings_kwargs)
        with mock.patch.object(
            _conversion,
            "interpret_module_to_result",
            return_value=_fake_interpreter_result(handle),
        ):
            trt_mod = _conversion.convert_module(None, [], settings, name="m")
        self.assertIs(trt_mod._engine_handle, handle)
        return handle

    def test_default_build_keeps_live_engine(self):
        handle = self._convert()
        self.assertTrue(handle.has_cuda_engine)
        self.assertFalse(handle.has_serialized)

    def test_weight_streaming_keeps_bytes_only(self):
        # A builder engine has no streaming session; budgets need a deserialized one.
        handle = self._convert(enable_weight_streaming=True)
        self.assertFalse(handle.has_cuda_engine)
        self.assertEqual(handle.serialized(), b"plan")

    def test_cross_compile_keeps_bytes_only(self):
        handle = self._convert(enable_cross_compile_for_windows=True)
        self.assertFalse(handle.has_cuda_engine)
        self.assertEqual(handle.serialized(), b"plan")


class TestTorchTensorRTModuleEngineState(TestCase):
    """Module-level engine bookkeeping that needs no engine setup."""

    def _module(self, **kwargs):
        return TorchTensorRTModule(
            input_binding_names=["x"],
            output_binding_names=["y"],
            settings=CompilationSettings(lazy_engine_init=True),
            **kwargs,
        )

    def test_rejects_bytes_and_handle_together(self):
        with self.assertRaises(ValueError):
            self._module(
                serialized_engine=b"plan",
                engine_handle=EngineHandle(serialized=b"plan"),
            )

    def test_serialized_engine_property_round_trip(self):
        trt_mod = self._module()
        self.assertIsNone(trt_mod.serialized_engine)
        with self.assertRaises(RuntimeError):
            trt_mod._pack_engine_info()

        trt_mod.serialized_engine = b"plan"
        self.assertEqual(trt_mod._engine_handle.serialized(), b"plan")
        trt_mod.serialized_engine = None
        self.assertIsNone(trt_mod._engine_handle)

    def test_set_extra_state_without_engine_clears_handle(self):
        trt_mod = self._module(serialized_engine=b"plan")
        trt_mod.set_extra_state(("m", None, ["x"], ["y"]))
        self.assertIsNone(trt_mod._engine_handle)
        self.assertIsNone(trt_mod.serialized_engine)

    def test_legacy_pickle_state_migrates_bytes(self):
        # Whole-module pickles from before EngineHandle stored raw bytes.
        state = self._module().__getstate__()
        state.pop("_engine_handle")
        state["serialized_engine"] = b"legacy-plan"

        restored = TorchTensorRTModule.__new__(TorchTensorRTModule)
        restored.__setstate__(state)
        self.assertNotIn("serialized_engine", restored.__dict__)
        self.assertEqual(restored.serialized_engine, b"legacy-plan")


class TestLiveEngineWrap(TestCase):
    def _compile(self, model, inputs, **compile_kwargs):
        compiled = torchtrt.compile(
            model,
            inputs=list(inputs),
            ir="dynamo",
            min_block_size=1,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
            **compile_kwargs,
        )
        trt_mod = next(
            m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)
        )
        return compiled, trt_mod

    def _compile_add(self, **compile_kwargs):
        class Add(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.add(a, b)

        model = Add().eval().cuda()
        inputs = (
            torch.randn((2, 4), device="cuda"),
            torch.randn((2, 4), device="cuda"),
        )
        compiled, trt_mod = self._compile(model, inputs, **compile_kwargs)
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

    def test_version_compatible_build_reuses_builder_bytes(self):
        # This path builds bytes first; they are kept rather than re-serialized.
        model, compiled, trt_mod, inputs = self._compile_add(version_compatible=True)
        handle = trt_mod._engine_handle
        self.assertTrue(handle.has_serialized)
        if not ENABLED_FEATURES.torch_tensorrt_runtime:
            self.assertTrue(handle.has_cuda_engine)
            self.assertEqual(trt_mod.engine.serialized_engine, handle.serialized())

        cos_sim = cosine_similarity(model(*inputs), compiled(*inputs))
        self.assertGreater(cos_sim, COSINE_THRESHOLD)

    def test_weight_streaming_budget_can_be_set(self):
        # Regression: a builder engine rejects setWeightStreamingBudgetV2, so a
        # weight-streaming build must run on a deserialized engine.
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
            )
            .eval()
            .cuda()
        )
        inputs = (torch.randn((8, 256), device="cuda"),)
        compiled, trt_mod = self._compile(model, inputs, enable_weight_streaming=True)

        streamable = trt_mod.get_streamable_device_memory_budget()
        self.assertGreater(streamable, 0)
        budget = streamable // 2
        self.assertEqual(trt_mod.set_device_memory_budget(budget), budget)

        cos_sim = cosine_similarity(model(*inputs), compiled(*inputs))
        self.assertGreater(cos_sim, COSINE_THRESHOLD)

    def test_proxy_engine_runs_from_bytes(self):
        # A builder outside this process may return a proxy that only carries the
        # engine bytes. The runtime must load a local engine from them, not run it.
        model, _, built, inputs = self._compile_add()

        class ProxyEngine:
            def __init__(self, serialized: bytes) -> None:
                self._serialized = serialized

            def serialize(self) -> bytes:
                return self._serialized

            def __getattr__(self, name):
                raise AssertionError(f"proxy engine used as a local engine: {name}")

        trt_mod = TorchTensorRTModule(
            engine_handle=EngineHandle(
                cuda_engine=ProxyEngine(built.serialized_engine)
            ),
            input_binding_names=built.input_binding_names,
            output_binding_names=built.output_binding_names,
            name="proxy",
            settings=built.settings,
        )
        self.assertFalse(isinstance(trt_mod._engine_handle._cuda_engine, ProxyEngine))
        torch.testing.assert_close(trt_mod(*inputs), model(*inputs))

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
